from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Protocol

from .autoencoder import ModelError, ReconstructionResult
from .calibration import (
    CalibrationProfile,
    CalibrationResult,
    build_calibration,
    profile_from_dict,
    profile_to_dict,
    robust_deviations,
)
from .contracts import FeaturePacket
from .store import GatewayStore
from .windowing import build_window


BASELINE_ANOMALY_Z = 4.0
QUALITY_FAILURE_LEVEL = 0.5
QUALITY_FAILURE_COUNT = 3


class AutoencoderProtocol(Protocol):
    model_id: str
    consecutive_windows: int

    def predict(self, window: Any) -> ReconstructionResult: ...


@dataclass(frozen=True, slots=True)
class GatewayDecision:
    decision: str
    triggered_by: tuple[str, ...]
    reason_codes: tuple[str, ...]
    scores: dict[str, float]
    contributing_signals: tuple[str, ...]
    measurements: dict[str, float | int]
    calibration: dict[str, float | str]
    timestamp: datetime

    def to_dict(self) -> dict[str, object]:
        return {
            "decision": self.decision,
            "triggered_by": list(self.triggered_by),
            "reason_codes": list(self.reason_codes),
            "scores": self.scores,
            "contributing_signals": list(self.contributing_signals),
            "measurements": self.measurements,
            "calibration": self.calibration,
            "timestamp": self.timestamp.isoformat().replace("+00:00", "Z"),
        }


class GatewayEngine:
    def __init__(
        self, store: GatewayStore, *, model: AutoencoderProtocol | None = None
    ) -> None:
        self.store = store
        self.model = model

    def _calibration(
        self, packet: FeaturePacket
    ) -> tuple[CalibrationProfile | None, dict[str, float | str]]:
        stored = self.store.load_calibration(packet.subject_id)
        if stored is not None:
            profile = profile_from_dict(stored)
            hours = (profile.ready_at - profile.started_at).total_seconds() / 3600.0
            return profile, {"status": "ready", "hours": round(hours, 2)}

        start = packet.timestamp - timedelta(hours=48)
        packets = self.store.packets_between(packet.subject_id, start, packet.timestamp)
        result = build_calibration(packets)
        if result.profile is not None:
            self.store.save_calibration(packet.subject_id, profile_to_dict(result.profile))
        return result.profile, {
            "status": result.status,
            "hours": round(result.elapsed_hours, 2),
            "coverage": round(result.coverage, 4),
        }

    def _persistent_quality_failures(self, packet: FeaturePacket) -> tuple[str, ...]:
        recent = self.store.packets_between(
            packet.subject_id, packet.timestamp - timedelta(minutes=15), packet.timestamp
        )
        recent = recent[-QUALITY_FAILURE_COUNT:]
        if len(recent) < QUALITY_FAILURE_COUNT:
            return ()
        reasons = []
        for name in ("ppg", "spo2", "temperature", "motion"):
            if all(getattr(item.quality, name) < QUALITY_FAILURE_LEVEL for item in recent):
                reasons.append(f"persistent_{name}_quality_failure")
        return tuple(reasons)

    def _model_is_persistent(
        self, packet: FeaturePacket, result: ReconstructionResult
    ) -> bool:
        if result.is_severe:
            return True
        if not result.is_above_threshold or self.model is None:
            return False
        required_previous = self.model.consecutive_windows - 1
        if required_previous <= 0:
            return True
        previous = self.store.recent_events(packet.subject_id, limit=required_previous)
        if len(previous) < required_previous:
            return False
        last_time = packet.timestamp
        for event in previous:
            try:
                event_time = datetime.fromisoformat(str(event["timestamp"]).replace("Z", "+00:00"))
                error = float(event["scores"]["gateway_model_error"])
            except (KeyError, TypeError, ValueError):
                return False
            if last_time - event_time > timedelta(minutes=10) or error < result.persistent_threshold:
                return False
            last_time = event_time
        return True

    def ingest(self, packet: FeaturePacket) -> GatewayDecision:
        self.store.add_packet(packet)
        profile, calibration = self._calibration(packet)
        triggered = []
        reasons = []
        contributing = set()
        scores = {"wearable": round(packet.wearable.score, 6)}

        if packet.wearable.decision == "anomaly":
            triggered.append("wearable")
            reasons.extend(packet.wearable.reason_codes)

        quality_reasons = self._persistent_quality_failures(packet)
        if quality_reasons:
            triggered.append("data_quality")
            reasons.extend(quality_reasons)
            contributing.update(reason.removeprefix("persistent_").removesuffix("_quality_failure") for reason in quality_reasons)

        if profile is not None:
            deviations = robust_deviations(packet, profile)
            scores["gateway_baseline_max_abs_z"] = round(
                max(abs(value) for value in deviations.values()), 6
            )
            baseline_signals = tuple(
                name for name, value in deviations.items() if abs(value) >= BASELINE_ANOMALY_Z
            )
            if baseline_signals:
                triggered.append("gateway_baseline")
                reasons.extend(f"{name}_personal_deviation" for name in baseline_signals)
                contributing.update(baseline_signals)

            if self.model is not None:
                start = packet.timestamp - timedelta(hours=24)
                packets = self.store.packets_between(packet.subject_id, start, packet.timestamp)
                try:
                    model_result = self.model.predict(build_window(packets, profile, packet.timestamp))
                except ModelError:
                    model_result = None
                if model_result is not None:
                    scores["gateway_model_error"] = round(model_result.overall_error, 8)
                    if self._model_is_persistent(packet, model_result):
                        triggered.append("gateway_autoencoder")
                        reasons.append(
                            "severe_reconstruction_error"
                            if model_result.is_severe
                            else "persistent_reconstruction_error"
                        )
                        highest = max(model_result.feature_errors.values(), default=0.0)
                        contributing.update(
                            name
                            for name, error in model_result.feature_errors.items()
                            if error == highest and error > 0
                        )

        decision = GatewayDecision(
            decision="anomaly" if triggered else "normal",
            triggered_by=tuple(dict.fromkeys(triggered)),
            reason_codes=tuple(dict.fromkeys(reasons)),
            scores=scores,
            contributing_signals=tuple(sorted(contributing)),
            measurements={
                "heart_rate_bpm": packet.heart_rate_bpm,
                "spo2_percent": packet.spo2_percent,
                "temperature_c": packet.temperature_c,
                "motion": packet.motion,
            },
            calibration=calibration,
            timestamp=packet.timestamp,
        )
        self.store.save_event(packet.subject_id, decision.to_dict(), packet.timestamp)
        return decision
