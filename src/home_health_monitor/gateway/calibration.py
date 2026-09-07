from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
import statistics
from typing import Iterable

from ..contracts import InputError
from .contracts import FeaturePacket, SIGNAL_NAMES


MIN_ELAPSED_HOURS = 48.0
MIN_VALID_COVERAGE = 0.80
MIN_LOW_MOTION_HOURS = 8.0
MIN_QUALITY = 0.80
MAD_MULTIPLIER = 1.4826
SCALE_FLOORS = {
    "heart_rate_bpm": 3.0,
    "spo2_percent": 0.5,
    "temperature_c": 0.1,
    "motion_intensity": 0.05,
}


@dataclass(frozen=True, slots=True)
class CalibrationSignal:
    median: float
    mad_scale: float


@dataclass(frozen=True, slots=True)
class CalibrationProfile:
    schema_version: int
    subject_id: str
    feature_manifest_id: str
    sensor_config_id: str
    temperature_type: str
    temperature_site: str
    started_at: datetime
    ready_at: datetime
    expected_interval_seconds: float
    packet_count: int
    coverage: float
    low_motion_hours: float
    signals: dict[str, CalibrationSignal]


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    status: str
    coverage: float
    elapsed_hours: float
    low_motion_hours: float
    valid_packet_count: int
    expected_packet_count: int
    reason_codes: tuple[str, ...]
    profile: CalibrationProfile | None


def profile_to_dict(profile: CalibrationProfile | None) -> dict[str, object]:
    if profile is None:
        raise InputError("a ready calibration profile is required")
    return {
        "schema_version": profile.schema_version,
        "subject_id": profile.subject_id,
        "feature_manifest_id": profile.feature_manifest_id,
        "sensor_config_id": profile.sensor_config_id,
        "temperature_type": profile.temperature_type,
        "temperature_site": profile.temperature_site,
        "started_at": profile.started_at.isoformat(),
        "ready_at": profile.ready_at.isoformat(),
        "expected_interval_seconds": profile.expected_interval_seconds,
        "packet_count": profile.packet_count,
        "coverage": profile.coverage,
        "low_motion_hours": profile.low_motion_hours,
        "signals": {
            name: {"median": signal.median, "mad_scale": signal.mad_scale}
            for name, signal in profile.signals.items()
        },
    }


def profile_from_dict(value: dict[str, object]) -> CalibrationProfile:
    try:
        if value["schema_version"] != 1:
            raise InputError("unsupported calibration profile schema_version")
        raw_signals = value["signals"]
        if not isinstance(raw_signals, dict) or set(raw_signals) != set(SIGNAL_NAMES):
            raise InputError("calibration profile signals do not match the gateway")
        signals = {}
        for name in SIGNAL_NAMES:
            item = raw_signals[name]
            if not isinstance(item, dict):
                raise InputError(f"invalid calibration signal {name}")
            median = float(item["median"])
            scale = float(item["mad_scale"])
            if not math.isfinite(median) or not math.isfinite(scale) or scale <= 0:
                raise InputError(f"invalid calibration signal {name}")
            signals[name] = CalibrationSignal(median, scale)
        started_at = datetime.fromisoformat(str(value["started_at"]))
        ready_at = datetime.fromisoformat(str(value["ready_at"]))
        if started_at.tzinfo is None or ready_at.tzinfo is None:
            raise InputError("calibration timestamps must include a timezone")
        return CalibrationProfile(
            schema_version=1,
            subject_id=str(value["subject_id"]),
            feature_manifest_id=str(value["feature_manifest_id"]),
            sensor_config_id=str(value["sensor_config_id"]),
            temperature_type=str(value["temperature_type"]),
            temperature_site=str(value["temperature_site"]),
            started_at=started_at,
            ready_at=ready_at,
            expected_interval_seconds=float(value["expected_interval_seconds"]),
            packet_count=int(value["packet_count"]),
            coverage=float(value["coverage"]),
            low_motion_hours=float(value["low_motion_hours"]),
            signals=signals,
        )
    except InputError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise InputError(f"invalid calibration profile: {exc}") from exc


def _is_valid_normal(packet: FeaturePacket) -> bool:
    quality = packet.quality
    return (
        packet.wearable.decision == "normal"
        and quality.ppg >= MIN_QUALITY
        and quality.spo2 >= MIN_QUALITY
        and quality.temperature >= MIN_QUALITY
        and quality.motion >= MIN_QUALITY
    )


def _ensure_compatible(packets: list[FeaturePacket]) -> None:
    first = packets[0]
    checks = (
        ("subject_id", "same subject"),
        ("feature_manifest_id", "same feature manifest"),
        ("sensor_config_id", "same sensor configuration"),
        ("temperature_type", "same temperature type"),
        ("temperature_site", "same temperature site"),
    )
    for field, description in checks:
        expected = getattr(first, field)
        if any(getattr(packet, field) != expected for packet in packets[1:]):
            raise InputError(f"calibration packets must use the {description}")


def _signal_summary(values: list[float], name: str) -> CalibrationSignal:
    median = float(statistics.median(values))
    mad = float(statistics.median(abs(value - median) for value in values))
    return CalibrationSignal(median, max(MAD_MULTIPLIER * mad, SCALE_FLOORS[name]))


def build_calibration(
    packets: Iterable[FeaturePacket], *, expected_interval_seconds: float = 60.0
) -> CalibrationResult:
    if not math.isfinite(expected_interval_seconds) or expected_interval_seconds <= 0:
        raise InputError("expected_interval_seconds must be positive and finite")

    ordered = sorted(packets, key=lambda packet: packet.timestamp)
    if not ordered:
        return CalibrationResult(
            status="collecting",
            coverage=0.0,
            elapsed_hours=0.0,
            low_motion_hours=0.0,
            valid_packet_count=0,
            expected_packet_count=1,
            reason_codes=(
                "need_48_elapsed_hours",
                "need_80_percent_valid_coverage",
                "need_eight_low_motion_hours",
            ),
            profile=None,
        )

    _ensure_compatible(ordered)
    started_at = ordered[0].timestamp
    ready_at = ordered[-1].timestamp
    elapsed_seconds = max(0.0, (ready_at - started_at).total_seconds())
    elapsed_hours = elapsed_seconds / 3600.0
    expected_count = math.floor(elapsed_seconds / expected_interval_seconds) + 1

    valid = [packet for packet in ordered if _is_valid_normal(packet)]
    occupied_slots = {
        min(
            expected_count - 1,
            math.floor((packet.timestamp - started_at).total_seconds() / expected_interval_seconds),
        )
        for packet in valid
    }
    valid_count = len(occupied_slots)
    coverage = min(1.0, valid_count / expected_count)
    low_motion_count = sum(packet.motion == 0 for packet in valid)
    low_motion_hours = low_motion_count * expected_interval_seconds / 3600.0

    reasons = []
    if elapsed_hours < MIN_ELAPSED_HOURS:
        reasons.append("need_48_elapsed_hours")
    if coverage < MIN_VALID_COVERAGE:
        reasons.append("need_80_percent_valid_coverage")
    if low_motion_hours < MIN_LOW_MOTION_HOURS:
        reasons.append("need_eight_low_motion_hours")

    profile = None
    if not reasons:
        low_motion = [packet for packet in valid if packet.motion == 0]
        signals = {}
        for name in SIGNAL_NAMES:
            source = valid if name == "motion_intensity" else low_motion
            signals[name] = _signal_summary(
                [float(getattr(packet, name)) for packet in source], name
            )
        first = ordered[0]
        profile = CalibrationProfile(
            schema_version=1,
            subject_id=first.subject_id,
            feature_manifest_id=first.feature_manifest_id,
            sensor_config_id=first.sensor_config_id,
            temperature_type=first.temperature_type,
            temperature_site=first.temperature_site,
            started_at=started_at,
            ready_at=ready_at,
            expected_interval_seconds=expected_interval_seconds,
            packet_count=len(valid),
            coverage=coverage,
            low_motion_hours=low_motion_hours,
            signals=signals,
        )

    return CalibrationResult(
        status="ready" if profile else "collecting",
        coverage=coverage,
        elapsed_hours=elapsed_hours,
        low_motion_hours=low_motion_hours,
        valid_packet_count=valid_count,
        expected_packet_count=expected_count,
        reason_codes=tuple(reasons),
        profile=profile,
    )


def robust_deviations(
    packet: FeaturePacket, profile: CalibrationProfile | None
) -> dict[str, float]:
    if profile is None:
        raise InputError("a ready calibration profile is required")
    if packet.subject_id != profile.subject_id:
        raise InputError("packet and calibration profile must use the same subject")
    if packet.feature_manifest_id != profile.feature_manifest_id:
        raise InputError("packet and calibration profile must use the same feature manifest")
    if packet.sensor_config_id != profile.sensor_config_id:
        raise InputError("packet and calibration profile must use the same sensor configuration")
    if (
        packet.temperature_type != profile.temperature_type
        or packet.temperature_site != profile.temperature_site
    ):
        raise InputError("packet and calibration profile must use the same temperature measurement")
    return {
        name: (float(getattr(packet, name)) - profile.signals[name].median)
        / profile.signals[name].mad_scale
        for name in SIGNAL_NAMES
    }
