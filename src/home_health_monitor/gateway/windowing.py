from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
import statistics
from typing import Iterable

from ..contracts import InputError
from .calibration import CalibrationProfile, MIN_QUALITY
from .contracts import FeaturePacket, SIGNAL_NAMES


STEP_COUNT = 288
STEP_MINUTES = 5
QUALITY_FEATURES = (
    "quality_ppg",
    "quality_spo2",
    "quality_temperature",
    "quality_motion",
)
FEATURE_NAMES = SIGNAL_NAMES + QUALITY_FEATURES
SIGNAL_QUALITY = {
    "heart_rate_bpm": "ppg",
    "spo2_percent": "spo2",
    "temperature_c": "temperature",
    "motion_intensity": "motion",
}


@dataclass(frozen=True, slots=True)
class ModelWindow:
    feature_names: tuple[str, ...]
    values: tuple[tuple[float, ...], ...]
    masks: tuple[tuple[float, ...], ...]
    started_at: datetime
    ended_at: datetime
    step_minutes: int


def _aligned_floor(value: datetime, step_minutes: int) -> datetime:
    if value.tzinfo is None:
        raise InputError("window end must include a timezone")
    utc = value.astimezone(timezone.utc)
    minute = utc.minute - utc.minute % step_minutes
    return utc.replace(minute=minute, second=0, microsecond=0)


def _validate_packet(packet: FeaturePacket, profile: CalibrationProfile) -> None:
    checks = (
        (packet.subject_id, profile.subject_id, "same subject"),
        (
            packet.feature_manifest_id,
            profile.feature_manifest_id,
            "same feature manifest",
        ),
        (packet.sensor_config_id, profile.sensor_config_id, "same sensor configuration"),
        (packet.temperature_type, profile.temperature_type, "same temperature type"),
        (packet.temperature_site, profile.temperature_site, "same temperature site"),
    )
    for actual, expected, description in checks:
        if actual != expected:
            raise InputError(f"window packets and calibration profile must use the {description}")


def _median(values: list[float]) -> float:
    return float(statistics.median(values))


def build_window(
    packets: Iterable[FeaturePacket],
    profile: CalibrationProfile,
    end: datetime,
    *,
    step_minutes: int = STEP_MINUTES,
) -> ModelWindow:
    if isinstance(step_minutes, bool) or not isinstance(step_minutes, int) or step_minutes <= 0:
        raise InputError("step_minutes must be a positive integer")

    final_bin = _aligned_floor(end, step_minutes)
    ended_at = final_bin + timedelta(minutes=step_minutes)
    started_at = ended_at - timedelta(minutes=step_minutes * STEP_COUNT)
    step_seconds = step_minutes * 60
    bins: list[list[FeaturePacket]] = [[] for _ in range(STEP_COUNT)]

    for packet in packets:
        _validate_packet(packet, profile)
        if not started_at <= packet.timestamp < ended_at:
            continue
        index = math.floor((packet.timestamp - started_at).total_seconds() / step_seconds)
        bins[index].append(packet)

    all_values = []
    all_masks = []
    for packets_in_bin in bins:
        values = []
        masks = []
        for name in SIGNAL_NAMES:
            quality_name = SIGNAL_QUALITY[name]
            accepted = [
                packet
                for packet in packets_in_bin
                if getattr(packet.quality, quality_name) >= MIN_QUALITY
            ]
            if accepted:
                observed = _median([float(getattr(packet, name)) for packet in accepted])
                signal = profile.signals[name]
                values.append((observed - signal.median) / signal.mad_scale)
                masks.append(1.0)
            else:
                values.append(0.0)
                masks.append(0.0)

        for quality_name in ("ppg", "spo2", "temperature", "motion"):
            if packets_in_bin:
                values.append(
                    _median(
                        [float(getattr(packet.quality, quality_name)) for packet in packets_in_bin]
                    )
                )
                masks.append(1.0)
            else:
                values.append(0.0)
                masks.append(0.0)

        all_values.append(tuple(values))
        all_masks.append(tuple(masks))

    return ModelWindow(
        feature_names=FEATURE_NAMES,
        values=tuple(all_values),
        masks=tuple(all_masks),
        started_at=started_at,
        ended_at=ended_at,
        step_minutes=step_minutes,
    )
