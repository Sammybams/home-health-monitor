from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Any

from ..contracts import InputError


SIGNAL_NAMES = (
    "heart_rate_bpm",
    "spo2_percent",
    "temperature_c",
    "motion_intensity",
)
QUALITY_NAMES = ("ppg", "spo2", "temperature", "motion")
PACKET_FIELDS = {
    "schema_version",
    "subject_id",
    "device_id",
    "sequence",
    "timestamp",
    "sample_duration_seconds",
    "firmware_version",
    "sensor_config_id",
    "feature_manifest_id",
    "heart_rate_bpm",
    "spo2_percent",
    "temperature_c",
    "temperature_type",
    "temperature_site",
    "motion_intensity",
    "motion",
    "quality",
    "wearable",
}


@dataclass(frozen=True, slots=True)
class Quality:
    ppg: float
    spo2: float
    temperature: float
    motion: float


@dataclass(frozen=True, slots=True)
class WearableAssessment:
    decision: str
    score: float
    deviations: dict[str, float]
    reason_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FeaturePacket:
    schema_version: int
    subject_id: str
    device_id: str
    sequence: int
    timestamp: datetime
    sample_duration_seconds: float
    firmware_version: str
    sensor_config_id: str
    feature_manifest_id: str
    heart_rate_bpm: float
    spo2_percent: float
    temperature_c: float
    temperature_type: str
    temperature_site: str
    motion_intensity: float
    motion: int
    quality: Quality
    wearable: WearableAssessment


def _object(value: Any, name: str, fields: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise InputError(f"{name} must be an object")
    missing = fields - set(value)
    if missing:
        raise InputError(f"missing {name} fields: {', '.join(sorted(missing))}")
    unknown = set(value) - fields
    if unknown:
        raise InputError(f"unknown {name} fields: {', '.join(sorted(unknown))}")
    return value


def _identifier(value: Any, name: str, *, max_length: int = 128) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > max_length:
        raise InputError(f"{name} must be a non-empty string of at most {max_length} characters")
    return value


def _number(value: Any, name: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise InputError(f"{name} must be finite")
    if not low <= result <= high:
        raise InputError(f"{name} must be between {low:g} and {high:g}")
    return result


def _timestamp(value: Any) -> datetime:
    if not isinstance(value, str):
        raise InputError("timestamp must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise InputError("timestamp must be valid ISO-8601") from exc
    if parsed.tzinfo is None:
        raise InputError("timestamp must include a timezone")
    return parsed.astimezone(timezone.utc)


def _quality(value: Any) -> Quality:
    item = _object(value, "quality", set(QUALITY_NAMES))
    values = {
        name: _number(item[name], f"quality.{name}", 0.0, 1.0)
        for name in QUALITY_NAMES
    }
    return Quality(**values)


def _wearable(value: Any) -> WearableAssessment:
    item = _object(value, "wearable", {"decision", "score", "deviations", "reason_codes"})
    decision = item["decision"]
    if decision not in {"normal", "anomaly"}:
        raise InputError("wearable.decision must be normal or anomaly")
    score = _number(item["score"], "wearable.score", 0.0, 1_000_000.0)
    raw_deviations = _object(item["deviations"], "wearable deviations", set(SIGNAL_NAMES))
    deviations = {}
    for name in SIGNAL_NAMES:
        raw = raw_deviations[name]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(float(raw)):
            raise InputError(f"wearable deviation {name} must be finite")
        deviations[name] = float(raw)
    raw_reasons = item["reason_codes"]
    if not isinstance(raw_reasons, list) or len(raw_reasons) > 16:
        raise InputError("wearable.reason_codes must be an array of at most 16 strings")
    reasons = tuple(
        _identifier(reason, "wearable reason code", max_length=64)
        for reason in raw_reasons
    )
    if len(set(reasons)) != len(reasons):
        raise InputError("wearable.reason_codes must not contain duplicates")
    if decision == "anomaly" and not reasons:
        raise InputError("anomaly decision requires a reason code")
    return WearableAssessment(decision, score, deviations, reasons)


def parse_packet(payload: object) -> FeaturePacket:
    item = _object(payload, "packet", PACKET_FIELDS)
    if item["schema_version"] != 1:
        raise InputError("packet schema_version must be 1")

    sequence = item["sequence"]
    if isinstance(sequence, bool) or not isinstance(sequence, int) or not 0 <= sequence < 2**63:
        raise InputError("sequence must be an integer between 0 and 9223372036854775807")

    motion = item["motion"]
    if isinstance(motion, bool) or motion not in (0, 1):
        raise InputError("motion must be 0 or 1")

    temperature_type = item["temperature_type"]
    if temperature_type not in {"skin", "core", "surface"}:
        raise InputError("temperature_type must be skin, core or surface")

    return FeaturePacket(
        schema_version=1,
        subject_id=_identifier(item["subject_id"], "subject_id"),
        device_id=_identifier(item["device_id"], "device_id"),
        sequence=sequence,
        timestamp=_timestamp(item["timestamp"]),
        sample_duration_seconds=_number(
            item["sample_duration_seconds"], "sample_duration_seconds", 2.0, 5.0
        ),
        firmware_version=_identifier(item["firmware_version"], "firmware_version", max_length=64),
        sensor_config_id=_identifier(item["sensor_config_id"], "sensor_config_id", max_length=64),
        feature_manifest_id=_identifier(
            item["feature_manifest_id"], "feature_manifest_id", max_length=64
        ),
        heart_rate_bpm=_number(item["heart_rate_bpm"], "heart_rate_bpm", 20.0, 250.0),
        spo2_percent=_number(item["spo2_percent"], "spo2_percent", 0.0, 100.0),
        temperature_c=_number(item["temperature_c"], "temperature_c", 10.0, 45.0),
        temperature_type=temperature_type,
        temperature_site=_identifier(item["temperature_site"], "temperature_site", max_length=64),
        motion_intensity=_number(item["motion_intensity"], "motion_intensity", 0.0, 200.0),
        motion=motion,
        quality=_quality(item["quality"]),
        wearable=_wearable(item["wearable"]),
    )
