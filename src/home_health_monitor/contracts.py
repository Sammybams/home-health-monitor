from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Any


class InputError(ValueError):
    """A safe, client-visible validation error."""


@dataclass(frozen=True, slots=True)
class Observation:
    timestamp: datetime
    body_temperature_c: float
    ambient_temperature_c: float
    heart_rate_bpm: float
    motion: int
    resting: bool | None = None


@dataclass(frozen=True, slots=True)
class BaselineSignal:
    median: float
    mad_scale: float


@dataclass(frozen=True, slots=True)
class BaselineProfile:
    schema_version: int
    subject_id: str
    created_at: datetime
    healthy_days: int
    sample_count: int
    signals: dict[str, BaselineSignal]


@dataclass(frozen=True, slots=True)
class PredictionRequest:
    observations: tuple[Observation, ...]
    subject_id: str | None = None
    baseline: BaselineProfile | None = None


SENSOR_RANGES = {
    "body_temperature_c": (25.0, 45.0),
    "ambient_temperature_c": (-20.0, 60.0),
    "heart_rate_bpm": (20.0, 250.0),
}

BASELINE_SIGNALS = {
    "resting_body_temperature_c",
    "resting_heart_rate_bpm",
    "resting_body_ambient_delta_c",
    "daily_motion_fraction",
}


def _number(record: dict[str, Any], field: str) -> float:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputError(f"{field} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise InputError(f"{field} must be finite")
    low, high = SENSOR_RANGES[field]
    if not low <= result <= high:
        raise InputError(f"{field} is outside the accepted sensor range")
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


def _baseline_signal(name: str, value: Any) -> BaselineSignal:
    if not isinstance(value, dict) or set(value) != {"median", "mad_scale"}:
        raise InputError(f"baseline signal {name} must contain median and mad_scale")
    median = value["median"]
    scale = value["mad_scale"]
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in (median, scale)):
        raise InputError(f"baseline signal {name} values must be numbers")
    median, scale = float(median), float(scale)
    if not math.isfinite(median) or not math.isfinite(scale) or scale <= 0:
        raise InputError(f"baseline signal {name} must have finite values and a positive mad_scale")
    return BaselineSignal(median, scale)


def parse_baseline(value: Any) -> BaselineProfile:
    if not isinstance(value, dict):
        raise InputError("baseline must be an object")
    required = {"schema_version", "subject_id", "created_at", "healthy_days", "sample_count", "signals"}
    if set(value) != required:
        raise InputError("baseline fields do not match schema version 1")
    if value["schema_version"] != 1:
        raise InputError("baseline schema_version must be 1")
    subject_id = value["subject_id"]
    if not isinstance(subject_id, str) or not subject_id or len(subject_id) > 128:
        raise InputError("baseline subject_id must be a non-empty string of at most 128 characters")
    healthy_days = value["healthy_days"]
    sample_count = value["sample_count"]
    if isinstance(healthy_days, bool) or not isinstance(healthy_days, int) or not 7 <= healthy_days <= 30:
        raise InputError("baseline healthy_days must be between 7 and 30")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 84:
        raise InputError("baseline sample_count must be at least 84")
    raw_signals = value["signals"]
    if not isinstance(raw_signals, dict) or set(raw_signals) != BASELINE_SIGNALS:
        raise InputError("baseline signals do not match schema version 1")
    return BaselineProfile(
        schema_version=1,
        subject_id=subject_id,
        created_at=_timestamp(value["created_at"]),
        healthy_days=healthy_days,
        sample_count=sample_count,
        signals={name: _baseline_signal(name, raw_signals[name]) for name in sorted(BASELINE_SIGNALS)},
    )


def parse_request(payload: Any) -> PredictionRequest:
    if not isinstance(payload, dict):
        raise InputError("request body must be a JSON object")
    allowed = {"subject_id", "observations", "baseline"}
    unknown = set(payload) - allowed
    if unknown:
        raise InputError(f"unknown request fields: {', '.join(sorted(unknown))}")

    subject_id = payload.get("subject_id")
    if subject_id is not None and (not isinstance(subject_id, str) or not subject_id or len(subject_id) > 128):
        raise InputError("subject_id must be a non-empty string of at most 128 characters")
    baseline = parse_baseline(payload["baseline"]) if payload.get("baseline") is not None else None
    if baseline is not None and subject_id is None:
        raise InputError("subject_id is required when baseline is provided")
    if baseline is not None and baseline.subject_id != subject_id:
        raise InputError("baseline subject_id does not match request subject_id")

    raw = payload.get("observations")
    if not isinstance(raw, list):
        raise InputError("observations must be an array")
    if not 12 <= len(raw) <= 10_000:
        raise InputError("observations must contain between 12 and 10000 samples")

    observations: list[Observation] = []
    previous: datetime | None = None
    required = {"timestamp", *SENSOR_RANGES, "motion"}
    allowed_observation = required | {"resting"}
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise InputError(f"observations[{index}] must be an object")
        missing = required - set(item)
        if missing:
            raise InputError(f"observations[{index}] is missing: {', '.join(sorted(missing))}")
        unknown_item = set(item) - allowed_observation
        if unknown_item:
            raise InputError(f"observations[{index}] has unknown fields")
        timestamp = _timestamp(item["timestamp"])
        if previous is not None and timestamp <= previous:
            raise InputError("observation timestamps must be strictly increasing")
        motion = item["motion"]
        if isinstance(motion, bool):
            motion = int(motion)
        if motion not in (0, 1):
            raise InputError("motion must be 0 or 1")
        resting = item.get("resting")
        if isinstance(resting, int) and not isinstance(resting, bool) and resting in (0, 1):
            resting = bool(resting)
        if resting is not None and not isinstance(resting, bool):
            raise InputError("resting must be true, false, 0 or 1 when provided")
        observations.append(Observation(
            timestamp=timestamp,
            body_temperature_c=_number(item, "body_temperature_c"),
            ambient_temperature_c=_number(item, "ambient_temperature_c"),
            heart_rate_bpm=_number(item, "heart_rate_bpm"),
            motion=int(motion),
            resting=resting,
        ))
        previous = timestamp

    span_seconds = (observations[-1].timestamp - observations[0].timestamp).total_seconds()
    if span_seconds < 30 * 60:
        raise InputError("observations must span at least 30 minutes")
    return PredictionRequest(tuple(observations), subject_id, baseline)
