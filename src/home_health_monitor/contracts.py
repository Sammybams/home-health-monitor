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


@dataclass(frozen=True, slots=True)
class PredictionRequest:
    observations: tuple[Observation, ...]
    subject_id: str | None = None


SENSOR_RANGES = {
    "body_temperature_c": (25.0, 45.0),
    "ambient_temperature_c": (-20.0, 60.0),
    "heart_rate_bpm": (20.0, 250.0),
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


def parse_request(payload: Any) -> PredictionRequest:
    if not isinstance(payload, dict):
        raise InputError("request body must be a JSON object")
    allowed = {"subject_id", "observations"}
    unknown = set(payload) - allowed
    if unknown:
        raise InputError(f"unknown request fields: {', '.join(sorted(unknown))}")

    subject_id = payload.get("subject_id")
    if subject_id is not None and (not isinstance(subject_id, str) or not subject_id or len(subject_id) > 128):
        raise InputError("subject_id must be a non-empty string of at most 128 characters")

    raw = payload.get("observations")
    if not isinstance(raw, list):
        raise InputError("observations must be an array")
    if not 12 <= len(raw) <= 10_000:
        raise InputError("observations must contain between 12 and 10000 samples")

    observations: list[Observation] = []
    previous: datetime | None = None
    required = {"timestamp", *SENSOR_RANGES, "motion"}
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise InputError(f"observations[{index}] must be an object")
        missing = required - set(item)
        if missing:
            raise InputError(f"observations[{index}] is missing: {', '.join(sorted(missing))}")
        unknown_item = set(item) - required
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
        observations.append(Observation(
            timestamp=timestamp,
            body_temperature_c=_number(item, "body_temperature_c"),
            ambient_temperature_c=_number(item, "ambient_temperature_c"),
            heart_rate_bpm=_number(item, "heart_rate_bpm"),
            motion=int(motion),
        ))
        previous = timestamp

    span_seconds = (observations[-1].timestamp - observations[0].timestamp).total_seconds()
    if span_seconds < 30 * 60:
        raise InputError("observations must span at least 30 minutes")
    return PredictionRequest(tuple(observations), subject_id)
