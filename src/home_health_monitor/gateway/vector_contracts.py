from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from typing import Any

from home_health_monitor.contracts import InputError
from home_health_monitor.datasets.pulse_transit_build import FEATURE_MANIFEST_ID
from home_health_monitor.datasets.pulse_transit_features import DATASET_FEATURE_NAMES


@dataclass(frozen=True, slots=True)
class VectorInterval:
    subject_id: str
    device_id: str
    sequence: int
    interval_start: datetime
    interval_end: datetime
    vectors: tuple[dict[str, float], ...]


def _timestamp(value: Any, name: str) -> datetime:
    if not isinstance(value, str):
        raise InputError(f"{name} must be an ISO-8601 timestamp")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise InputError(f"{name} must be an ISO-8601 timestamp") from exc
    if result.tzinfo is None:
        raise InputError(f"{name} must include a timezone")
    return result


def parse_vector_interval(payload: Any) -> VectorInterval:
    if not isinstance(payload, dict):
        raise InputError("request body must be an object")
    expected = {
        "schema_version", "subject_id", "device_id", "sequence",
        "interval_start", "interval_end", "feature_manifest_id", "vectors",
    }
    if set(payload) != expected:
        raise InputError("interval fields do not match the V2 contract")
    if payload["schema_version"] != 1:
        raise InputError("unsupported schema_version")
    if payload["feature_manifest_id"] != FEATURE_MANIFEST_ID:
        raise InputError("feature_manifest_id does not match the V2 model")
    subject_id, device_id = payload["subject_id"], payload["device_id"]
    if not isinstance(subject_id, str) or not 1 <= len(subject_id) <= 128:
        raise InputError("subject_id must contain 1 to 128 characters")
    if not isinstance(device_id, str) or not 1 <= len(device_id) <= 128:
        raise InputError("device_id must contain 1 to 128 characters")
    sequence = payload["sequence"]
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        raise InputError("sequence must be a non-negative integer")
    start = _timestamp(payload["interval_start"], "interval_start")
    end = _timestamp(payload["interval_end"], "interval_end")
    duration = (end - start).total_seconds()
    if not 60 <= duration <= 15 * 60:
        raise InputError("interval duration must be between 1 and 15 minutes")
    raw_vectors = payload["vectors"]
    if not isinstance(raw_vectors, list) or not 1 <= len(raw_vectors) <= 32:
        raise InputError("vectors must contain 1 to 32 feature vectors")
    vectors: list[dict[str, float]] = []
    required = set(DATASET_FEATURE_NAMES)
    for index, raw in enumerate(raw_vectors):
        if not isinstance(raw, dict) or set(raw) != required:
            raise InputError(f"vectors[{index}] fields do not match the feature manifest")
        vector: dict[str, float] = {}
        for name in DATASET_FEATURE_NAMES:
            value = raw[name]
            if isinstance(value, bool):
                raise InputError(f"vectors[{index}].{name} must be numeric")
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise InputError(f"vectors[{index}].{name} must be numeric") from exc
            if not math.isfinite(number):
                raise InputError(f"vectors[{index}].{name} must be finite")
            vector[name] = number
        if vector["heart_rate_valid"] not in (0.0, 1.0):
            raise InputError(f"vectors[{index}].heart_rate_valid must be 0 or 1")
        vectors.append(vector)
    return VectorInterval(subject_id, device_id, sequence, start, end, tuple(vectors))
