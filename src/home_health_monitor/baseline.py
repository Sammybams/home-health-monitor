from __future__ import annotations

from datetime import datetime, timezone
from statistics import median
from typing import Iterable

from .contracts import BaselineProfile, BaselineSignal, InputError, Observation, PredictionRequest


SCALE_FLOORS = {
    "resting_body_temperature_c": 0.10,
    "resting_heart_rate_bpm": 3.0,
    "resting_body_ambient_delta_c": 0.20,
    "daily_motion_fraction": 0.05,
}


def is_resting(observation: Observation) -> bool:
    """Use an explicit rest marker when available, otherwise use no motion."""
    if observation.resting is not None:
        return observation.resting
    return observation.motion == 0


def _robust_signal(values: list[float], name: str) -> BaselineSignal:
    center = float(median(values))
    absolute_deviations = [abs(value - center) for value in values]
    # 1.4826 makes MAD comparable to standard deviation for normal data.
    scale = max(float(median(absolute_deviations)) * 1.4826, SCALE_FLOORS[name])
    return BaselineSignal(center, scale)


def build_baseline(
    requests: Iterable[PredictionRequest],
    *,
    created_at: datetime | None = None,
) -> BaselineProfile:
    days = tuple(requests)
    if not 7 <= len(days) <= 30:
        raise InputError("baseline requires between 7 and 30 healthy daily windows")

    subject_ids = {request.subject_id for request in days}
    if None in subject_ids or len(subject_ids) != 1:
        raise InputError("all baseline windows must have the same non-empty subject_id")
    subject_id = days[0].subject_id
    if subject_id is None:  # Kept explicit for type checkers; guarded above.
        raise InputError("baseline subject_id is required")
    if any(request.baseline is not None for request in days):
        raise InputError("baseline source windows cannot contain another baseline")

    window_dates = {request.observations[-1].timestamp.date() for request in days}
    if len(window_dates) != len(days):
        raise InputError("baseline windows must represent distinct UTC dates")

    daily_values: dict[str, list[float]] = {name: [] for name in SCALE_FLOORS}
    sample_count = 0
    for request in days:
        observations = request.observations
        span_hours = (observations[-1].timestamp - observations[0].timestamp).total_seconds() / 3600
        if span_hours < 6:
            raise InputError("each baseline window must span at least 6 hours")
        resting = [item for item in observations if is_resting(item)]
        if len(resting) < 12:
            raise InputError("each baseline window must contain at least 12 resting samples")
        sample_count += len(observations)
        daily_values["resting_body_temperature_c"].append(
            float(median(item.body_temperature_c for item in resting))
        )
        daily_values["resting_heart_rate_bpm"].append(
            float(median(item.heart_rate_bpm for item in resting))
        )
        daily_values["resting_body_ambient_delta_c"].append(
            float(median(item.body_temperature_c - item.ambient_temperature_c for item in resting))
        )
        daily_values["daily_motion_fraction"].append(
            sum(item.motion for item in observations) / len(observations)
        )

    timestamp = created_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        raise InputError("baseline created_at must include a timezone")
    return BaselineProfile(
        schema_version=1,
        subject_id=subject_id,
        created_at=timestamp.astimezone(timezone.utc),
        healthy_days=len(days),
        sample_count=sample_count,
        signals={name: _robust_signal(values, name) for name, values in daily_values.items()},
    )


def baseline_to_dict(profile: BaselineProfile) -> dict:
    return {
        "schema_version": profile.schema_version,
        "subject_id": profile.subject_id,
        "created_at": profile.created_at.isoformat(),
        "healthy_days": profile.healthy_days,
        "sample_count": profile.sample_count,
        "signals": {
            name: {"median": signal.median, "mad_scale": signal.mad_scale}
            for name, signal in profile.signals.items()
        },
    }
