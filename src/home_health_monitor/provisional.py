from __future__ import annotations

from statistics import fmean, median
from typing import Callable

from .baseline import SCALE_FLOORS, is_resting
from .change import CHANGE_THRESHOLD
from .contracts import Observation, PredictionRequest


FUTURE_HORIZON_HOURS = 24
TREND_PROJECTION_HOURS = 6


def _slope(items: list[Observation], value: Callable[[Observation], float]) -> float:
    if len(items) < 3:
        return 0.0
    origin = items[0].timestamp
    xs = [(item.timestamp - origin).total_seconds() / 3600 for item in items]
    ys = [value(item) for item in items]
    x_center, y_center = fmean(xs), fmean(ys)
    denominator = sum((x - x_center) ** 2 for x in xs)
    if denominator == 0:
        return 0.0
    return sum((x - x_center) * (y - y_center) for x, y in zip(xs, ys)) / denominator


def _robust_reference(values: list[float], floor: float) -> tuple[float, float]:
    center = float(median(values))
    scale = max(float(median(abs(value - center) for value in values)) * 1.4826, floor)
    return center, scale


def _classification(score: float) -> str:
    return "higher_risk" if score >= CHANGE_THRESHOLD else "lower_risk"


def _risk_score(deviation: float) -> float:
    # A bounded screening score, deliberately not described as a probability.
    return round(min(max(deviation / (CHANGE_THRESHOLD * 2), 0.0), 1.0), 6)


def provisional_prediction(request: PredictionRequest) -> tuple[dict, list[str]]:
    observations = list(request.observations)
    end = observations[-1].timestamp.timestamp()
    recent = [item for item in observations if item.timestamp.timestamp() >= end - 6 * 3600]
    warnings: list[str] = []
    getters: dict[str, Callable[[Observation], float]] = {
        "resting_body_temperature_c": lambda item: item.body_temperature_c,
        "resting_heart_rate_bpm": lambda item: item.heart_rate_bpm,
        "resting_body_ambient_delta_c": lambda item: item.body_temperature_c - item.ambient_temperature_c,
    }
    if request.baseline is not None:
        recent_resting = [item for item in recent if is_resting(item)]
        if len(recent_resting) >= 3:
            current_items = recent_resting
        else:
            current_items = recent
            warnings.append("provisional_prediction_used_non_resting_data")
        reference = {
            name: (signal.median, signal.mad_scale)
            for name, signal in request.baseline.signals.items()
        }
        method = "personal_baseline_trend"
        confidence = "moderate"
    else:
        split = max(3, len(observations) * 2 // 3)
        reference_items = observations[:split]
        current_candidates = observations[split:]
        current_resting = [item for item in current_candidates if is_resting(item)]
        if len(current_resting) >= 3:
            current_items = current_resting
        else:
            current_items = current_candidates
            warnings.append("provisional_prediction_used_non_resting_data")
        reference_resting = [item for item in reference_items if is_resting(item)]
        if len(reference_resting) >= 3:
            physiological_reference = reference_resting
        else:
            physiological_reference = reference_items
        reference = {
            name: _robust_reference(
                [getter(item) for item in physiological_reference],
                SCALE_FLOORS[name],
            )
            for name, getter in getters.items()
        }
        motion_values = [float(item.motion) for item in reference_items]
        reference["daily_motion_fraction"] = _robust_reference(
            motion_values,
            SCALE_FLOORS["daily_motion_fraction"],
        )
        method = "within_day_trend"
        confidence = "low"
        warnings.append("personal_baseline_missing_used_within_day_reference")

    current_values = {
        name: float(median(getter(item) for item in current_items))
        for name, getter in getters.items()
    }
    current_values["daily_motion_fraction"] = sum(item.motion for item in observations) / len(observations)
    signed_deviations = {
        name: (value - reference[name][0]) / reference[name][1]
        for name, value in current_values.items()
    }
    current_deviation = max(abs(value) for value in signed_deviations.values())

    projected_deviations: dict[str, float] = {}
    for name, getter in getters.items():
        center, scale = reference[name]
        projected_value = current_values[name] + _slope(current_items, getter) * TREND_PROJECTION_HOURS
        projected_deviations[name] = (projected_value - center) / scale

    halfway = max(1, len(observations) // 2)
    first_motion = fmean(item.motion for item in observations[:halfway])
    second_motion = fmean(item.motion for item in observations[halfway:])
    projected_motion = current_values["daily_motion_fraction"] + (second_motion - first_motion)
    motion_center, motion_scale = reference["daily_motion_fraction"]
    projected_deviations["daily_motion_fraction"] = (projected_motion - motion_center) / motion_scale
    future_deviation = max(abs(value) for value in projected_deviations.values())

    return ({
        "method": method,
        "confidence": confidence,
        "calibration_status": "not_applicable_uncalibrated_score",
        "current_risk": {
            "classification": _classification(current_deviation),
            "score": _risk_score(current_deviation),
            "score_type": "uncalibrated_risk_score",
            "deviation": round(current_deviation, 6),
            "threshold": CHANGE_THRESHOLD,
        },
        "future_risk": {
            "classification": _classification(future_deviation),
            "score": _risk_score(future_deviation),
            "score_type": "uncalibrated_risk_score",
            "deviation": round(future_deviation, 6),
            "horizon_hours": FUTURE_HORIZON_HOURS,
            "trend_projection_hours": TREND_PROJECTION_HOURS,
            "threshold": CHANGE_THRESHOLD,
        },
        "explanation": (
            "Provisional screening prediction based on change and recent trend; "
            "it is not a calibrated illness probability."
        ),
    }, warnings)
