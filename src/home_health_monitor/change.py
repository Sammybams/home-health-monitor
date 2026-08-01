from __future__ import annotations

from statistics import median
from typing import Any

from .baseline import is_resting
from .contracts import PredictionRequest


CHANGE_THRESHOLD = 3.5


def _signal_result(observed: float, center: float, scale: float) -> dict[str, Any]:
    signed_deviation = (observed - center) / scale
    if signed_deviation > 0:
        direction = "above_baseline"
    elif signed_deviation < 0:
        direction = "below_baseline"
    else:
        direction = "at_baseline"
    return {
        "observed": round(observed, 6),
        "baseline_median": round(center, 6),
        "baseline_scale": round(scale, 6),
        "robust_deviation": round(abs(signed_deviation), 6),
        "direction": direction,
    }


def assess_change(request: PredictionRequest) -> tuple[dict[str, Any], list[str]]:
    """Compare recent resting physiology with this person's healthy profile."""
    if request.baseline is None:
        return ({
            "status": "insufficient_data",
            "reason": "personal_baseline_required",
            "score": None,
        }, ["personal_baseline_missing"])

    observations = request.observations
    cutoff = observations[-1].timestamp.timestamp() - 6 * 3600
    recent_resting = [
        item for item in observations
        if item.timestamp.timestamp() >= cutoff and is_resting(item)
    ]
    warnings: list[str] = []
    window_hours = 6
    if len(recent_resting) < 12:
        recent_resting = [item for item in observations if is_resting(item)]
        window_hours = 24
        warnings.append("too_few_recent_resting_samples_used_full_history")
    if len(recent_resting) < 12:
        return ({
            "status": "insufficient_data",
            "reason": "at_least_12_resting_samples_required",
            "score": None,
        }, warnings + ["insufficient_resting_data"])

    current = {
        "resting_body_temperature_c": float(median(item.body_temperature_c for item in recent_resting)),
        "resting_heart_rate_bpm": float(median(item.heart_rate_bpm for item in recent_resting)),
        "resting_body_ambient_delta_c": float(median(
            item.body_temperature_c - item.ambient_temperature_c for item in recent_resting
        )),
        "daily_motion_fraction": sum(item.motion for item in observations) / len(observations),
    }
    signals = {
        name: _signal_result(value, request.baseline.signals[name].median, request.baseline.signals[name].mad_scale)
        for name, value in current.items()
    }
    score = max(float(item["robust_deviation"]) for item in signals.values())
    unusual = score >= CHANGE_THRESHOLD
    return ({
        "status": "unusual_change" if unusual else "within_personal_baseline",
        "score": round(score, 6),
        "threshold": CHANGE_THRESHOLD,
        "resting_window_hours": window_hours,
        "resting_sample_count": len(recent_resting),
        "signals": signals,
        "interpretation": (
            "One or more measurements changed substantially from this person's healthy baseline."
            if unusual else
            "No configured measurement changed substantially from this person's healthy baseline."
        ),
        "disclaimer": "Change detection only; this score is not an illness probability or diagnosis.",
    }, warnings)
