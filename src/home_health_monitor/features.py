from __future__ import annotations

import math
from statistics import fmean, pstdev

from .contracts import Observation, PredictionRequest


SIGNALS = ("body_temperature_c", "ambient_temperature_c", "heart_rate_bpm")
WINDOWS_HOURS = (1, 6, 24)
STATS = ("mean", "min", "max", "std", "slope_per_hour", "latest")


def feature_names() -> tuple[str, ...]:
    names: list[str] = []
    for hours in WINDOWS_HOURS:
        for signal in SIGNALS:
            names.extend(f"{hours}h_{signal}_{stat}" for stat in STATS)
        names.extend((f"{hours}h_motion_fraction", f"{hours}h_motion_transitions"))
    names.extend(("body_ambient_delta_mean", "sample_count", "span_hours", "largest_gap_minutes"))
    return tuple(names)


def _slope_per_hour(items: tuple[Observation, ...], signal: str) -> float:
    if len(items) < 2:
        return 0.0
    origin = items[0].timestamp
    xs = [(item.timestamp - origin).total_seconds() / 3600 for item in items]
    ys = [float(getattr(item, signal)) for item in items]
    x_mean, y_mean = fmean(xs), fmean(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator == 0:
        return 0.0
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator


def _window_features(items: tuple[Observation, ...], hours: int) -> dict[str, float]:
    result: dict[str, float] = {}
    for signal in SIGNALS:
        values = [float(getattr(item, signal)) for item in items]
        prefix = f"{hours}h_{signal}"
        result[f"{prefix}_mean"] = fmean(values)
        result[f"{prefix}_min"] = min(values)
        result[f"{prefix}_max"] = max(values)
        result[f"{prefix}_std"] = pstdev(values) if len(values) > 1 else 0.0
        result[f"{prefix}_slope_per_hour"] = _slope_per_hour(items, signal)
        result[f"{prefix}_latest"] = values[-1]
    motions = [item.motion for item in items]
    result[f"{hours}h_motion_fraction"] = fmean(motions)
    result[f"{hours}h_motion_transitions"] = float(sum(a != b for a, b in zip(motions, motions[1:])))
    return result


def extract_features(request: PredictionRequest) -> tuple[dict[str, float], list[str]]:
    observations = request.observations
    end = observations[-1].timestamp
    values: dict[str, float] = {}
    warnings: list[str] = []
    for hours in WINDOWS_HOURS:
        cutoff = end.timestamp() - hours * 3600
        items = tuple(item for item in observations if item.timestamp.timestamp() >= cutoff)
        values.update(_window_features(items, hours))

    values["body_ambient_delta_mean"] = fmean(
        item.body_temperature_c - item.ambient_temperature_c for item in observations
    )
    values["sample_count"] = float(len(observations))
    span = (observations[-1].timestamp - observations[0].timestamp).total_seconds()
    values["span_hours"] = span / 3600
    gaps = [
        (b.timestamp - a.timestamp).total_seconds() / 60
        for a, b in zip(observations, observations[1:])
    ]
    values["largest_gap_minutes"] = max(gaps, default=0.0)
    if span < 20 * 3600:
        warnings.append("history_shorter_than_recommended_20_hours")
    median_interval = sorted(gaps)[len(gaps) // 2] if gaps else math.inf
    if values["largest_gap_minutes"] > max(15.0, median_interval * 5):
        warnings.append("large_sampling_gap_detected")
    return values, warnings
