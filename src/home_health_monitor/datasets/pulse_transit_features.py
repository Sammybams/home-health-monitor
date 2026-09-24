from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np


DATASET_FEATURE_NAMES = (
    "heart_rate_bpm",
    "heart_rate_valid",
    "ppg_rr_interval_std_ms",
    "ppg_relative_pulse_amplitude",
    "ppg_ac_rms_normalized",
    "ppg_signal_quality",
    "temperature_mean_c",
    "temperature_std_c",
    "acceleration_magnitude_mean",
    "acceleration_magnitude_std",
    "dynamic_acceleration_rms",
    "motion_intensity",
)

_REQUIRED_COLUMNS = ("pleth_2", "peaks", "temp_1", "a_x", "a_y", "a_z")


@dataclass(frozen=True, slots=True)
class SegmentFeatures:
    values: dict[str, float]
    ecg_reference_hr_bpm: float | None


def _finite_array(rows: Sequence[Mapping[str, str]], name: str) -> np.ndarray:
    try:
        values = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain finite values")
    return values


def ecg_reference_heart_rate(peaks: Sequence[float], sample_rate_hz: float) -> float | None:
    """Calculate validation-only ECG heart rate from the supplied peak annotation."""
    if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be finite and positive")
    peak_indices = np.flatnonzero(np.asarray(peaks, dtype=np.float64) > 0.0)
    if peak_indices.size < 2:
        return None
    intervals = np.diff(peak_indices) / sample_rate_hz
    physiological = intervals[(intervals >= 0.25) & (intervals <= 2.0)]
    if physiological.size == 0:
        return None
    return float(60.0 / np.median(physiological))


def _candidate_peaks(signal: np.ndarray, sample_rate_hz: float) -> np.ndarray:
    candidates = np.flatnonzero(
        (signal[1:-1] > signal[:-2]) & (signal[1:-1] >= signal[2:])
    ) + 1
    robust_span = float(np.percentile(signal, 95) - np.percentile(signal, 5))
    threshold = float(np.median(signal) + 0.15 * robust_span)
    candidates = candidates[signal[candidates] >= threshold]
    minimum_distance = max(1, int(round(0.3 * sample_rate_hz)))
    selected: list[int] = []
    for candidate in candidates:
        candidate = int(candidate)
        if not selected or candidate - selected[-1] >= minimum_distance:
            selected.append(candidate)
        elif signal[candidate] > signal[selected[-1]]:
            selected[-1] = candidate
    return np.asarray(selected, dtype=np.int64)


def _ppg_summary(ppg: np.ndarray, sample_rate_hz: float) -> dict[str, float]:
    centered = ppg - np.mean(ppg)
    frequencies = np.fft.rfftfreq(ppg.size, d=1.0 / sample_rate_hz)
    spectrum = np.fft.rfft(centered)
    spectrum[(frequencies < 0.7) | (frequencies > 3.0)] = 0.0
    filtered = np.fft.irfft(spectrum, n=ppg.size)

    options = [_candidate_peaks(filtered, sample_rate_hz), _candidate_peaks(-filtered, sample_rate_hz)]
    best: tuple[float, np.ndarray, np.ndarray] | None = None
    for peaks in options:
        intervals = np.diff(peaks) / sample_rate_hz
        intervals = intervals[(intervals >= 0.3) & (intervals <= 1.5)]
        if intervals.size == 0:
            continue
        coefficient_of_variation = float(np.std(intervals) / np.mean(intervals))
        score = float(intervals.size) - coefficient_of_variation
        if best is None or score > best[0]:
            best = (score, peaks, intervals)

    ppg_dc = max(abs(float(np.median(ppg))), np.finfo(np.float64).eps)
    pulse_amplitude = float(np.percentile(ppg, 95) - np.percentile(ppg, 5))
    normalized_rms = float(np.sqrt(np.mean(centered**2)) / ppg_dc)
    if best is None:
        heart_rate = 0.0
        heart_rate_valid = 0.0
        interval_std_ms = 0.0
        quality = 0.0
    else:
        intervals = best[2]
        heart_rate = float(60.0 / np.median(intervals))
        heart_rate_valid = 1.0
        interval_std_ms = float(np.std(intervals) * 1_000.0)
        quality = float(max(0.0, min(1.0, 1.0 - np.std(intervals) / np.mean(intervals))))

    return {
        "heart_rate_bpm": heart_rate,
        "heart_rate_valid": heart_rate_valid,
        "ppg_rr_interval_std_ms": interval_std_ms,
        "ppg_relative_pulse_amplitude": pulse_amplitude / ppg_dc,
        "ppg_ac_rms_normalized": normalized_rms,
        "ppg_signal_quality": quality,
    }


def extract_segment_features(
    rows: Sequence[Mapping[str, str]], *, sample_rate_hz: float
) -> SegmentFeatures:
    """Convert one complete short sensor segment into a dataset-facing vector."""
    if not rows:
        raise ValueError("segment must contain at least one row")
    if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be finite and positive")
    arrays = {name: _finite_array(rows, name) for name in _REQUIRED_COLUMNS}
    ppg_values = _ppg_summary(arrays["pleth_2"], sample_rate_hz)
    temperature = arrays["temp_1"]
    acceleration = np.column_stack((arrays["a_x"], arrays["a_y"], arrays["a_z"]))
    magnitude = np.linalg.norm(acceleration, axis=1)
    dynamic = acceleration - np.mean(acceleration, axis=0)
    magnitude_differences = np.diff(magnitude)
    motion_intensity = (
        float(np.sqrt(np.mean(magnitude_differences**2)))
        if magnitude_differences.size
        else 0.0
    )
    values = {
        **ppg_values,
        "temperature_mean_c": float(np.mean(temperature)),
        "temperature_std_c": float(np.std(temperature)),
        "acceleration_magnitude_mean": float(np.mean(magnitude)),
        "acceleration_magnitude_std": float(np.std(magnitude)),
        "dynamic_acceleration_rms": float(np.sqrt(np.mean(dynamic**2))),
        "motion_intensity": motion_intensity,
    }
    if tuple(values) != DATASET_FEATURE_NAMES:
        raise AssertionError("feature order does not match DATASET_FEATURE_NAMES")
    return SegmentFeatures(
        values=values,
        ecg_reference_hr_bpm=ecg_reference_heart_rate(arrays["peaks"], sample_rate_hz),
    )
