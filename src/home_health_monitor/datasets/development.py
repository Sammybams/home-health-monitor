from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import statistics
from typing import Iterable

from ..gateway.autoencoder import FEATURE_MANIFEST_ID
from ..gateway.windowing import FEATURE_NAMES, STEP_COUNT


@dataclass(frozen=True, slots=True)
class SuppliedMonitoringRow:
    heart_rate_bpm: float
    spo2_percent: float
    temperature_c: float
    motion_intensity: float
    condition: str

    @property
    def is_normal(self) -> bool:
        return self.condition.strip().casefold() == "normal"


@dataclass(frozen=True, slots=True)
class DevelopmentWindow:
    subject_id: str
    decision: str
    values: tuple[tuple[float, ...], ...]
    masks: tuple[tuple[float, ...], ...]
    scenario: str

    def training_record(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "subject_id": self.subject_id,
            "feature_manifest_id": FEATURE_MANIFEST_ID,
            "decision": self.decision,
            "feature_names": list(FEATURE_NAMES),
            "values": [list(step) for step in self.values],
            "masks": [list(step) for step in self.masks],
        }


@dataclass(frozen=True, slots=True)
class DevelopmentCorpus:
    normal_windows: tuple[DevelopmentWindow, ...]
    simulated_anomaly_windows: tuple[DevelopmentWindow, ...]
    supplied_unique_rows: int
    supplied_normal_rows: int


def _finite(row: dict[str, str], name: str) -> float:
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"supplied CSV has an invalid {name}") from exc
    if not math.isfinite(value):
        raise ValueError(f"supplied CSV has a non-finite {name}")
    return value


def load_supplied_monitoring_rows(path: str | Path) -> tuple[SuppliedMonitoringRow, ...]:
    """Load distinct target-feature rows from the supplied synthetic CSV."""
    unique: dict[SuppliedMonitoringRow, None] = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            acceleration = _finite(row, "acceleration_magnitude")
            item = SuppliedMonitoringRow(
                heart_rate_bpm=_finite(row, "heart_rate"),
                spo2_percent=_finite(row, "oxygen_level"),
                temperature_c=_finite(row, "temperature"),
                motion_intensity=abs(acceleration - 9.81),
                condition=str(row.get("health_condition", "")).strip(),
            )
            unique[item] = None
    if not unique:
        raise ValueError("supplied CSV contains no usable rows")
    return tuple(unique)


def _robust_scale(values: Iterable[float], minimum: float) -> float:
    items = tuple(float(value) for value in values)
    median = statistics.median(items)
    mad = statistics.median(abs(value - median) for value in items)
    return max(1.4826 * mad, minimum)


def _normal_window(rng: random.Random, subject_id: str, phase: float) -> DevelopmentWindow:
    values: list[tuple[float, ...]] = []
    masks: list[tuple[float, ...]] = []
    activity_start = rng.randrange(70, 190)
    activity_length = rng.randrange(8, 25)
    for step in range(STEP_COUNT):
        day_angle = 2 * math.pi * step / STEP_COUNT + phase
        active = 1.0 if activity_start <= step < activity_start + activity_length else 0.0
        motion = max(0.0, 0.16 + 0.10 * math.sin(day_angle * 3) + active * 1.15)
        heart_rate = 0.38 * math.sin(day_angle - 0.5) + active * 1.20 + rng.gauss(0, 0.16)
        spo2 = -0.10 * active + rng.gauss(0, 0.10)
        temperature = 0.34 * math.sin(day_angle - 1.2) + rng.gauss(0, 0.08)
        quality = tuple(max(0.72, min(1.0, rng.gauss(0.95, 0.025))) for _ in range(4))
        values.append((heart_rate, spo2, temperature, motion, *quality))
        masks.append((1.0,) * len(FEATURE_NAMES))
    return DevelopmentWindow(subject_id, "normal", tuple(values), tuple(masks), "normal")


def _simulated_anomaly(window: DevelopmentWindow, scenario_index: int) -> DevelopmentWindow:
    scenarios = (
        ("sustained_high_heart_rate", 0, 3.8),
        ("sustained_low_spo2", 1, -4.2),
        ("sustained_high_temperature", 2, 4.0),
        ("sustained_motion_change", 3, 4.5),
    )
    name, feature, shift = scenarios[scenario_index % len(scenarios)]
    values = [list(step) for step in window.values]
    start = 120 + scenario_index % 24
    for step in range(start, min(STEP_COUNT, start + 48)):
        values[step][feature] += shift
    return DevelopmentWindow(
        window.subject_id,
        "engineering_simulation",
        tuple(tuple(step) for step in values),
        window.masks,
        name,
    )


def build_development_corpus(
    supplied_rows: Iterable[SuppliedMonitoringRow],
    *,
    subject_count: int = 60,
    windows_per_subject: int = 4,
    seed: int = 42,
) -> DevelopmentCorpus:
    """Create deterministic demo windows; this is not a real participant cohort."""
    rows = tuple(supplied_rows)
    normal = tuple(row for row in rows if row.is_normal)
    if not normal:
        raise ValueError("development corpus needs at least one supplied normal row")
    if subject_count < 3 or windows_per_subject < 1:
        raise ValueError("development corpus needs three subjects and one window each")

    # These scales record how the supplied normal rows shaped the simulation.
    scales = (
        _robust_scale((row.heart_rate_bpm for row in normal), 3.0),
        _robust_scale((row.spo2_percent for row in normal), 0.5),
        _robust_scale((row.temperature_c for row in normal), 0.1),
        _robust_scale((row.motion_intensity for row in normal), 0.1),
    )
    scale_phase = sum(scales) / 100.0
    rng = random.Random(seed)
    windows: list[DevelopmentWindow] = []
    for subject_index in range(subject_count):
        subject_id = f"development-{subject_index + 1:03d}"
        phase = rng.uniform(-math.pi, math.pi) + scale_phase
        for _ in range(windows_per_subject):
            windows.append(_normal_window(rng, subject_id, phase + rng.uniform(-0.15, 0.15)))
    anomalies = tuple(_simulated_anomaly(window, index) for index, window in enumerate(windows))
    return DevelopmentCorpus(
        normal_windows=tuple(windows),
        simulated_anomaly_windows=anomalies,
        supplied_unique_rows=len(rows),
        supplied_normal_rows=len(normal),
    )


def write_normal_windows(corpus: DevelopmentCorpus, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for window in corpus.normal_windows:
            handle.write(json.dumps(window.training_record(), separators=(",", ":")) + "\n")
