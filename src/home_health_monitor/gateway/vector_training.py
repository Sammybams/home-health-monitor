from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable

from home_health_monitor.datasets.pulse_transit_build import (
    DATASET_ID,
    FEATURE_MANIFEST_ID,
)
from home_health_monitor.datasets.pulse_transit_features import DATASET_FEATURE_NAMES


class VectorTrainingDataError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RealPpgVector:
    subject_id: str
    record: str
    activity: str
    gender: str
    age: int
    values: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ParticipantPlan:
    development_subjects: tuple[str, ...]
    locked_test_subjects: tuple[str, ...]
    validation_folds: tuple[tuple[str, ...], ...]


@dataclass(frozen=True, slots=True)
class RobustNormalizer:
    center: tuple[float, ...]
    scale: tuple[float, ...]

    def transform(self, rows: Iterable[RealPpgVector], np: Any) -> Any:
        rows = tuple(rows)
        values = np.asarray([row.values for row in rows], dtype=np.float32)
        result = (values - np.asarray(self.center, dtype=np.float32)) / np.asarray(
            self.scale, dtype=np.float32
        )
        valid_index = DATASET_FEATURE_NAMES.index("heart_rate_valid")
        invalid = values[:, valid_index] == 0.0
        for name in ("heart_rate_bpm", "ppg_rr_interval_std_ms"):
            result[invalid, DATASET_FEATURE_NAMES.index(name)] = 0.0
        return result

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": "median_iqr",
            "feature_names": list(DATASET_FEATURE_NAMES),
            "center": list(self.center),
            "scale": list(self.scale),
            "missing_heart_rate_policy": "normalized_zero_with_validity_feature",
        }


def _finite_values(value: Any, line_number: int) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != len(DATASET_FEATURE_NAMES):
        raise VectorTrainingDataError(
            f"row {line_number} values must contain {len(DATASET_FEATURE_NAMES)} features"
        )
    result = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise VectorTrainingDataError(f"row {line_number} values must be numeric")
        number = float(item)
        if not math.isfinite(number):
            raise VectorTrainingDataError(f"row {line_number} values must be finite")
        result.append(number)
    valid_index = DATASET_FEATURE_NAMES.index("heart_rate_valid")
    if result[valid_index] not in (0.0, 1.0):
        raise VectorTrainingDataError(
            f"row {line_number} heart_rate_valid must be binary"
        )
    return tuple(result)


def load_real_ppg_vectors(path: str | Path) -> tuple[RealPpgVector, ...]:
    rows: list[RealPpgVector] = []
    subject_demographics: dict[str, tuple[str, int]] = {}
    try:
        handle = Path(path).open(encoding="utf-8")
    except OSError as exc:
        raise VectorTrainingDataError(f"could not open vector dataset: {exc}") from exc
    with handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise VectorTrainingDataError(f"row {line_number} is not valid JSON") from exc
            if not isinstance(item, dict) or item.get("schema_version") != 1:
                raise VectorTrainingDataError(f"row {line_number} has an unsupported schema")
            if item.get("dataset_id") != DATASET_ID:
                raise VectorTrainingDataError(f"row {line_number} has the wrong dataset id")
            if item.get("feature_manifest_id") != FEATURE_MANIFEST_ID:
                raise VectorTrainingDataError(
                    f"row {line_number} has the wrong feature manifest"
                )
            if tuple(item.get("feature_names", ())) != DATASET_FEATURE_NAMES:
                raise VectorTrainingDataError(f"row {line_number} has the wrong feature order")
            subject_id = item.get("subject_id")
            record = item.get("record")
            activity = item.get("activity")
            demographics = item.get("demographics")
            if not all(isinstance(value, str) and value for value in (subject_id, record, activity)):
                raise VectorTrainingDataError(f"row {line_number} has invalid identifiers")
            if not isinstance(demographics, dict):
                raise VectorTrainingDataError(f"row {line_number} has invalid demographics")
            gender = demographics.get("gender")
            age = demographics.get("age")
            if not isinstance(gender, str) or not gender or isinstance(age, bool) or not isinstance(age, int):
                raise VectorTrainingDataError(f"row {line_number} has invalid demographics")
            prior = subject_demographics.setdefault(subject_id, (gender, age))
            if prior != (gender, age):
                raise VectorTrainingDataError(
                    f"row {line_number} changes demographics for {subject_id}"
                )
            rows.append(
                RealPpgVector(
                    subject_id=subject_id,
                    record=record,
                    activity=activity,
                    gender=gender,
                    age=age,
                    values=_finite_values(item.get("values"), line_number),
                )
            )
    if not rows:
        raise VectorTrainingDataError("vector dataset contains no rows")
    return tuple(rows)


def participant_plan(
    rows: Iterable[RealPpgVector],
    *,
    seed: int = 42,
    fold_count: int = 5,
    locked_fraction: float = 0.2,
) -> ParticipantPlan:
    subjects = sorted({row.subject_id for row in rows})
    if fold_count < 2:
        raise ValueError("fold_count must be at least 2")
    if not 0.0 < locked_fraction < 1.0:
        raise ValueError("locked_fraction must be between 0 and 1")
    if len(subjects) < fold_count + 1:
        raise VectorTrainingDataError(
            "participant-grouped validation needs at least fold_count + 1 subjects"
        )
    shuffled = subjects.copy()
    random.Random(seed).shuffle(shuffled)
    locked_count = max(1, round(len(shuffled) * locked_fraction))
    locked = tuple(sorted(shuffled[:locked_count]))
    development = shuffled[locked_count:]
    if len(development) < fold_count:
        raise VectorTrainingDataError("too few development subjects for requested folds")
    random.Random(seed + 1).shuffle(development)
    folds = [[] for _ in range(fold_count)]
    for index, subject in enumerate(development):
        folds[index % fold_count].append(subject)
    return ParticipantPlan(
        development_subjects=tuple(sorted(development)),
        locked_test_subjects=locked,
        validation_folds=tuple(tuple(sorted(fold)) for fold in folds),
    )


def rows_for_subjects(
    rows: Iterable[RealPpgVector], subjects: Iterable[str]
) -> tuple[RealPpgVector, ...]:
    selected = set(subjects)
    return tuple(row for row in rows if row.subject_id in selected)


def fit_robust_normalizer(
    rows: Iterable[RealPpgVector], np: Any
) -> RobustNormalizer:
    rows = tuple(rows)
    if not rows:
        raise VectorTrainingDataError("normalizer requires at least one row")
    values = np.asarray([row.values for row in rows], dtype=np.float64)
    valid_index = DATASET_FEATURE_NAMES.index("heart_rate_valid")
    heart_rate_valid = values[:, valid_index] == 1.0
    centers = []
    scales = []
    for index, name in enumerate(DATASET_FEATURE_NAMES):
        column = values[:, index]
        if name in {"heart_rate_bpm", "ppg_rr_interval_std_ms"}:
            column = column[heart_rate_valid]
        if column.size == 0:
            raise VectorTrainingDataError(f"no valid training values for {name}")
        center = float(np.median(column))
        scale = float(np.percentile(column, 75) - np.percentile(column, 25))
        if not math.isfinite(scale) or scale <= 1e-8:
            scale = float(np.std(column))
        if not math.isfinite(scale) or scale <= 1e-8:
            scale = 1.0
        centers.append(center)
        scales.append(scale)
    return RobustNormalizer(tuple(centers), tuple(scales))
