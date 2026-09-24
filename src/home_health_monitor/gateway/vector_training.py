from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
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


def _quantile(values: Any, probability: float, np: Any) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)) or np.any(array < 0):
        raise VectorTrainingDataError("threshold scores must be finite and non-negative")
    return float(np.quantile(array, probability))


def vector_thresholds(scores: Any, np: Any) -> tuple[float, float]:
    persistent = max(_quantile(scores, 0.95, np), 1e-8)
    severe = max(_quantile(scores, 0.99, np), persistent * 2.0)
    return persistent, severe


def _build_autoencoder(tf: Any) -> Any:
    inputs = tf.keras.Input(shape=(len(DATASET_FEATURE_NAMES),), name="feature_vector")
    encoded = tf.keras.layers.Dense(8, activation="relu", name="encoder_8")(inputs)
    bottleneck = tf.keras.layers.Dense(3, activation="relu", name="bottleneck_3")(encoded)
    decoded = tf.keras.layers.Dense(8, activation="relu", name="decoder_8")(bottleneck)
    outputs = tf.keras.layers.Dense(
        len(DATASET_FEATURE_NAMES), name="reconstruction"
    )(decoded)
    model = tf.keras.Model(inputs, outputs, name="real_ppg_vector_autoencoder")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    return model


def _tflite_bytes(model: Any, representative: Any, tf: Any, np: Any) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_dataset():
        for item in representative[: min(300, len(representative))]:
            yield [item[np.newaxis, :].astype(np.float32)]

    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def _quantized_reconstructions(model_bytes: bytes, values: Any, tf: Any, np: Any) -> Any:
    interpreter = tf.lite.Interpreter(model_content=model_bytes)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    input_scale, input_zero_point = input_detail["quantization"]
    output_scale, output_zero_point = output_detail["quantization"]
    if input_scale <= 0 or output_scale <= 0:
        raise VectorTrainingDataError("quantized model has invalid tensor scales")
    result = []
    for item in values:
        quantized = np.clip(
            np.rint(item / input_scale) + input_zero_point, -128, 127
        ).astype(np.int8)
        interpreter.set_tensor(input_detail["index"], quantized[np.newaxis, :])
        interpreter.invoke()
        output = interpreter.get_tensor(output_detail["index"])[0]
        result.append((output.astype(np.float32) - output_zero_point) * output_scale)
    return np.asarray(result, dtype=np.float32)


def _quantized_tensor_metadata(model_bytes: bytes, tf: Any) -> dict[str, Any]:
    interpreter = tf.lite.Interpreter(model_content=model_bytes)
    interpreter.allocate_tensors()
    result = {}
    for name, detail in (
        ("input", interpreter.get_input_details()[0]),
        ("output", interpreter.get_output_details()[0]),
    ):
        scale, zero_point = detail["quantization"]
        if scale <= 0:
            raise VectorTrainingDataError(f"quantized {name} tensor has invalid scale")
        result[name] = {
            "shape": [int(value) for value in detail["shape"]],
            "dtype": str(detail["dtype"].__name__),
            "scale": float(scale),
            "zero_point": int(zero_point),
        }
    return result


def _reconstruction_scores(values: Any, reconstructions: Any, np: Any) -> Any:
    return np.mean((values - reconstructions) ** 2, axis=1)


def controlled_anomalies(values: Any, np: Any) -> tuple[Any, list[str]]:
    """Create labelled sensitivity checks in normalized feature space."""
    scenarios = (
        ("heart_rate_shift", {"heart_rate_bpm": 6.0}),
        ("temperature_shift", {"temperature_mean_c": 6.0}),
        (
            "motion_shift",
            {"dynamic_acceleration_rms": 6.0, "motion_intensity": 6.0},
        ),
        (
            "combined_drift",
            {
                "heart_rate_bpm": 6.0,
                "temperature_mean_c": 6.0,
                "dynamic_acceleration_rms": 6.0,
            },
        ),
    )
    generated = []
    labels = []
    for name, shifts in scenarios:
        changed = values.copy()
        for feature, shift in shifts.items():
            changed[:, DATASET_FEATURE_NAMES.index(feature)] += shift
        generated.append(changed)
        labels.extend([name] * len(changed))
    return np.concatenate(generated, axis=0), labels


def _score_summary(scores: Any, threshold: float, np: Any) -> dict[str, float | int]:
    values = np.asarray(scores, dtype=np.float64)
    return {
        "count": int(values.size),
        "minimum": float(np.min(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "maximum": float(np.max(values)),
        "anomaly_fraction": float(np.mean(values >= threshold)),
    }


def _group_summaries(
    rows: tuple[RealPpgVector, ...], scores: Any, field: str, threshold: float, np: Any
) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[float]] = {}
    for row, score in zip(rows, scores):
        if field == "age_group":
            key = "20-29" if row.age < 30 else "30+"
        else:
            key = str(getattr(row, field))
        groups.setdefault(key, []).append(float(score))
    return {
        key: _score_summary(values, threshold, np)
        for key, values in sorted(groups.items())
    }


def aggregate_record_scores(
    rows: Iterable[RealPpgVector], scores: Any, np: Any
) -> tuple[dict[str, Any], ...]:
    grouped: dict[str, tuple[RealPpgVector, list[float]]] = {}
    for row, score in zip(rows, scores):
        if row.record not in grouped:
            grouped[row.record] = (row, [])
        grouped[row.record][1].append(float(score))
    result = []
    for record, (row, values) in sorted(grouped.items()):
        array = np.asarray(values, dtype=np.float64)
        result.append(
            {
                "record": record,
                "subject_id": row.subject_id,
                "activity": row.activity,
                "gender": row.gender,
                "age_group": "20-29" if row.age < 30 else "30+",
                "vector_count": len(values),
                "median_vector_error": float(np.median(array)),
                "p95_vector_error": float(np.percentile(array, 95)),
                "maximum_vector_error": float(np.max(array)),
            }
        )
    return tuple(result)


def _interval_group_summaries(
    intervals: tuple[dict[str, Any], ...], field: str, threshold: float, np: Any
) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[float]] = {}
    for interval in intervals:
        groups.setdefault(str(interval[field]), []).append(
            float(interval["p95_vector_error"])
        )
    return {
        key: _score_summary(values, threshold, np)
        for key, values in sorted(groups.items())
    }


def train_vector_autoencoder(
    input_path: str | Path,
    output_directory: str | Path,
    *,
    epochs: int = 30,
    batch_size: int = 64,
    seed: int = 42,
) -> dict[str, Any]:
    try:
        import numpy as np
        import tensorflow as tf
    except ImportError as exc:
        raise SystemExit(
            "real-PPG training requires Python 3.9-3.12 and: "
            "python -m pip install -e '.[gateway-train]'"
        ) from exc
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    rows = load_real_ppg_vectors(input_path)
    plan = participant_plan(rows, seed=seed)
    development_rows = rows_for_subjects(rows, plan.development_subjects)
    locked_rows = rows_for_subjects(rows, plan.locked_test_subjects)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    out_of_fold_scores: list[float] = []
    out_of_fold_rows: list[RealPpgVector] = []
    fold_reports = []
    best_epoch_counts = []
    for fold_index, validation_subjects in enumerate(plan.validation_folds, 1):
        training_subjects = tuple(
            subject
            for subject in plan.development_subjects
            if subject not in set(validation_subjects)
        )
        training_rows = rows_for_subjects(rows, training_subjects)
        validation_rows = rows_for_subjects(rows, validation_subjects)
        normalizer = fit_robust_normalizer(training_rows, np)
        training_values = normalizer.transform(training_rows, np)
        validation_values = normalizer.transform(validation_rows, np)
        model = _build_autoencoder(tf)
        history = model.fit(
            training_values,
            training_values,
            validation_data=(validation_values, validation_values),
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=5, restore_best_weights=True
                )
            ],
            verbose=0,
        )
        best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
        best_epoch_counts.append(best_epoch)
        quantized = _tflite_bytes(model, training_values, tf, np)
        reconstructions = _quantized_reconstructions(
            quantized, validation_values, tf, np
        )
        scores = _reconstruction_scores(validation_values, reconstructions, np)
        out_of_fold_scores.extend(float(score) for score in scores)
        out_of_fold_rows.extend(validation_rows)
        fold_reports.append(
            {
                "fold": fold_index,
                "training_subjects": list(training_subjects),
                "validation_subjects": list(validation_subjects),
                "training_rows": len(training_rows),
                "validation_rows": len(validation_rows),
                "best_epoch": best_epoch,
                "minimum_validation_loss": float(min(history.history["val_loss"])),
            }
        )
        tf.keras.backend.clear_session()

    out_of_fold_intervals = aggregate_record_scores(
        out_of_fold_rows, out_of_fold_scores, np
    )
    out_of_fold_interval_scores = [
        interval["p95_vector_error"] for interval in out_of_fold_intervals
    ]
    persistent_threshold, severe_threshold = vector_thresholds(
        out_of_fold_interval_scores, np
    )
    normalizer = fit_robust_normalizer(development_rows, np)
    development_values = normalizer.transform(development_rows, np)
    locked_values = normalizer.transform(locked_rows, np)
    final_epochs = max(1, int(round(float(np.median(best_epoch_counts)))))
    final_model = _build_autoencoder(tf)
    final_history = final_model.fit(
        development_values,
        development_values,
        epochs=final_epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=0,
    )
    model_bytes = _tflite_bytes(final_model, development_values, tf, np)
    locked_reconstructions = _quantized_reconstructions(
        model_bytes, locked_values, tf, np
    )
    locked_scores = _reconstruction_scores(
        locked_values, locked_reconstructions, np
    )
    locked_intervals = aggregate_record_scores(locked_rows, locked_scores, np)
    locked_interval_scores = [
        interval["p95_vector_error"] for interval in locked_intervals
    ]
    simulated_values, simulated_scenarios = controlled_anomalies(locked_values, np)
    simulated_reconstructions = _quantized_reconstructions(
        model_bytes, simulated_values, tf, np
    )
    simulated_scores = _reconstruction_scores(
        simulated_values, simulated_reconstructions, np
    )
    checksum = hashlib.sha256(model_bytes).hexdigest()
    input_checksum = hashlib.sha256(Path(input_path).read_bytes()).hexdigest()
    tensor_metadata = _quantized_tensor_metadata(model_bytes, tf)
    model_id = f"real-ppg-vector-ae-{checksum[:12]}"
    scenario_groups: dict[str, tuple[dict[str, Any], ...]] = {}
    offset = 0
    for scenario in dict.fromkeys(simulated_scenarios):
        count = simulated_scenarios.count(scenario)
        scenario_groups[scenario] = aggregate_record_scores(
            locked_rows, simulated_scores[offset : offset + count], np
        )
        offset += count
    report = {
        "schema_version": 1,
        "model_id": model_id,
        "artifact_role": "real_data_development_candidate",
        "evaluation_scope": "healthy_public_data_and_controlled_sensitivity_checks",
        "participant_plan": {
            "seed": seed,
            "development_subjects": list(plan.development_subjects),
            "locked_test_subjects": list(plan.locked_test_subjects),
            "validation_folds": [list(fold) for fold in plan.validation_folds],
        },
        "cross_validation": {
            "folds": fold_reports,
            "out_of_fold_vectors": _score_summary(
                out_of_fold_scores, persistent_threshold, np
            ),
            "out_of_fold_intervals": _score_summary(
                out_of_fold_interval_scores, persistent_threshold, np
            ),
            "intervals_by_activity": _interval_group_summaries(
                out_of_fold_intervals, "activity", persistent_threshold, np
            ),
            "intervals_by_gender": _interval_group_summaries(
                out_of_fold_intervals, "gender", persistent_threshold, np
            ),
            "intervals_by_age_group": _interval_group_summaries(
                out_of_fold_intervals, "age_group", persistent_threshold, np
            ),
        },
        "locked_normal_test": {
            "vector_summary": _score_summary(locked_scores, persistent_threshold, np),
            "interval_summary": _score_summary(
                locked_interval_scores, persistent_threshold, np
            ),
            "intervals_by_activity": _interval_group_summaries(
                locked_intervals, "activity", persistent_threshold, np
            ),
            "intervals_by_gender": _interval_group_summaries(
                locked_intervals, "gender", persistent_threshold, np
            ),
            "intervals_by_age_group": _interval_group_summaries(
                locked_intervals, "age_group", persistent_threshold, np
            ),
            "intervals": list(locked_intervals),
        },
        "controlled_sensitivity": {
            name: {
                "standardized_shift": 6.0,
                "summary": _score_summary(
                    [interval["p95_vector_error"] for interval in intervals],
                    persistent_threshold,
                    np,
                ),
                "intervals": list(intervals),
            }
            for name, intervals in sorted(scenario_groups.items())
        },
        "aggregation": {
            "interval_minutes": 8,
            "vector_seconds": 5,
            "vector_score": "mean_feature_squared_reconstruction_error",
            "interval_score": "95th_percentile_of_vector_scores",
            "persistent_rule": "two_consecutive_intervals_at_or_above_threshold",
            "severe_rule": "one_interval_at_or_above_severe_threshold",
        },
        "thresholds": {
            "source": "quantized_out_of_fold_normal_eight_minute_scores",
            "persistent_error": persistent_threshold,
            "severe_error": severe_threshold,
        },
        "training": {
            "requested_max_epochs": epochs,
            "selected_final_epochs": final_epochs,
            "batch_size": batch_size,
            "final_loss": [float(value) for value in final_history.history["loss"]],
        },
        "controlled_scores": [float(score) for score in simulated_scores],
        "controlled_scenarios": simulated_scenarios,
    }
    metadata = {
        "schema_version": 1,
        "model_id": model_id,
        "artifact_role": "real_data_development_candidate",
        "deployment_status": "blocked_pending_ble_contract_and_pi_validation",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_sha256": checksum,
        "model_size_bytes": len(model_bytes),
        "dataset_id": DATASET_ID,
        "feature_manifest_id": FEATURE_MANIFEST_ID,
        "input": {
            **tensor_metadata["input"],
            "features": list(DATASET_FEATURE_NAMES),
            "normalization": normalizer.as_dict(),
        },
        "output": tensor_metadata["output"],
        "thresholds": report["thresholds"],
        "training_subjects": list(plan.development_subjects),
        "locked_test_subjects": list(plan.locked_test_subjects),
        "training_data": {
            "file_name": Path(input_path).name,
            "sha256": input_checksum,
            "rows": len(rows),
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "tensorflow": tf.__version__,
        },
    }
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "model.tflite").write_bytes(model_bytes)
    (destination / "model-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (destination / "training-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the real-PPG short-vector autoencoder"
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    metadata = train_vector_autoencoder(
        args.input,
        args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
