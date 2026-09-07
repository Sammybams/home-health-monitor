from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import tempfile
from typing import Any, Iterable
import uuid

from .autoencoder import FEATURE_MANIFEST_ID, MODEL_FEATURES
from .windowing import FEATURE_NAMES, STEP_COUNT


class TrainingDataError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class TrainingWindow:
    subject_id: str
    values: tuple[tuple[float, ...], ...]
    masks: tuple[tuple[float, ...], ...]


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    train: tuple[TrainingWindow, ...]
    validation: tuple[TrainingWindow, ...]
    test: tuple[TrainingWindow, ...]
    train_subjects: tuple[str, ...]
    validation_subjects: tuple[str, ...]
    test_subjects: tuple[str, ...]


def _matrix(value: Any, name: str, line_number: int) -> tuple[tuple[float, ...], ...]:
    if not isinstance(value, list) or len(value) != STEP_COUNT:
        raise TrainingDataError(f"row {line_number} {name} must contain exactly 288 steps")
    result = []
    for step in value:
        if not isinstance(step, list) or len(step) != len(FEATURE_NAMES):
            raise TrainingDataError(
                f"row {line_number} {name} steps must contain {len(FEATURE_NAMES)} features"
            )
        values = []
        for item in step:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise TrainingDataError(f"row {line_number} {name} values must be numbers")
            number = float(item)
            if not math.isfinite(number):
                raise TrainingDataError(f"row {line_number} {name} values must be finite")
            values.append(number)
        result.append(tuple(values))
    return tuple(result)


def load_windows(path: str | Path) -> tuple[TrainingWindow, ...]:
    rows = []
    try:
        handle = Path(path).open(encoding="utf-8")
    except OSError as exc:
        raise TrainingDataError(f"could not open training windows: {exc}") from exc
    with handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TrainingDataError(f"row {line_number} is not valid JSON") from exc
            if not isinstance(item, dict) or item.get("schema_version") != 1:
                raise TrainingDataError(f"row {line_number} has an unsupported schema")
            if item.get("decision") != "normal":
                raise TrainingDataError("autoencoder training data must be normal-only")
            if item.get("feature_manifest_id") != FEATURE_MANIFEST_ID:
                raise TrainingDataError(f"row {line_number} has the wrong feature manifest")
            if tuple(item.get("feature_names", ())) != FEATURE_NAMES:
                raise TrainingDataError(f"row {line_number} has the wrong feature order")
            subject_id = item.get("subject_id")
            if not isinstance(subject_id, str) or not subject_id or len(subject_id) > 128:
                raise TrainingDataError(f"row {line_number} needs a valid subject_id")
            values = _matrix(item.get("values"), "values", line_number)
            masks = _matrix(item.get("masks"), "masks", line_number)
            if any(value not in (0.0, 1.0) for step in masks for value in step):
                raise TrainingDataError(f"row {line_number} masks must contain only 0 or 1")
            if not any(value == 1.0 for step in masks for value in step):
                raise TrainingDataError(f"row {line_number} masks contain no observed values")
            rows.append(TrainingWindow(subject_id, values, masks))
    if not rows:
        raise TrainingDataError("training file contains no windows")
    return tuple(rows)


def grouped_split(
    rows: Iterable[TrainingWindow], *, seed: int = 42
) -> DatasetSplit:
    rows = tuple(rows)
    subjects = sorted({row.subject_id for row in rows})
    if len(subjects) < 3:
        raise TrainingDataError("training requires at least three distinct subjects")
    random.Random(seed).shuffle(subjects)
    validation_count = max(1, round(len(subjects) * 0.15))
    test_count = max(1, round(len(subjects) * 0.15))
    if validation_count + test_count >= len(subjects):
        validation_count = test_count = 1
    test_subjects = tuple(sorted(subjects[:test_count]))
    validation_subjects = tuple(
        sorted(subjects[test_count : test_count + validation_count])
    )
    train_subjects = tuple(sorted(subjects[test_count + validation_count :]))
    train_set = set(train_subjects)
    validation_set = set(validation_subjects)
    test_set = set(test_subjects)
    return DatasetSplit(
        train=tuple(row for row in rows if row.subject_id in train_set),
        validation=tuple(row for row in rows if row.subject_id in validation_set),
        test=tuple(row for row in rows if row.subject_id in test_set),
        train_subjects=train_subjects,
        validation_subjects=validation_subjects,
        test_subjects=test_subjects,
    )


def _arrays(rows: tuple[TrainingWindow, ...], np: Any) -> Any:
    values = np.asarray([row.values for row in rows], dtype=np.float32)
    masks = np.asarray([row.masks for row in rows], dtype=np.float32)
    return np.concatenate([values, masks], axis=-1)


def _build_model(tf: Any) -> Any:
    inputs = tf.keras.Input(shape=(STEP_COUNT, len(MODEL_FEATURES)), name="window")
    values = inputs[:, :, : len(FEATURE_NAMES)]
    masks = inputs[:, :, len(FEATURE_NAMES) :]
    encoded = tf.keras.layers.Conv1D(12, 5, padding="same", activation="relu")(values)
    encoded = tf.keras.layers.Conv1D(8, 5, strides=2, padding="same", activation="relu")(
        encoded
    )
    encoded = tf.keras.layers.Conv1D(4, 3, strides=2, padding="same", activation="relu")(
        encoded
    )
    decoded = tf.keras.layers.UpSampling1D(2)(encoded)
    decoded = tf.keras.layers.Conv1D(8, 3, padding="same", activation="relu")(decoded)
    decoded = tf.keras.layers.UpSampling1D(2)(decoded)
    decoded = tf.keras.layers.Conv1D(12, 5, padding="same", activation="relu")(decoded)
    reconstructed = tf.keras.layers.Conv1D(
        len(FEATURE_NAMES), 5, padding="same", name="reconstruction"
    )(decoded)
    observed_reconstruction = tf.keras.layers.Multiply()([reconstructed, masks])
    outputs = tf.keras.layers.Concatenate(name="masked_reconstruction")(
        [observed_reconstruction, masks]
    )
    model = tf.keras.Model(inputs, outputs, name="home_gateway_autoencoder")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    return model


def _errors(expected: Any, reconstructed: Any, np: Any) -> Any:
    feature_count = len(FEATURE_NAMES)
    values = expected[:, :, :feature_count]
    masks = expected[:, :, feature_count:]
    output = reconstructed[:, :, :feature_count]
    squared = ((values - output) ** 2) * masks
    counts = np.maximum(masks.sum(axis=(1, 2)), 1.0)
    return squared.sum(axis=(1, 2)) / counts


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise TrainingDataError("threshold selection requires validation errors")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def select_thresholds(errors: Iterable[float]) -> tuple[float, float]:
    values = [float(value) for value in errors]
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise TrainingDataError("threshold errors must be finite non-negative values")
    persistent = max(_quantile(values, 0.99), 1e-8)
    severe = max(_quantile(values, 0.999), persistent * 2.0)
    return persistent, severe


def summarize_scores(errors: Iterable[float], *, threshold: float) -> dict[str, float]:
    values = [float(value) for value in errors]
    if not values:
        raise TrainingDataError("score summary requires at least one error")
    return {
        "count": len(values),
        "minimum": min(values),
        "mean": sum(values) / len(values),
        "maximum": max(values),
        "fraction_at_or_above_threshold": sum(value >= threshold for value in values)
        / len(values),
    }


def _engineering_anomalies(test_data: Any, np: Any) -> tuple[Any, list[str]]:
    result = test_data.copy()
    scenarios = (
        ("sustained_high_heart_rate", 0, 3.8),
        ("sustained_low_spo2", 1, -4.2),
        ("sustained_high_temperature", 2, 4.0),
        ("sustained_motion_change", 3, 4.5),
    )
    names = []
    for index in range(len(result)):
        name, feature, shift = scenarios[index % len(scenarios)]
        start = 120 + index % 24
        result[index, start : start + 48, feature] += shift
        names.append(name)
    return result, names


def _quantized_predictions(interpreter: Any, data: Any, np: Any) -> Any:
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    input_scale, input_zero_point = input_detail["quantization"]
    output_scale, output_zero_point = output_detail["quantization"]
    predictions = []
    for item in data:
        quantized = np.clip(
            np.rint(item / input_scale) + input_zero_point, -128, 127
        ).astype(np.int8)
        interpreter.set_tensor(input_detail["index"], quantized[np.newaxis, ...])
        interpreter.invoke()
        output = interpreter.get_tensor(output_detail["index"])[0]
        predictions.append((output.astype(np.float32) - output_zero_point) * output_scale)
    return np.asarray(predictions, dtype=np.float32)


def train_and_export(
    input_path: str | Path,
    output_directory: str | Path,
    *,
    epochs: int = 30,
    batch_size: int = 16,
    seed: int = 42,
    artifact_role: str = "candidate",
) -> dict[str, Any]:
    try:
        import numpy as np
        import tensorflow as tf
    except ImportError as exc:
        raise SystemExit(
            "gateway training requires Python 3.9-3.12 and: "
            "python -m pip install -e '.[gateway-train]'"
        ) from exc

    rows = load_windows(input_path)
    split = grouped_split(rows, seed=seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    train_data = _arrays(split.train, np)
    validation_data = _arrays(split.validation, np)
    test_data = _arrays(split.test, np)

    model = _build_model(tf)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )
    ]
    history = model.fit(
        train_data,
        train_data,
        validation_data=(validation_data, validation_data),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=2,
    )

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_dataset():
        for item in train_data[: min(100, len(train_data))]:
            yield [item[np.newaxis, ...].astype(np.float32)]

    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()

    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    input_scale, input_zero_point = input_detail["quantization"]
    output_scale, output_zero_point = output_detail["quantization"]
    simulated_data, simulated_scenarios = _engineering_anomalies(test_data, np)
    validation_reconstructions = _quantized_predictions(interpreter, validation_data, np)
    test_reconstructions = _quantized_predictions(interpreter, test_data, np)
    simulated_reconstructions = _quantized_predictions(interpreter, simulated_data, np)
    validation_errors = _errors(validation_data, validation_reconstructions, np)
    test_errors = _errors(test_data, test_reconstructions, np)
    simulated_errors = _errors(simulated_data, simulated_reconstructions, np)
    persistent_threshold, severe_threshold = select_thresholds(validation_errors)
    checksum = hashlib.sha256(tflite_model).hexdigest()
    metadata = {
        "schema_version": 1,
        "model_id": f"gateway-ae-{uuid.uuid4().hex[:12]}",
        "artifact_role": artifact_role,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_manifest_id": FEATURE_MANIFEST_ID,
        "model_sha256": checksum,
        "input": {
            "shape": [int(value) for value in input_detail["shape"]],
            "features": list(MODEL_FEATURES),
            "dtype": "int8",
            "scale": float(input_scale),
            "zero_point": int(input_zero_point),
        },
        "output": {
            "shape": [int(value) for value in output_detail["shape"]],
            "dtype": "int8",
            "scale": float(output_scale),
            "zero_point": int(output_zero_point),
        },
        "thresholds": {
            "persistent_error": persistent_threshold,
            "severe_error": severe_threshold,
            "consecutive_windows": 2,
        },
        "dataset": {
            "normal_only": True,
            "windows": len(rows),
            "train_windows": len(split.train),
            "validation_windows": len(split.validation),
            "test_windows": len(split.test),
            "train_subjects": list(split.train_subjects),
            "validation_subjects": list(split.validation_subjects),
            "test_subjects": list(split.test_subjects),
        },
        "metrics": {
            "validation_error_p99": persistent_threshold,
            "test_mean_reconstruction_error": float(np.mean(test_errors)),
            "test_false_anomaly_fraction": float(
                np.mean(test_errors >= persistent_threshold)
            ),
            "engineering_simulated_anomaly_detection_fraction": float(
                np.mean(simulated_errors >= persistent_threshold)
            ),
        },
    }
    if len(tflite_model) >= 1024 * 1024:
        raise TrainingDataError("exported model exceeds the 1 MB gateway release limit")
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "model.tflite").write_bytes(tflite_model)
    (destination / "model-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    feature_count = len(FEATURE_NAMES)
    report = {
        "schema_version": 1,
        "artifact_role": artifact_role,
        "model_id": metadata["model_id"],
        "threshold": persistent_threshold,
        "history": {
            name: [float(value) for value in values]
            for name, values in history.history.items()
        },
        "normal_validation_scores": [float(value) for value in validation_errors],
        "normal_test_scores": [float(value) for value in test_errors],
        "simulated_anomaly_scores": [float(value) for value in simulated_errors],
        "simulated_anomaly_scenarios": simulated_scenarios,
        "summaries": {
            "normal_validation": summarize_scores(
                validation_errors, threshold=persistent_threshold
            ),
            "normal_test": summarize_scores(test_errors, threshold=persistent_threshold),
            "engineering_simulation": summarize_scores(
                simulated_errors, threshold=persistent_threshold
            ),
        },
        "example": {
            "feature_names": list(FEATURE_NAMES[:feature_count]),
            "normal_input": test_data[0, :, :feature_count].astype(float).tolist(),
            "normal_reconstruction": test_reconstructions[0, :, :feature_count]
            .astype(float)
            .tolist(),
            "simulated_input": simulated_data[0, :, :feature_count]
            .astype(float)
            .tolist(),
            "simulated_reconstruction": simulated_reconstructions[0, :, :feature_count]
            .astype(float)
            .tolist(),
            "simulated_scenario": simulated_scenarios[0],
        },
    }
    (destination / "training-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    from .autoencoder import AutoencoderModel

    AutoencoderModel.load(
        destination / "model.tflite",
        destination / "model-metadata.json",
        interpreter_factory=tf.lite.Interpreter,
    )
    return metadata


def _smoke_rows(path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for subject_index in range(3):
            for window_index in range(2):
                values = []
                for step in range(STEP_COUNT):
                    values.append(
                        [
                            0.05 * math.sin((step + subject_index + window_index) / (8 + feature))
                            for feature in range(len(FEATURE_NAMES))
                        ]
                    )
                item = {
                    "schema_version": 1,
                    "subject_id": f"smoke-{subject_index}",
                    "feature_manifest_id": FEATURE_MANIFEST_ID,
                    "decision": "normal",
                    "feature_names": list(FEATURE_NAMES),
                    "values": values,
                    "masks": [[1.0] * len(FEATURE_NAMES) for _ in range(STEP_COUNT)],
                }
                handle.write(json.dumps(item) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the normal-only gateway autoencoder")
    parser.add_argument("input", type=Path, nargs="?")
    parser.add_argument("output", type=Path, nargs="?", default=Path("artifacts/gateway"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--artifact-role", choices=("candidate", "development_demo"), default="candidate"
    )
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch size must be positive")
    if args.smoke_test:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "smoke.jsonl"
            _smoke_rows(source)
            metadata = train_and_export(
                source,
                args.output,
                epochs=1,
                batch_size=2,
                seed=args.seed,
                artifact_role="development_demo",
            )
    elif args.input is None:
        parser.error("input is required unless --smoke-test is used")
    else:
        metadata = train_and_export(
            args.input,
            args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=args.seed,
            artifact_role=args.artifact_role,
        )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
