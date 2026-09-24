from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Callable, Iterable, Mapping, Sequence

from home_health_monitor.datasets.pulse_transit_build import FEATURE_MANIFEST_ID
from home_health_monitor.datasets.pulse_transit_features import DATASET_FEATURE_NAMES

from .autoencoder import (
    InterpreterProtocol,
    ModelError,
    TensorContract,
    _runtime_factory,
    _runtime_tensor,
    _shape,
    _validate_runtime_quantization,
)


EXPECTED_VECTORS_PER_INTERVAL = 16
MINIMUM_PERSONAL_CALIBRATION_INTERVALS = 288


@dataclass(frozen=True, slots=True)
class VectorReconstruction:
    overall_error: float
    feature_errors: dict[str, float]


@dataclass(frozen=True, slots=True)
class IntervalPrediction:
    decision: str
    score: float
    threshold: float
    severe_threshold: float
    above_threshold: bool
    severe: bool
    persistent: bool
    vector_count: int
    expected_vector_count: int
    coverage: float
    contributing_signals: tuple[str, ...]


def _finite(value: Any, name: str, *, positive: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ModelError(f"invalid {name}") from exc
    if not math.isfinite(number) or (positive and number <= 0):
        raise ModelError(f"invalid {name}")
    return number


def _contract(raw: Any, name: str) -> TensorContract:
    if not isinstance(raw, dict) or _shape(raw.get("shape")) != (
        1,
        len(DATASET_FEATURE_NAMES),
    ):
        raise ModelError(f"invalid vector {name} shape")
    if raw.get("dtype") != "int8":
        raise ModelError(f"vector {name} dtype must be int8")
    scale = _finite(raw.get("scale"), f"vector {name} scale", positive=True)
    zero_point = raw.get("zero_point")
    if isinstance(zero_point, bool) or not isinstance(zero_point, int) or not -128 <= zero_point <= 127:
        raise ModelError(f"invalid vector {name} zero point")
    return TensorContract((1, len(DATASET_FEATURE_NAMES)), "int8", scale, zero_point)


class VectorAutoencoderModel:
    def __init__(
        self,
        *,
        model_id: str,
        interpreter: InterpreterProtocol,
        input_contract: TensorContract,
        output_contract: TensorContract,
        center: tuple[float, ...],
        scale: tuple[float, ...],
        persistent_threshold: float,
        severe_threshold: float,
        input_index: int,
        output_index: int,
    ) -> None:
        self.model_id = model_id
        self.interpreter = interpreter
        self.input_contract = input_contract
        self.output_contract = output_contract
        self.center = center
        self.scale = scale
        self.persistent_threshold = persistent_threshold
        self.severe_threshold = severe_threshold
        self.input_index = input_index
        self.output_index = output_index

    @classmethod
    def load(
        cls,
        model_path: str | Path,
        metadata_path: str | Path,
        *,
        interpreter_factory: Callable[..., InterpreterProtocol] | None = None,
    ) -> "VectorAutoencoderModel":
        try:
            model_bytes = Path(model_path).read_bytes()
            metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelError(f"could not load vector autoencoder: {exc}") from exc
        if hashlib.sha256(model_bytes).hexdigest() != metadata.get("model_sha256"):
            raise ModelError("vector model checksum does not match metadata")
        factory = interpreter_factory or _runtime_factory()
        try:
            interpreter = factory(model_path=str(model_path))
        except Exception as exc:
            raise ModelError(f"could not initialize vector interpreter: {exc}") from exc
        return cls.from_interpreter(metadata, interpreter)

    @classmethod
    def from_interpreter(
        cls, metadata: dict[str, Any], interpreter: InterpreterProtocol
    ) -> "VectorAutoencoderModel":
        try:
            if metadata.get("schema_version") != 1:
                raise ModelError("unsupported vector model metadata schema")
            if metadata.get("feature_manifest_id") != FEATURE_MANIFEST_ID:
                raise ModelError("vector model feature manifest does not match")
            raw_input = metadata["input"]
            if tuple(raw_input.get("features", ())) != DATASET_FEATURE_NAMES:
                raise ModelError("vector model feature order does not match")
            input_contract = _contract(raw_input, "input")
            output_contract = _contract(metadata["output"], "output")
            normalization = raw_input["normalization"]
            if tuple(normalization.get("feature_names", ())) != DATASET_FEATURE_NAMES:
                raise ModelError("vector normalization feature order does not match")
            center = tuple(
                _finite(value, "normalization center")
                for value in normalization["center"]
            )
            scale = tuple(
                _finite(value, "normalization scale", positive=True)
                for value in normalization["scale"]
            )
            if len(center) != len(DATASET_FEATURE_NAMES) or len(scale) != len(
                DATASET_FEATURE_NAMES
            ):
                raise ModelError("vector normalization has the wrong size")
            thresholds = metadata["thresholds"]
            persistent = _finite(
                thresholds["persistent_error"], "persistent threshold", positive=True
            )
            severe = _finite(
                thresholds["severe_error"], "severe threshold", positive=True
            )
            if severe <= persistent:
                raise ModelError("severe threshold must exceed persistent threshold")
            model_id = metadata["model_id"]
            if not isinstance(model_id, str) or not model_id:
                raise ModelError("vector model_id must be non-empty")
            interpreter.allocate_tensors()
            inputs = interpreter.get_input_details()
            outputs = interpreter.get_output_details()
            if len(inputs) != 1 or _shape(inputs[0].get("shape")) != input_contract.shape:
                raise ModelError("vector interpreter input shape does not match")
            if len(outputs) != 1 or _shape(outputs[0].get("shape")) != output_contract.shape:
                raise ModelError("vector interpreter output shape does not match")
            _validate_runtime_quantization(inputs[0], input_contract, "input")
            _validate_runtime_quantization(outputs[0], output_contract, "output")
        except ModelError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelError(f"invalid vector model metadata: {exc}") from exc
        return cls(
            model_id=model_id,
            interpreter=interpreter,
            input_contract=input_contract,
            output_contract=output_contract,
            center=center,
            scale=scale,
            persistent_threshold=persistent,
            severe_threshold=severe,
            input_index=int(inputs[0]["index"]),
            output_index=int(outputs[0]["index"]),
        )

    def _ordered_values(
        self, vector: Mapping[str, float] | Sequence[float]
    ) -> list[float]:
        if isinstance(vector, Mapping):
            if set(vector) != set(DATASET_FEATURE_NAMES):
                raise ModelError("vector fields do not match feature manifest")
            raw = [vector[name] for name in DATASET_FEATURE_NAMES]
        else:
            raw = list(vector)
        if len(raw) != len(DATASET_FEATURE_NAMES):
            raise ModelError("vector has the wrong feature count")
        values = [_finite(value, "vector feature") for value in raw]
        valid_index = DATASET_FEATURE_NAMES.index("heart_rate_valid")
        if values[valid_index] not in (0.0, 1.0):
            raise ModelError("heart_rate_valid must be binary")
        return values

    def predict(
        self, vector: Mapping[str, float] | Sequence[float]
    ) -> VectorReconstruction:
        values = self._ordered_values(vector)
        normalized = [
            (value - center) / scale
            for value, center, scale in zip(values, self.center, self.scale)
        ]
        if values[DATASET_FEATURE_NAMES.index("heart_rate_valid")] == 0.0:
            for name in ("heart_rate_bpm", "ppg_rr_interval_std_ms"):
                normalized[DATASET_FEATURE_NAMES.index(name)] = 0.0
        quantized = [[
            max(
                -128,
                min(
                    127,
                    round(value / self.input_contract.scale)
                    + self.input_contract.zero_point,
                ),
            )
            for value in normalized
        ]]
        self.interpreter.set_tensor(self.input_index, _runtime_tensor(quantized))
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_index)[0]
        feature_errors = {}
        for index, name in enumerate(DATASET_FEATURE_NAMES):
            reconstructed = (
                float(output[index]) - self.output_contract.zero_point
            ) * self.output_contract.scale
            feature_errors[name] = (normalized[index] - reconstructed) ** 2
        return VectorReconstruction(
            overall_error=fmean(feature_errors.values()),
            feature_errors=feature_errors,
        )

    def aggregate(
        self,
        results: Iterable[VectorReconstruction],
        *,
        previous_interval_above_threshold: bool = False,
        personal_threshold: float | None = None,
    ) -> IntervalPrediction:
        items = tuple(results)
        if not items:
            raise ModelError("interval aggregation requires at least one vector")
        threshold = (
            self.persistent_threshold
            if personal_threshold is None
            else _finite(personal_threshold, "personal threshold", positive=True)
        )
        severe_threshold = max(self.severe_threshold, threshold * 2.0)
        ordered = sorted(item.overall_error for item in items)
        position = (len(ordered) - 1) * 0.95
        lower = math.floor(position)
        upper = math.ceil(position)
        score = ordered[lower] if lower == upper else (
            ordered[lower] * (upper - position)
            + ordered[upper] * (position - lower)
        )
        above = score >= threshold
        severe = score >= severe_threshold
        persistent = above and previous_interval_above_threshold
        mean_feature_errors = {
            name: fmean(item.feature_errors[name] for item in items)
            for name in DATASET_FEATURE_NAMES
        }
        highest = max(mean_feature_errors.values())
        contributing = tuple(
            name for name, value in mean_feature_errors.items() if value == highest
        )
        return IntervalPrediction(
            decision="anomaly" if severe or persistent else "normal",
            score=score,
            threshold=threshold,
            severe_threshold=severe_threshold,
            above_threshold=above,
            severe=severe,
            persistent=persistent,
            vector_count=len(items),
            expected_vector_count=EXPECTED_VECTORS_PER_INTERVAL,
            coverage=min(1.0, len(items) / EXPECTED_VECTORS_PER_INTERVAL),
            contributing_signals=contributing,
        )


def calibrate_personal_reconstruction_threshold(
    normal_interval_scores: Iterable[float], *, elapsed_hours: float
) -> float:
    elapsed = _finite(elapsed_hours, "calibration elapsed hours")
    if elapsed < 48.0:
        raise ModelError("personal calibration requires 48 elapsed hours")
    values = [
        _finite(value, "calibration reconstruction score")
        for value in normal_interval_scores
    ]
    if len(values) < MINIMUM_PERSONAL_CALIBRATION_INTERVALS:
        raise ModelError(
            f"personal calibration requires {MINIMUM_PERSONAL_CALIBRATION_INTERVALS} intervals"
        )
    threshold = fmean(values) + 3.0 * stdev(values)
    return max(threshold, 1e-8)
