from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Protocol

from .windowing import FEATURE_NAMES, ModelWindow, STEP_COUNT


FEATURE_MANIFEST_ID = "features-v1"
MODEL_FEATURES = FEATURE_NAMES + tuple(f"mask_{name}" for name in FEATURE_NAMES)


class ModelError(RuntimeError):
    pass


class InterpreterProtocol(Protocol):
    def allocate_tensors(self) -> None: ...
    def get_input_details(self) -> list[dict[str, Any]]: ...
    def get_output_details(self) -> list[dict[str, Any]]: ...
    def set_tensor(self, index: int, value: Any) -> None: ...
    def invoke(self) -> None: ...
    def get_tensor(self, index: int) -> Any: ...


@dataclass(frozen=True, slots=True)
class TensorContract:
    shape: tuple[int, ...]
    dtype: str
    scale: float
    zero_point: int


@dataclass(frozen=True, slots=True)
class ReconstructionResult:
    overall_error: float
    feature_errors: dict[str, float]
    persistent_threshold: float
    severe_threshold: float
    is_above_threshold: bool
    is_severe: bool


def _shape(value: Any) -> tuple[int, ...]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    try:
        return tuple(int(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ModelError("invalid tensor shape") from exc


def _finite_number(value: Any, name: str, *, positive: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ModelError(f"invalid {name}") from exc
    if not math.isfinite(number) or (positive and number <= 0):
        raise ModelError(f"invalid {name}")
    return number


def _tensor_contract(item: Any, name: str) -> TensorContract:
    if not isinstance(item, dict):
        raise ModelError(f"invalid {name} tensor metadata")
    shape = _shape(item.get("shape"))
    if shape != (1, STEP_COUNT, len(MODEL_FEATURES)):
        raise ModelError(f"invalid {name} shape")
    if item.get("dtype") != "int8":
        raise ModelError(f"{name} dtype must be int8")
    scale = _finite_number(item.get("scale"), f"{name} quantization scale", positive=True)
    zero_point = item.get("zero_point")
    if isinstance(zero_point, bool) or not isinstance(zero_point, int) or not -128 <= zero_point <= 127:
        raise ModelError(f"invalid {name} quantization zero point")
    return TensorContract(shape, "int8", scale, zero_point)


def _runtime_factory() -> Callable[..., InterpreterProtocol]:
    try:
        from ai_edge_litert.interpreter import Interpreter

        return Interpreter
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter

            return Interpreter
        except ImportError:
            try:
                import tensorflow as tf

                return tf.lite.Interpreter
            except ImportError as exc:
                raise ModelError("no LiteRT interpreter is installed") from exc


def _runtime_tensor(value: list[list[list[int]]]) -> Any:
    try:
        import numpy

        return numpy.asarray(value, dtype=numpy.int8)
    except ImportError:
        return value


def _validate_runtime_quantization(
    detail: dict[str, Any], contract: TensorContract, name: str
) -> None:
    quantization = detail.get("quantization")
    if quantization is None:
        return
    try:
        scale, zero_point = quantization
        matches = math.isclose(float(scale), contract.scale, rel_tol=1e-7) and int(
            zero_point
        ) == contract.zero_point
    except (TypeError, ValueError):
        matches = False
    if not matches:
        raise ModelError(f"interpreter {name} quantization does not match metadata")


class AutoencoderModel:
    def __init__(
        self,
        *,
        model_id: str,
        interpreter: InterpreterProtocol,
        input_contract: TensorContract,
        output_contract: TensorContract,
        persistent_threshold: float,
        severe_threshold: float,
        consecutive_windows: int,
    ) -> None:
        self.model_id = model_id
        self.interpreter = interpreter
        self.input_contract = input_contract
        self.output_contract = output_contract
        self.persistent_threshold = persistent_threshold
        self.severe_threshold = severe_threshold
        self.consecutive_windows = consecutive_windows
        self._input_index = 0
        self._output_index = 0

    @classmethod
    def load(
        cls,
        model_path: str | Path,
        metadata_path: str | Path,
        *,
        interpreter_factory: Callable[..., InterpreterProtocol] | None = None,
    ) -> "AutoencoderModel":
        try:
            model_bytes = Path(model_path).read_bytes()
            metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ModelError(f"could not load autoencoder artifact: {exc}") from exc
        if metadata.get("feature_manifest_id") != FEATURE_MANIFEST_ID:
            raise ModelError("model feature manifest does not match this gateway")
        checksum = hashlib.sha256(model_bytes).hexdigest()
        if metadata.get("model_sha256") != checksum:
            raise ModelError("model checksum does not match metadata")
        factory = interpreter_factory or _runtime_factory()
        try:
            interpreter = factory(model_path=str(model_path))
        except Exception as exc:
            raise ModelError(f"could not initialize TensorFlow Lite interpreter: {exc}") from exc
        return cls.from_interpreter(metadata, interpreter)

    @classmethod
    def from_interpreter(
        cls, metadata: dict[str, Any], interpreter: InterpreterProtocol
    ) -> "AutoencoderModel":
        try:
            if metadata.get("schema_version") != 1:
                raise ModelError("unsupported autoencoder metadata schema_version")
            if metadata.get("feature_manifest_id") != FEATURE_MANIFEST_ID:
                raise ModelError("model feature manifest does not match this gateway")
            raw_input = metadata["input"]
            if tuple(raw_input.get("features", ())) != MODEL_FEATURES:
                raise ModelError("model feature order does not match this gateway")
            input_contract = _tensor_contract(raw_input, "input")
            output_contract = _tensor_contract(metadata["output"], "output")
            thresholds = metadata["thresholds"]
            persistent = _finite_number(
                thresholds["persistent_error"], "persistent error threshold", positive=True
            )
            severe = _finite_number(
                thresholds["severe_error"], "severe error threshold", positive=True
            )
            if severe <= persistent:
                raise ModelError("severe error threshold must exceed persistent threshold")
            consecutive = thresholds["consecutive_windows"]
            if isinstance(consecutive, bool) or not isinstance(consecutive, int) or consecutive < 1:
                raise ModelError("consecutive_windows must be a positive integer")
            model_id = metadata["model_id"]
            if not isinstance(model_id, str) or not model_id:
                raise ModelError("model_id must be a non-empty string")

            interpreter.allocate_tensors()
            inputs = interpreter.get_input_details()
            outputs = interpreter.get_output_details()
            if len(inputs) != 1 or _shape(inputs[0].get("shape")) != input_contract.shape:
                raise ModelError("interpreter input shape does not match metadata")
            if len(outputs) != 1 or _shape(outputs[0].get("shape")) != output_contract.shape:
                raise ModelError("interpreter output shape does not match metadata")
            _validate_runtime_quantization(inputs[0], input_contract, "input")
            _validate_runtime_quantization(outputs[0], output_contract, "output")
        except ModelError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelError(f"invalid autoencoder metadata: {exc}") from exc

        model = cls(
            model_id=model_id,
            interpreter=interpreter,
            input_contract=input_contract,
            output_contract=output_contract,
            persistent_threshold=persistent,
            severe_threshold=severe,
            consecutive_windows=consecutive,
        )
        model._input_index = int(inputs[0]["index"])
        model._output_index = int(outputs[0]["index"])
        return model

    def predict(self, window: ModelWindow) -> ReconstructionResult:
        if window.feature_names != FEATURE_NAMES:
            raise ModelError("window feature order does not match model")
        if len(window.values) != STEP_COUNT or len(window.masks) != STEP_COUNT:
            raise ModelError("window must contain exactly 288 steps")

        raw_steps = [
            [*map(float, values), *map(float, masks)]
            for values, masks in zip(window.values, window.masks)
        ]
        quantized = [[[
            max(-128, min(127, round(value / self.input_contract.scale) + self.input_contract.zero_point))
            for value in step
        ] for step in raw_steps]]
        self.interpreter.set_tensor(self._input_index, _runtime_tensor(quantized))
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self._output_index)

        errors = {name: [] for name in FEATURE_NAMES}
        for step_index, (values, masks) in enumerate(zip(window.values, window.masks)):
            reconstructed_step = output[0][step_index]
            for feature_index, name in enumerate(FEATURE_NAMES):
                if masks[feature_index] <= 0:
                    continue
                reconstructed = (
                    float(reconstructed_step[feature_index]) - self.output_contract.zero_point
                ) * self.output_contract.scale
                errors[name].append((float(values[feature_index]) - reconstructed) ** 2)

        feature_errors = {
            name: (sum(items) / len(items) if items else 0.0)
            for name, items in errors.items()
        }
        populated = [item for name, item in feature_errors.items() if errors[name]]
        overall = sum(populated) / len(populated) if populated else 0.0
        return ReconstructionResult(
            overall_error=overall,
            feature_errors=feature_errors,
            persistent_threshold=self.persistent_threshold,
            severe_threshold=self.severe_threshold,
            is_above_threshold=overall >= self.persistent_threshold,
            is_severe=overall >= self.severe_threshold,
        )
