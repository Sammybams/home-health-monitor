from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from home_health_monitor.gateway.autoencoder import AutoencoderModel, ModelError, _runtime_factory
from home_health_monitor.gateway.windowing import FEATURE_NAMES, ModelWindow


MODEL_FEATURES = FEATURE_NAMES + tuple(f"mask_{name}" for name in FEATURE_NAMES)


def metadata(checksum: str = "0" * 64) -> dict:
    return {
        "schema_version": 1,
        "model_id": "gateway-ae-test",
        "feature_manifest_id": "features-v1",
        "model_sha256": checksum,
        "input": {
            "shape": [1, 288, len(MODEL_FEATURES)],
            "features": list(MODEL_FEATURES),
            "dtype": "int8",
            "scale": 0.1,
            "zero_point": 0,
        },
        "output": {
            "shape": [1, 288, len(MODEL_FEATURES)],
            "dtype": "int8",
            "scale": 0.1,
            "zero_point": 0,
        },
        "thresholds": {
            "persistent_error": 0.25,
            "severe_error": 0.75,
            "consecutive_windows": 2,
        },
    }


def window() -> ModelWindow:
    values = tuple(
        tuple(1.0 if feature == 0 else 0.0 for feature in range(len(FEATURE_NAMES)))
        for _ in range(288)
    )
    masks = tuple(tuple(1.0 for _ in FEATURE_NAMES) for _ in range(288))
    return ModelWindow(
        feature_names=FEATURE_NAMES,
        values=values,
        masks=masks,
        started_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
        ended_at=datetime(2026, 9, 2, tzinfo=timezone.utc),
        step_minutes=5,
    )


class FakeInterpreter:
    def __init__(self, *, identity: bool = True) -> None:
        self.identity = identity
        self.tensor = None

    def allocate_tensors(self) -> None:
        pass

    def get_input_details(self):
        return [{"index": 0, "shape": [1, 288, len(MODEL_FEATURES)]}]

    def get_output_details(self):
        return [{"index": 1, "shape": [1, 288, len(MODEL_FEATURES)]}]

    def set_tensor(self, index, value) -> None:
        self.tensor = value

    def invoke(self) -> None:
        pass

    def get_tensor(self, index):
        if self.identity:
            return self.tensor
        return [[[0 for _ in MODEL_FEATURES] for _ in range(288)]]


class GatewayAutoencoderTests(unittest.TestCase):
    def test_runtime_factory_supports_tensorflow_lite_attribute(self) -> None:
        marker = object()
        tensorflow = SimpleNamespace(lite=SimpleNamespace(Interpreter=marker))

        with patch.dict("sys.modules", {"tensorflow": tensorflow}):
            factory = _runtime_factory()

        self.assertIs(marker, factory)

    def test_identity_reconstruction_has_zero_error(self) -> None:
        model = AutoencoderModel.from_interpreter(metadata(), FakeInterpreter())

        result = model.predict(window())

        self.assertAlmostEqual(0.0, result.overall_error)

    def test_reconstruction_result_reports_feature_errors(self) -> None:
        model = AutoencoderModel.from_interpreter(
            metadata(), FakeInterpreter(identity=False)
        )

        result = model.predict(window())

        self.assertGreater(result.overall_error, 0)
        self.assertEqual(set(FEATURE_NAMES), set(result.feature_errors))

    def test_loader_rejects_feature_manifest_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.tflite"
            model_path.write_bytes(b"tiny model")
            item = metadata(hashlib.sha256(b"tiny model").hexdigest())
            item["feature_manifest_id"] = "wrong-features"
            metadata_path = Path(directory) / "model-metadata.json"
            metadata_path.write_text(json.dumps(item), encoding="utf-8")

            with self.assertRaisesRegex(ModelError, "feature manifest"):
                AutoencoderModel.load(
                    model_path,
                    metadata_path,
                    interpreter_factory=lambda **_: FakeInterpreter(),
                )

    def test_loader_rejects_model_checksum_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.tflite"
            model_path.write_bytes(b"tiny model")
            metadata_path = Path(directory) / "model-metadata.json"
            metadata_path.write_text(json.dumps(metadata()), encoding="utf-8")

            with self.assertRaisesRegex(ModelError, "checksum"):
                AutoencoderModel.load(
                    model_path,
                    metadata_path,
                    interpreter_factory=lambda **_: FakeInterpreter(),
                )

    def test_invalid_interpreter_shape_is_rejected(self) -> None:
        interpreter = FakeInterpreter()
        interpreter.get_input_details = lambda: [{"index": 0, "shape": [1, 10, 2]}]

        with self.assertRaisesRegex(ModelError, "input shape"):
            AutoencoderModel.from_interpreter(metadata(), interpreter)

    def test_interpreter_quantization_must_match_metadata(self) -> None:
        interpreter = FakeInterpreter()
        interpreter.get_input_details = lambda: [{
            "index": 0,
            "shape": [1, 288, len(MODEL_FEATURES)],
            "quantization": (0.2, 0),
        }]

        with self.assertRaisesRegex(ModelError, "input quantization"):
            AutoencoderModel.from_interpreter(metadata(), interpreter)

    def test_window_feature_order_must_match(self) -> None:
        model = AutoencoderModel.from_interpreter(metadata(), FakeInterpreter())
        item = window()
        wrong = ModelWindow(
            feature_names=tuple(reversed(item.feature_names)),
            values=item.values,
            masks=item.masks,
            started_at=item.started_at,
            ended_at=item.ended_at,
            step_minutes=item.step_minutes,
        )

        with self.assertRaisesRegex(ModelError, "feature order"):
            model.predict(wrong)


if __name__ == "__main__":
    unittest.main()
