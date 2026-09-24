from __future__ import annotations

import unittest

import numpy as np

from home_health_monitor.datasets.pulse_transit_build import FEATURE_MANIFEST_ID
from home_health_monitor.datasets.pulse_transit_features import DATASET_FEATURE_NAMES
from home_health_monitor.gateway.autoencoder import ModelError
from home_health_monitor.gateway.vector_autoencoder import (
    MINIMUM_PERSONAL_CALIBRATION_INTERVALS,
    VectorAutoencoderModel,
    VectorReconstruction,
    calibrate_personal_reconstruction_threshold,
)


class IdentityInterpreter:
    def allocate_tensors(self) -> None:
        self.value = None

    def get_input_details(self):
        return [{"shape": [1, 12], "index": 1, "quantization": (0.1, 0)}]

    def get_output_details(self):
        return [{"shape": [1, 12], "index": 2, "quantization": (0.1, 0)}]

    def set_tensor(self, index, value):
        self.value = value

    def invoke(self):
        pass

    def get_tensor(self, index):
        return self.value


def metadata():
    tensor = {"shape": [1, 12], "dtype": "int8", "scale": 0.1, "zero_point": 0}
    return {
        "schema_version": 1,
        "model_id": "test-vector-model",
        "feature_manifest_id": FEATURE_MANIFEST_ID,
        "input": {
            **tensor,
            "features": list(DATASET_FEATURE_NAMES),
            "normalization": {
                "feature_names": list(DATASET_FEATURE_NAMES),
                "center": [0.0] * 12,
                "scale": [1.0] * 12,
            },
        },
        "output": tensor,
        "thresholds": {"persistent_error": 1.0, "severe_error": 3.0},
    }


class VectorAutoencoderTests(unittest.TestCase):
    def test_identity_interpreter_has_only_quantization_rounding_error(self) -> None:
        model = VectorAutoencoderModel.from_interpreter(metadata(), IdentityInterpreter())
        values = [0.2] * 12
        values[DATASET_FEATURE_NAMES.index("heart_rate_valid")] = 1.0

        result = model.predict(values)

        self.assertAlmostEqual(0.0, result.overall_error, places=6)

    def test_missing_heart_rate_is_imputed_but_prediction_still_exists(self) -> None:
        model = VectorAutoencoderModel.from_interpreter(metadata(), IdentityInterpreter())
        values = [0.2] * 12
        values[DATASET_FEATURE_NAMES.index("heart_rate_bpm")] = 0.0
        values[DATASET_FEATURE_NAMES.index("heart_rate_valid")] = 0.0
        values[DATASET_FEATURE_NAMES.index("ppg_rr_interval_std_ms")] = 0.0

        result = model.predict(values)

        self.assertTrue(np.isfinite(result.overall_error))

    def test_interval_requires_persistence_unless_score_is_severe(self) -> None:
        model = VectorAutoencoderModel.from_interpreter(metadata(), IdentityInterpreter())
        item = VectorReconstruction(1.5, {name: 1.5 for name in DATASET_FEATURE_NAMES})

        first = model.aggregate([item] * 16)
        second = model.aggregate(
            [item] * 16, previous_interval_above_threshold=True
        )
        severe = model.aggregate(
            [VectorReconstruction(4.0, item.feature_errors)] * 16
        )

        self.assertEqual("normal", first.decision)
        self.assertTrue(first.above_threshold)
        self.assertEqual("anomaly", second.decision)
        self.assertTrue(second.persistent)
        self.assertEqual("anomaly", severe.decision)
        self.assertTrue(severe.severe)

    def test_personal_threshold_requires_48_hour_interval_count(self) -> None:
        with self.assertRaisesRegex(ModelError, "requires 288"):
            calibrate_personal_reconstruction_threshold([0.1] * 287)

        threshold = calibrate_personal_reconstruction_threshold(
            [0.1] * (MINIMUM_PERSONAL_CALIBRATION_INTERVALS - 1) + [0.2]
        )

        self.assertGreater(threshold, 0.1)


if __name__ == "__main__":
    unittest.main()
