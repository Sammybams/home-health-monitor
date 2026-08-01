from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from home_health_monitor.features import feature_names
from home_health_monitor.model import TinyModel


class ModelTests(unittest.TestCase):
    def test_zero_weight_model_is_deterministic(self) -> None:
        names = feature_names()
        artifact = {
            "schema_version": 1,
            "model_id": "test-model",
            "trained_at": "2026-01-01T00:00:00Z",
            "future_horizon_hours": 24,
            "features": list(names),
            "scaler": {"mean": [0] * len(names), "scale": [1] * len(names)},
            "heads": {
                "current": {"coefficients": [0] * len(names), "intercept": 0, "threshold": 0.5},
                "future": {"coefficients": [0] * len(names), "intercept": -1, "threshold": 0.5},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            path.write_text(json.dumps(artifact), encoding="utf-8")
            model = TinyModel.load(path)
        result = model.predict({name: 0 for name in names}, [])
        self.assertEqual("trained_logistic_model", result["prediction"]["method"])
        self.assertEqual("model_probability", result["prediction"]["future_risk"]["score_type"])
        self.assertEqual("higher_risk", result["prediction"]["current_risk"]["classification"])
        self.assertEqual("lower_risk", result["prediction"]["future_risk"]["classification"])
        self.assertAlmostEqual(0.268941, result["prediction"]["future_risk"]["probability"])
