from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from home_health_monitor.datasets.pulse_transit_features import DATASET_FEATURE_NAMES
from home_health_monitor.gateway.vector_autoencoder import IntervalPrediction, VectorReconstruction
from home_health_monitor.gateway.vector_contracts import parse_vector_interval
from home_health_monitor.gateway.vector_engine import VectorGatewayEngine
from home_health_monitor.gateway.store import GatewayStore
from tests.test_vector_contracts import valid_interval


class FakeVectorModel:
    model_id = "fake-v2"

    def predict(self, vector):
        score = vector["heart_rate_bpm"]
        return VectorReconstruction(score, {name: score for name in DATASET_FEATURE_NAMES})

    def aggregate(self, results, *, previous_interval_above_threshold=False, personal_threshold=None):
        score = max(item.overall_error for item in results)
        threshold = personal_threshold or 1.0
        above = score >= threshold
        severe = score >= max(3.0, threshold * 2)
        persistent = above and previous_interval_above_threshold
        return IntervalPrediction("anomaly" if severe or persistent else "normal", score,
            threshold, max(3.0, threshold * 2), above, severe, persistent,
            len(results), 16, min(1, len(results) / 16), ("heart_rate_bpm",))


class VectorEngineTests(unittest.TestCase):
    def test_persists_history_and_returns_idempotent_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with GatewayStore(Path(directory) / "gateway.db") as store:
                engine = VectorGatewayEngine(store, FakeVectorModel())
                payload = valid_interval()
                for vector in payload["vectors"]:
                    vector["heart_rate_bpm"] = 1.5
                first = engine.ingest(parse_vector_interval(payload))
                duplicate = engine.ingest(parse_vector_interval(payload))
                payload["sequence"] = 2
                payload["interval_start"] = "2026-09-24T08:08:00+01:00"
                payload["interval_end"] = "2026-09-24T08:16:00+01:00"
                second = engine.ingest(parse_vector_interval(payload))

                self.assertEqual("normal", first["decision"])
                self.assertEqual(first, duplicate)
                self.assertEqual("anomaly", second["decision"])
                self.assertTrue(second["persistent"])
                self.assertEqual(second, store.latest_vector_result("subject-1"))


if __name__ == "__main__":
    unittest.main()
