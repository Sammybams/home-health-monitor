from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from home_health_monitor.datasets.pulse_transit_build import DATASET_ID, FEATURE_MANIFEST_ID
from home_health_monitor.datasets.pulse_transit_features import DATASET_FEATURE_NAMES
from home_health_monitor.gateway.vector_training import (
    RealPpgVector,
    fit_robust_normalizer,
    load_real_ppg_vectors,
    participant_plan,
    controlled_anomalies,
    aggregate_record_scores,
    vector_thresholds,
)
from home_health_monitor.gateway.vector_training import _reconstruction_scores


def vector(subject: str, *, valid: float = 1.0, heart_rate: float = 70.0) -> RealPpgVector:
    values = [float(index + 1) for index in range(len(DATASET_FEATURE_NAMES))]
    values[DATASET_FEATURE_NAMES.index("heart_rate_bpm")] = heart_rate
    values[DATASET_FEATURE_NAMES.index("heart_rate_valid")] = valid
    return RealPpgVector(subject, f"{subject}_sit", "sit", "female", 30, tuple(values))


class VectorTrainingTests(unittest.TestCase):
    def test_vector_score_is_mean_feature_squared_error(self) -> None:
        expected = np.zeros((1, 12), dtype=np.float32)
        reconstructed = np.zeros((1, 12), dtype=np.float32)
        reconstructed[0, :4] = [1.0, 2.0, 3.0, 4.0]

        scores = _reconstruction_scores(expected, reconstructed, np)

        self.assertAlmostEqual((1.0 + 4.0 + 9.0 + 16.0) / 12.0, float(scores[0]))

    def test_thresholds_are_selected_from_quantized_normal_scores(self) -> None:
        persistent, severe = vector_thresholds([0.1, 0.2, 0.3, 0.4], np)

        self.assertGreaterEqual(persistent, 0.2)
        self.assertGreaterEqual(severe, persistent * 2.0)

    def test_controlled_anomalies_cover_each_expected_signal_family(self) -> None:
        values = np.zeros((2, len(DATASET_FEATURE_NAMES)), dtype=np.float32)

        changed, labels = controlled_anomalies(values, np)

        self.assertEqual((8, len(DATASET_FEATURE_NAMES)), changed.shape)
        self.assertEqual(
            {"heart_rate_shift", "temperature_shift", "motion_shift", "combined_drift"},
            set(labels),
        )
        self.assertTrue(np.all(np.any(changed != 0.0, axis=1)))

    def test_record_aggregation_uses_upper_tail_of_five_second_scores(self) -> None:
        rows = [
            RealPpgVector("s1", "s1_sit", "sit", "female", 30, (1.0,) * 12),
            RealPpgVector("s1", "s1_sit", "sit", "female", 30, (1.0,) * 12),
            RealPpgVector("s1", "s1_walk", "walk", "female", 30, (1.0,) * 12),
        ]

        intervals = aggregate_record_scores(rows, [0.1, 0.9, 0.3], np)

        self.assertEqual(2, len(intervals))
        self.assertEqual("s1_sit", intervals[0]["record"])
        self.assertGreater(intervals[0]["p95_vector_error"], 0.8)

    def test_participant_plan_is_deterministic_and_has_no_leakage(self) -> None:
        rows = [vector(f"s{index}") for index in range(1, 23)]

        first = participant_plan(rows, seed=42)
        second = participant_plan(rows, seed=42)

        self.assertEqual(first, second)
        self.assertEqual(5, len(first.validation_folds))
        self.assertTrue(set(first.development_subjects).isdisjoint(first.locked_test_subjects))
        validation_subjects = [subject for fold in first.validation_folds for subject in fold]
        self.assertCountEqual(first.development_subjects, validation_subjects)
        self.assertEqual(len(validation_subjects), len(set(validation_subjects)))

    def test_normalizer_ignores_missing_heart_rate_then_imputes_normalized_zero(self) -> None:
        rows = [
            vector("s1", heart_rate=60.0),
            vector("s2", heart_rate=80.0),
            vector("s3", valid=0.0, heart_rate=0.0),
        ]

        normalizer = fit_robust_normalizer(rows, np)
        transformed = normalizer.transform(rows, np)

        heart_rate = DATASET_FEATURE_NAMES.index("heart_rate_bpm")
        self.assertAlmostEqual(70.0, normalizer.center[heart_rate])
        self.assertEqual(0.0, transformed[2, heart_rate])
        self.assertTrue(np.all(np.isfinite(transformed)))

    def test_loader_validates_manifest_and_demographics(self) -> None:
        item = {
            "schema_version": 1,
            "dataset_id": DATASET_ID,
            "feature_manifest_id": FEATURE_MANIFEST_ID,
            "feature_names": list(DATASET_FEATURE_NAMES),
            "subject_id": "s1",
            "record": "s1_sit",
            "activity": "sit",
            "demographics": {"gender": "female", "age": 25},
            "values": [1.0] * len(DATASET_FEATURE_NAMES),
        }
        item["values"][DATASET_FEATURE_NAMES.index("heart_rate_valid")] = 1.0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vectors.jsonl"
            path.write_text(json.dumps(item) + "\n", encoding="utf-8")

            rows = load_real_ppg_vectors(path)

        self.assertEqual(1, len(rows))
        self.assertEqual("s1", rows[0].subject_id)
        self.assertEqual(25, rows[0].age)


if __name__ == "__main__":
    unittest.main()
