from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest

from home_health_monitor.gateway.training import (
    FEATURE_NAMES,
    TrainingDataError,
    grouped_split,
    load_windows,
    select_thresholds,
    summarize_scores,
)


def row(subject_id: str, *, decision: str = "normal") -> dict:
    return {
        "schema_version": 1,
        "subject_id": subject_id,
        "feature_manifest_id": "features-v1",
        "decision": decision,
        "feature_names": list(FEATURE_NAMES),
        "values": [[0.0 for _ in FEATURE_NAMES] for _ in range(288)],
        "masks": [[1.0 for _ in FEATURE_NAMES] for _ in range(288)],
    }


class GatewayTrainingTests(unittest.TestCase):
    def write_rows(self, rows: list[dict]) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "windows.jsonl"
        path.write_text("".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8")
        return path

    def test_loader_accepts_normal_windows(self) -> None:
        rows = load_windows(self.write_rows([row("person-1"), row("person-2")]))

        self.assertEqual(2, len(rows))
        self.assertEqual(288, len(rows[0].values))

    def test_loader_rejects_non_normal_training_rows(self) -> None:
        path = self.write_rows([row("person-1", decision="anomaly")])

        with self.assertRaisesRegex(TrainingDataError, "normal-only"):
            load_windows(path)

    def test_loader_rejects_wrong_window_shape(self) -> None:
        item = row("person-1")
        item["values"] = item["values"][:-1]

        with self.assertRaisesRegex(TrainingDataError, "288"):
            load_windows(self.write_rows([item]))

    def test_loader_rejects_non_finite_values(self) -> None:
        item = row("person-1")
        item["values"][0][0] = math.inf

        with self.assertRaisesRegex(TrainingDataError, "finite"):
            load_windows(self.write_rows([item]))

    def test_loader_rejects_non_binary_masks(self) -> None:
        item = row("person-1")
        item["masks"][0][0] = 0.5

        with self.assertRaisesRegex(TrainingDataError, "masks"):
            load_windows(self.write_rows([item]))

    def test_split_keeps_subjects_in_one_partition(self) -> None:
        rows = load_windows(
            self.write_rows([row(f"person-{index}") for index in range(12)])
        )

        split = grouped_split(rows, seed=42)

        self.assertFalse(set(split.train_subjects) & set(split.test_subjects))
        self.assertFalse(set(split.train_subjects) & set(split.validation_subjects))
        self.assertFalse(set(split.validation_subjects) & set(split.test_subjects))
        self.assertEqual(len(rows), len(split.train) + len(split.validation) + len(split.test))

    def test_split_is_deterministic(self) -> None:
        rows = load_windows(
            self.write_rows([row(f"person-{index}") for index in range(12)])
        )

        first = grouped_split(rows, seed=7)
        second = grouped_split(rows, seed=7)

        self.assertEqual(first.train_subjects, second.train_subjects)
        self.assertEqual(first.validation_subjects, second.validation_subjects)
        self.assertEqual(first.test_subjects, second.test_subjects)

    def test_split_requires_three_subjects(self) -> None:
        rows = load_windows(self.write_rows([row("person-1"), row("person-2")]))

        with self.assertRaisesRegex(TrainingDataError, "three distinct subjects"):
            grouped_split(rows)

    def test_zero_validation_errors_still_produce_valid_thresholds(self) -> None:
        persistent, severe = select_thresholds([0.0, 0.0, 0.0])

        self.assertGreater(persistent, 0.0)
        self.assertGreater(severe, persistent)

    def test_score_summary_reports_rates_against_threshold(self) -> None:
        summary = summarize_scores([0.1, 0.2, 0.5], threshold=0.2)

        self.assertAlmostEqual(2 / 3, summary["fraction_at_or_above_threshold"])
        self.assertEqual(0.1, summary["minimum"])
        self.assertEqual(0.5, summary["maximum"])


if __name__ == "__main__":
    unittest.main()
