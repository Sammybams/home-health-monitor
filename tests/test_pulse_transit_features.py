from __future__ import annotations

import json
import math
from pathlib import Path
import unittest

import numpy as np

from home_health_monitor.datasets.pulse_transit_features import (
    DATASET_FEATURE_NAMES,
    ecg_reference_heart_rate,
    extract_segment_features,
)


class PulseTransitFeatureTests(unittest.TestCase):
    def test_extracts_ppg_heart_rate_and_sensor_summaries(self) -> None:
        sample_rate = 500.0
        seconds = 5.0
        time = np.arange(int(sample_rate * seconds)) / sample_rate
        heart_rate_bpm = 72.0
        ppg = 70_000.0 + 2_000.0 * np.sin(
            2.0 * math.pi * (heart_rate_bpm / 60.0) * time
        )
        peaks = np.zeros(time.size)
        peaks[(np.arange(0.4, seconds, 60.0 / heart_rate_bpm) * sample_rate).astype(int)] = 1
        rows = []
        for index in range(time.size):
            rows.append(
                {
                    "pleth_2": str(ppg[index]),
                    "peaks": str(peaks[index]),
                    "temp_1": "33.5",
                    "a_x": "0.0",
                    "a_y": "0.0",
                    "a_z": "9.81",
                }
            )

        result = extract_segment_features(rows, sample_rate_hz=sample_rate)

        self.assertEqual(DATASET_FEATURE_NAMES, tuple(result.values))
        self.assertAlmostEqual(heart_rate_bpm, result.values["heart_rate_bpm"], delta=1.0)
        self.assertAlmostEqual(heart_rate_bpm, result.ecg_reference_hr_bpm, delta=0.2)
        self.assertAlmostEqual(33.5, result.values["temperature_mean_c"], places=6)
        self.assertAlmostEqual(9.81, result.values["acceleration_magnitude_mean"], places=6)
        self.assertAlmostEqual(0.0, result.values["dynamic_acceleration_rms"], places=6)
        self.assertGreater(result.values["ppg_signal_quality"], 0.9)

    def test_rejects_non_finite_or_missing_required_values(self) -> None:
        rows = [
            {
                "pleth_2": "nan",
                "peaks": "0",
                "temp_1": "33.5",
                "a_x": "0",
                "a_y": "0",
                "a_z": "9.81",
            }
        ]

        with self.assertRaisesRegex(ValueError, "finite"):
            extract_segment_features(rows, sample_rate_hz=500.0)

    def test_ecg_reference_requires_two_peaks(self) -> None:
        self.assertIsNone(ecg_reference_heart_rate([0, 0, 1, 0], 500.0))

    def test_committed_manifest_matches_extractor_and_is_not_ble_contract(self) -> None:
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads(
            (root / "models" / "real-ppg-v2" / "dataset-feature-manifest.json").read_text()
        )

        self.assertEqual(list(DATASET_FEATURE_NAMES), manifest["feature_order"])
        self.assertEqual("dataset_feature_candidate", manifest["status"])
        self.assertFalse(manifest["is_production_ble_contract"])
        self.assertNotIn("ecg_reference_hr_bpm", manifest["feature_order"])


if __name__ == "__main__":
    unittest.main()
