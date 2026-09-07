from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

from home_health_monitor.datasets.audit import audit_galaxy, audit_synthetic
from home_health_monitor.datasets.galaxyppg import convert_galaxy
from home_health_monitor.datasets.synthetic import convert_synthetic


class DatasetToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_csv(self, path: Path, fields: list[str], rows: list[list[object]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(fields)
            writer.writerows(rows)

    def test_audit_detects_repeated_synthetic_cycle(self) -> None:
        path = self.root / "healthmonitoringandfalldetection.csv"
        unique = [[index, 97 + index / 100] for index in range(36)]
        self.write_csv(path, ["id", "spo2"], unique * 17)

        report = audit_synthetic(path)

        self.assertEqual(612, report.total_rows)
        self.assertEqual(36, report.unique_rows)
        self.assertEqual(36, report.repeated_cycle_size)
        self.assertFalse(report.training_eligible)

    def test_galaxy_converter_never_labels_illness(self) -> None:
        watch = self.root / "P01" / "GalaxyWatch"
        self.write_csv(
            watch / "HR.csv",
            ["timestamp", "heartRate", "status"],
            [[1000, 72, 0]],
        )
        self.write_csv(
            watch / "ACC.csv",
            ["timestamp", "x", "y", "z"],
            [[1000, 0, 0, 9.81]],
        )

        records = list(convert_galaxy(self.root))

        self.assertTrue(records)
        self.assertNotIn("decision", records[0])
        self.assertTrue(all(item["dataset_role"] == "engineering_only" for item in records))
        self.assertEqual({"heart_rate", "accelerometer"}, {item["sensor"] for item in records})

    def test_galaxy_accelerometer_derives_motion_intensity(self) -> None:
        watch = self.root / "P02" / "GalaxyWatch"
        self.write_csv(
            watch / "ACC.csv",
            ["timestamp", "x", "y", "z"],
            [[1000, 3.0, 4.0, 0.0]],
        )

        record = next(convert_galaxy(self.root))

        self.assertAlmostEqual(4.81, record["motion_intensity"], places=2)

    def test_galaxy_audit_reports_missing_spo2(self) -> None:
        watch = self.root / "P01" / "GalaxyWatch"
        self.write_csv(watch / "HR.csv", ["timestamp", "heartRate"], [[1000, 72]])

        report = audit_galaxy(self.root)

        self.assertEqual(1, report.participant_count)
        self.assertIn("spo2", report.missing_required_streams)
        self.assertFalse(report.production_training_eligible)

    def test_synthetic_converter_requires_corrected_version(self) -> None:
        path = self.root / "healthmonitoringandfalldetection.csv"
        self.write_csv(path, ["temperature", "spo2"], [[36.8, 97]])

        with self.assertRaisesRegex(ValueError, "corrected Version 2"):
            list(convert_synthetic(path))

    def test_corrected_synthetic_rows_are_explicitly_demo_only(self) -> None:
        path = self.root / "healthmonitoringandfalldetection_corrected_v2.csv"
        self.write_csv(path, ["temperature", "spo2"], [[36.8, 97]])

        item = next(convert_synthetic(path))

        self.assertEqual("synthetic_demo", item["dataset_role"])
        self.assertFalse(item["production_training_eligible"])


if __name__ == "__main__":
    unittest.main()
