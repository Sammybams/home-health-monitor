from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest
import zipfile

from home_health_monitor.datasets.audit import audit_galaxy, audit_synthetic
from home_health_monitor.datasets.galaxyppg import convert_galaxy
from home_health_monitor.datasets.development import (
    build_development_corpus,
    load_supplied_monitoring_rows,
    write_normal_windows,
)
from home_health_monitor.datasets.evidence import profile_bidmc, profile_galaxy
from home_health_monitor.reporting import (
    availability_matrix,
    binary_performance,
    scenario_score_summary,
)
from home_health_monitor.gateway.training import load_windows
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

    def test_galaxy_converter_reads_published_skin_temperature_headers(self) -> None:
        watch = self.root / "P03" / "GalaxyWatch"
        self.write_csv(
            watch / "SkinTemp.csv",
            ["dataReceived", "timestamp", "ambientTemp", "objectTemp", "status"],
            [[1001, 1000, 30.2, 32.4, 0]],
        )

        record = next(convert_galaxy(self.root))

        self.assertEqual(32.4, record["skin_temperature_c"])
        self.assertEqual(30.2, record["ambient_temperature_c"])

    def test_development_corpus_deduplicates_and_trains_on_normal_only(self) -> None:
        path = self.root / "healthmonitoringandfalldetection.csv"
        fields = [
            "heart_rate",
            "oxygen_level",
            "temperature",
            "acceleration_magnitude",
            "health_condition",
        ]
        normal = [72, 98, 36.7, 9.81, "Normal"]
        abnormal = [128, 84, 38.5, 7.2, "Hypoxia"]
        self.write_csv(path, fields, [normal, abnormal, normal, abnormal])

        rows = load_supplied_monitoring_rows(path)
        corpus = build_development_corpus(rows, subject_count=6, windows_per_subject=2, seed=9)

        self.assertEqual(2, len(rows))
        self.assertEqual(1, sum(row.is_normal for row in rows))
        self.assertEqual(12, len(corpus.normal_windows))
        self.assertEqual(6, len({row.subject_id for row in corpus.normal_windows}))
        self.assertEqual(12, len(corpus.simulated_anomaly_windows))
        self.assertTrue(all(row.decision == "normal" for row in corpus.normal_windows))
        self.assertTrue(
            all(row.decision == "engineering_simulation" for row in corpus.simulated_anomaly_windows)
        )
        self.assertEqual(288, len(corpus.normal_windows[0].values))

    def test_development_corpus_is_deterministic(self) -> None:
        path = self.root / "healthmonitoringandfalldetection.csv"
        self.write_csv(
            path,
            [
                "heart_rate",
                "oxygen_level",
                "temperature",
                "acceleration_magnitude",
                "health_condition",
            ],
            [[72, 98, 36.7, 9.81, "Normal"]],
        )
        rows = load_supplied_monitoring_rows(path)

        first = build_development_corpus(rows, subject_count=3, windows_per_subject=1, seed=3)
        second = build_development_corpus(rows, subject_count=3, windows_per_subject=1, seed=3)

        self.assertEqual(first, second)

    def test_development_writer_emits_loader_compatible_normal_data(self) -> None:
        path = self.root / "healthmonitoringandfalldetection.csv"
        self.write_csv(
            path,
            [
                "heart_rate",
                "oxygen_level",
                "temperature",
                "acceleration_magnitude",
                "health_condition",
            ],
            [[72, 98, 36.7, 9.81, "Normal"]],
        )
        corpus = build_development_corpus(
            load_supplied_monitoring_rows(path),
            subject_count=3,
            windows_per_subject=1,
        )
        output = self.root / "normal-windows.jsonl"

        write_normal_windows(corpus, output)

        self.assertEqual(3, len(load_windows(output)))

    def test_bidmc_profile_counts_real_hr_and_spo2_rows(self) -> None:
        archive = self.root / "bidmc.zip"
        content = "Time [s], HR, PULSE, RESP, SpO2\n0,72,72,14,98\n1,73,73,14,97\n"
        with zipfile.ZipFile(archive, "w") as handle:
            handle.writestr("bidmc_csv/bidmc_01_Numerics.csv", content)

        profile = profile_bidmc(archive)

        self.assertEqual(1, profile["participants"])
        self.assertEqual(2, profile["heart_rate"]["count"])
        self.assertEqual(97.0, profile["spo2"]["minimum"])

    def test_galaxy_profile_reports_target_stream_coverage(self) -> None:
        watch = self.root / "P03" / "GalaxyWatch"
        self.write_csv(watch / "HR.csv", ["timestamp", "hr"], [[1000, 72]])
        self.write_csv(watch / "ACC.csv", ["timestamp", "x", "y", "z"], [[1000, 0, 0, 9.81]])
        self.write_csv(
            watch / "SkinTemp.csv",
            ["timestamp", "ambientTemp", "objectTemp"],
            [[1000, 30, 32]],
        )

        profile = profile_galaxy(self.root)

        self.assertEqual(1, profile["participants"])
        self.assertEqual(1, profile["heart_rate"]["count"])
        self.assertFalse(profile["feature_availability"]["spo2_percent"])

    def test_availability_matrix_preserves_source_and_feature_order(self) -> None:
        evidence = {
            "sources": [
                {
                    "dataset": "first",
                    "feature_availability": {
                        "heart_rate_bpm": True,
                        "spo2_percent": False,
                        "temperature_c": True,
                        "motion_intensity": False,
                    },
                }
            ]
        }

        sources, features, matrix = availability_matrix(evidence)

        self.assertEqual(["first"], sources)
        self.assertEqual("Heart rate", features[0])
        self.assertEqual([[1, 0, 1, 0]], matrix)

    def test_binary_performance_uses_normal_as_negative_class(self) -> None:
        performance = binary_performance(
            normal_scores=[0.1, 0.3],
            anomaly_scores=[0.4, 0.8],
            threshold=0.35,
        )

        self.assertEqual({"tn": 2, "fp": 0, "fn": 0, "tp": 2}, performance["confusion"])
        self.assertEqual(1.0, performance["accuracy"])
        self.assertEqual(1.0, performance["precision"])
        self.assertEqual(1.0, performance["recall"])
        self.assertEqual(1.0, performance["specificity"])

    def test_scenario_summary_keeps_each_simulation_separate(self) -> None:
        summary = scenario_score_summary(
            ["low_spo2", "high_temperature", "low_spo2"],
            [0.4, 0.3, 0.5],
            threshold=0.35,
        )

        self.assertEqual(2, summary["low_spo2"]["count"])
        self.assertEqual(1.0, summary["low_spo2"]["detection_fraction"])
        self.assertEqual(0.0, summary["high_temperature"]["detection_fraction"])

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
