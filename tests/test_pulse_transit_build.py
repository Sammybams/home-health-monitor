from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from home_health_monitor.datasets.pulse_transit_build import (
    DATASET_ID,
    FEATURE_MANIFEST_ID,
    build_feature_dataset,
)


class PulseTransitBuildTests(unittest.TestCase):
    def test_committed_full_archive_report_is_portable_and_reproducible(self) -> None:
        root = Path(__file__).resolve().parents[1]
        report = json.loads(
            (root / "models" / "real-ppg-v2" / "feature-extraction-report.json").read_text()
        )

        self.assertEqual(6_430, report["feature_rows"])
        self.assertEqual(0, report["extraction_failure_count"])
        self.assertEqual(
            "fbd8defee1be6af4496b9191cbc915a5bdf81b4766148b830aae6f4cfa7677f9",
            report["source_archive_sha256"],
        )
        self.assertNotIn("/Users/", json.dumps(report))
        self.assertLess(report["ppg_vs_ecg_heart_rate"]["mae_bpm"], 2.0)

    def test_builds_atomic_provenanced_feature_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "source.zip"
            output_path = root / "features.jsonl"
            csv_lines = ["time,ecg,peaks,pleth_1,pleth_2,pleth_3,temp_1,a_x,a_y,a_z"]
            for index in range(10):
                csv_lines.append(
                    f"t{index},0,{int(index in (1, 3, 5, 7, 9))},{69900 + index},{70000 + index},{70100 + index},33.5,0,0,9.81"
                )
            with zipfile.ZipFile(archive_path, "w") as archive:
                dataset = "pulse-transit-time-ppg/1.1.0"
                archive.writestr(
                    f"{dataset}/s1_sit.hea",
                    "s1_sit 6 2 10\n"
                    "x 1 1 1 1 pleth_2\nx 1 1 1 1 temp_1\nx 1 1 1 1 a_x\n"
                    "x 1 1 1 1 a_y\nx 1 1 1 1 a_z\nx 1 1 1 1 ecg\n",
                )
                archive.writestr(
                    f"{dataset}/csv/subjects_info.csv",
                    "record,activity,gender,age\ns1_sit,sit,female,25\n",
                )
                archive.writestr(
                    f"{dataset}/csv/s1_sit.csv", "\n".join(csv_lines) + "\n"
                )

            report = build_feature_dataset(archive_path, output_path)
            rows = [json.loads(line) for line in output_path.read_text().splitlines()]

            self.assertEqual(1, len(rows))
            self.assertEqual(DATASET_ID, rows[0]["dataset_id"])
            self.assertEqual(FEATURE_MANIFEST_ID, rows[0]["feature_manifest_id"])
            self.assertEqual("s1", rows[0]["subject_id"])
            self.assertEqual(1, report["feature_rows"])
            self.assertEqual(0, report["extraction_failure_count"])
            self.assertEqual(64, len(report["source_archive_sha256"]))
            self.assertEqual(64, len(report["output_sha256"]))

    def test_rejects_invalid_segment_duration_without_leaving_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "features.jsonl"
            with self.assertRaisesRegex(ValueError, "segment_seconds"):
                build_feature_dataset("not-read.zip", output, segment_seconds=0)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
