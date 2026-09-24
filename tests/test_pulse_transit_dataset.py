from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from home_health_monitor.datasets.pulse_transit import (
    PulseTransitArchive,
    PulseTransitDataError,
)
from home_health_monitor.datasets.audit import audit_pulse_transit


HEADER = """s1_sit 3 500 4
s1_sit.dat 212 1(0)/mV 12 0 0 0 0 ecg
s1_sit.dat 212 1(0)/NU 12 0 0 0 0 pleth_1
s1_sit.dat 212 1(0)/C 12 0 0 0 0 temp_1
# <filename>: s1_sit <activity>: sit <gender>: female <age>: 25 <spo2_start>: 98 <spo2_end>: 97
"""


class PulseTransitArchiveTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
        temporary.close()
        self.path = Path(temporary.name)
        fields = ["time", "ecg", "peaks", "pleth_1", "temp_1"]
        rows = [
            ["2021-01-01 00:00:00.000", 10, 0, 100, 33.1],
            ["2021-01-01 00:00:00.002", 11, 1, 101, 33.1],
            ["2021-01-01 00:00:00.004", 12, 0, 102, 33.2],
            ["2021-01-01 00:00:00.006", 13, 0, 103, 33.2],
        ]
        csv_buffer = io.StringIO()
        writer = csv.writer(csv_buffer, lineterminator="\n")
        writer.writerow(fields)
        writer.writerows(rows)
        with zipfile.ZipFile(self.path, "w") as archive:
            root = "pulse-transit-time-ppg/1.1.0"
            archive.writestr(f"{root}/README.txt", "Pulse Transit Time PPG Dataset")
            archive.writestr(f"{root}/s1_sit.hea", HEADER)
            archive.writestr(f"{root}/csv/s1_sit.csv", csv_buffer.getvalue())
            archive.writestr(
                f"{root}/csv/subjects_info.csv",
                "record,activity,gender,age,spo2_start,spo2_end\n"
                "s1_sit,sit,female,25,98,97\n",
            )

    def tearDown(self) -> None:
        self.path.unlink(missing_ok=True)

    def test_reads_record_shape_channels_and_metadata(self) -> None:
        with PulseTransitArchive(self.path) as dataset:
            records = dataset.records()

        self.assertEqual(1, len(records))
        record = records[0]
        self.assertEqual("s1_sit", record.name)
        self.assertEqual("s1", record.subject_id)
        self.assertEqual("sit", record.activity)
        self.assertEqual(500.0, record.sample_rate_hz)
        self.assertEqual(4, record.sample_count)
        self.assertEqual(("ecg", "pleth_1", "temp_1"), record.channels)
        self.assertEqual("female", record.metadata["gender"])
        self.assertEqual("98", record.metadata["spo2_start"])

    def test_streams_csv_rows_without_extracting_archive(self) -> None:
        with PulseTransitArchive(self.path) as dataset:
            rows = list(dataset.iter_csv_rows("s1_sit"))

        self.assertEqual(4, len(rows))
        self.assertEqual(2, rows[0].line_number)
        self.assertEqual("1", rows[1].values["peaks"])
        self.assertEqual("33.2", rows[-1].values["temp_1"])

    def test_rejects_an_archive_without_subject_metadata(self) -> None:
        bad_path = self.path.with_name("missing-subjects.zip")
        self.addCleanup(bad_path.unlink, missing_ok=True)
        with zipfile.ZipFile(bad_path, "w") as archive:
            archive.writestr("dataset/README.txt", "dataset")

        with self.assertRaisesRegex(PulseTransitDataError, "subjects_info"):
            PulseTransitArchive(bad_path)

    def test_audit_reports_real_dataset_role_and_missing_continuous_spo2(self) -> None:
        report = audit_pulse_transit(self.path)

        self.assertEqual(1, report.participant_count)
        self.assertEqual(1, report.recording_count)
        self.assertEqual(4, report.csv_valid_rows)
        self.assertEqual(0, report.csv_malformed_rows)
        self.assertEqual(0, report.recordings_with_csv_length_mismatch)
        self.assertTrue(report.feature_availability["heart_rate_bpm"])
        self.assertFalse(report.feature_availability["spo2_percent"])
        self.assertFalse(report.production_training_eligible)
        self.assertEqual("real_data_development_candidate", report.dataset_role)

    def test_csv_scan_records_malformed_rows_without_hiding_them(self) -> None:
        malformed = self.path.with_name("malformed.zip")
        self.addCleanup(malformed.unlink, missing_ok=True)
        with zipfile.ZipFile(self.path) as source, zipfile.ZipFile(malformed, "w") as target:
            for item in source.infolist():
                content = source.read(item.filename)
                if item.filename.endswith("/csv/s1_sit.csv"):
                    content += b"2021-01-01 00:00:01,10,0\n"
                target.writestr(item, content)

        report = audit_pulse_transit(malformed)

        self.assertEqual(4, report.csv_valid_rows)
        self.assertEqual(1, report.csv_malformed_rows)
        self.assertEqual(1, report.recordings_with_csv_length_mismatch)
        self.assertEqual("s1_sit", report.malformed_rows[0]["record"])
        self.assertEqual(6, report.malformed_rows[0]["line_number"])
        self.assertEqual(1, report.csv_length_mismatches[0]["difference"])

    def test_csv_stream_can_explicitly_skip_a_malformed_row(self) -> None:
        malformed = self.path.with_name("stream-malformed.zip")
        self.addCleanup(malformed.unlink, missing_ok=True)
        with zipfile.ZipFile(self.path) as source, zipfile.ZipFile(malformed, "w") as target:
            for item in source.infolist():
                content = source.read(item.filename)
                if item.filename.endswith("/csv/s1_sit.csv"):
                    content += b"2021-01-01 00:00:01,10,0\n"
                target.writestr(item, content)

        with PulseTransitArchive(malformed) as dataset:
            rows = list(dataset.iter_csv_rows("s1_sit", malformed="skip"))

        self.assertEqual(4, len(rows))

    def test_csv_stream_rejects_unknown_malformed_policy(self) -> None:
        with PulseTransitArchive(self.path) as dataset:
            with self.assertRaisesRegex(ValueError, "malformed policy"):
                list(dataset.iter_csv_rows("s1_sit", malformed="ignore"))

    def test_committed_real_archive_audit_is_portable_and_records_defects(self) -> None:
        root = Path(__file__).resolve().parents[1]
        report = json.loads(
            (root / "models" / "real-ppg-v2" / "data-audit.json").read_text()
        )

        self.assertEqual("pulse-transit-time-ppg.zip", report["archive_name"])
        self.assertNotIn("/Users/", json.dumps(report))
        self.assertEqual(22, report["participant_count"])
        self.assertEqual(66, report["recording_count"])
        self.assertEqual(1, report["csv_malformed_rows"])
        self.assertEqual(1, report["recordings_with_csv_length_mismatch"])
        self.assertFalse(report["feature_availability"]["spo2_percent"])


if __name__ == "__main__":
    unittest.main()
