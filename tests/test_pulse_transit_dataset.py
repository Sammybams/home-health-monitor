from __future__ import annotations

import csv
import io
from pathlib import Path
import tempfile
import unittest
import zipfile

from home_health_monitor.datasets.pulse_transit import (
    PulseTransitArchive,
    PulseTransitDataError,
)


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


if __name__ == "__main__":
    unittest.main()
