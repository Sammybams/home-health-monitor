from __future__ import annotations

import csv
from dataclasses import dataclass
import io
from pathlib import Path, PurePosixPath
import re
from typing import Iterator
import zipfile


class PulseTransitDataError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class PulseTransitRecord:
    name: str
    subject_id: str
    activity: str
    sample_rate_hz: float
    sample_count: int
    channels: tuple[str, ...]
    metadata: dict[str, str]

    @property
    def duration_seconds(self) -> float:
        return self.sample_count / self.sample_rate_hz


@dataclass(frozen=True, slots=True)
class PulseTransitCsvRow:
    line_number: int
    values: dict[str, str]


@dataclass(frozen=True, slots=True)
class PulseTransitCsvScan:
    record: str
    valid_rows: int
    malformed_rows: tuple[dict[str, int | str], ...]


def _metadata_comment(line: str) -> dict[str, str]:
    return {
        match.group(1): match.group(2).strip()
        for match in re.finditer(
            r"<([^>]+)>:\s*(.*?)(?=\s+<[^>]+>:\s*|$)", line.removeprefix("#").strip()
        )
    }


def _record_sort_key(name: str) -> tuple[int, str]:
    match = re.fullmatch(r"s(\d+)_(sit|walk|run)", name)
    if match is None:
        return (10**9, name)
    return (int(match.group(1)), match.group(2))


class PulseTransitArchive:
    """Read PhysioNet Pulse Transit Time PPG data directly from its ZIP."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        try:
            self._archive = zipfile.ZipFile(self.path)
        except (OSError, zipfile.BadZipFile) as exc:
            raise PulseTransitDataError(f"could not open Pulse Transit archive: {exc}") from exc
        self._names = frozenset(self._archive.namelist())
        subject_files = sorted(
            name for name in self._names if name.endswith("/csv/subjects_info.csv")
        )
        if len(subject_files) != 1:
            self._archive.close()
            raise PulseTransitDataError("archive must contain exactly one csv/subjects_info.csv")
        self._subjects_member = subject_files[0]
        self._root = PurePosixPath(self._subjects_member).parents[1]
        self._subject_rows = self._load_subject_rows()

    def __enter__(self) -> "PulseTransitArchive":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._archive.close()

    def _load_subject_rows(self) -> dict[str, dict[str, str]]:
        with self._archive.open(self._subjects_member) as raw:
            reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
            if reader.fieldnames is None or "record" not in reader.fieldnames:
                raise PulseTransitDataError("subjects_info.csv must contain a record column")
            rows: dict[str, dict[str, str]] = {}
            for row in reader:
                name = str(row.get("record", "")).strip()
                if not name or None in row:
                    raise PulseTransitDataError("subjects_info.csv contains a malformed row")
                if name in rows:
                    raise PulseTransitDataError(f"subjects_info.csv repeats record {name}")
                rows[name] = {key: str(value).strip() for key, value in row.items()}
        return rows

    def _member(self, relative: str) -> str:
        member = str(self._root / relative)
        if member not in self._names:
            raise PulseTransitDataError(f"archive is missing {relative}")
        return member

    def records(self) -> tuple[PulseTransitRecord, ...]:
        result = []
        for name in sorted(self._subject_rows, key=_record_sort_key):
            member = self._member(f"{name}.hea")
            try:
                text = self._archive.read(member).decode("utf-8-sig")
            except (OSError, UnicodeDecodeError) as exc:
                raise PulseTransitDataError(f"could not read {name}.hea: {exc}") from exc
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            try:
                summary = lines[0].split()
                header_name = summary[0]
                signal_count = int(summary[1])
                sample_rate = float(summary[2])
                sample_count = int(summary[3])
            except (IndexError, ValueError) as exc:
                raise PulseTransitDataError(f"invalid WFDB header for {name}") from exc
            if header_name != name or signal_count <= 0 or sample_rate <= 0 or sample_count <= 0:
                raise PulseTransitDataError(f"invalid WFDB summary for {name}")
            signal_lines = [line for line in lines[1:] if not line.startswith("#")]
            if len(signal_lines) != signal_count:
                raise PulseTransitDataError(f"WFDB signal count does not match {name}.hea")
            channels = tuple(line.split()[-1] for line in signal_lines)
            comments = {}
            for line in lines[1:]:
                if line.startswith("#"):
                    comments.update(_metadata_comment(line))
            metadata = {**comments, **self._subject_rows[name]}
            activity = metadata.get("activity", name.rsplit("_", 1)[-1])
            result.append(
                PulseTransitRecord(
                    name=name,
                    subject_id=name.split("_", 1)[0],
                    activity=activity,
                    sample_rate_hz=sample_rate,
                    sample_count=sample_count,
                    channels=channels,
                    metadata=metadata,
                )
            )
        return tuple(result)

    def iter_csv_rows(self, record_name: str) -> Iterator[PulseTransitCsvRow]:
        if record_name not in self._subject_rows:
            raise PulseTransitDataError(f"unknown record {record_name}")
        member = self._member(f"csv/{record_name}.csv")
        with self._archive.open(member) as raw:
            reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
            header = next(reader, None)
            if not header or len(header) != len(set(header)):
                raise PulseTransitDataError(f"{record_name}.csv has an invalid header")
            for line_number, row in enumerate(reader, 2):
                if len(row) != len(header):
                    raise PulseTransitDataError(
                        f"{record_name}.csv row {line_number} has {len(row)} columns; "
                        f"expected {len(header)}"
                    )
                yield PulseTransitCsvRow(line_number, dict(zip(header, row)))

    def scan_csv(self, record_name: str) -> PulseTransitCsvScan:
        """Count structurally valid rows while retaining every malformed location."""
        if record_name not in self._subject_rows:
            raise PulseTransitDataError(f"unknown record {record_name}")
        member = self._member(f"csv/{record_name}.csv")
        malformed = []
        valid = 0
        with self._archive.open(member) as raw:
            reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
            header = next(reader, None)
            if not header or len(header) != len(set(header)):
                raise PulseTransitDataError(f"{record_name}.csv has an invalid header")
            for line_number, row in enumerate(reader, 2):
                if len(row) == len(header):
                    valid += 1
                else:
                    malformed.append(
                        {
                            "record": record_name,
                            "line_number": line_number,
                            "expected_columns": len(header),
                            "actual_columns": len(row),
                        }
                    )
        return PulseTransitCsvScan(record_name, valid, tuple(malformed))
