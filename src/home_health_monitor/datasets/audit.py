from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import statistics

from .pulse_transit import PulseTransitArchive


@dataclass(frozen=True, slots=True)
class SyntheticAudit:
    path: str
    total_rows: int
    unique_rows: int
    duplicate_rows: int
    repeated_cycle_size: int | None
    training_eligible: bool
    reason_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GalaxyAudit:
    path: str
    participant_count: int
    csv_file_count: int
    streams: tuple[str, ...]
    missing_required_streams: tuple[str, ...]
    production_training_eligible: bool
    dataset_role: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PulseTransitAudit:
    path: str
    archive_sha256: str
    participant_count: int
    recording_count: int
    activity_counts: dict[str, int]
    total_duration_hours: float
    total_waveform_samples: int
    channel_names: tuple[str, ...]
    feature_availability: dict[str, bool]
    endpoint_spo2_count: int
    endpoint_spo2_range: tuple[float, float]
    gender_counts: dict[str, int]
    age_range: tuple[int, int]
    mean_age: float
    csv_valid_rows: int
    csv_malformed_rows: int
    malformed_rows: tuple[dict[str, int | str], ...]
    recordings_with_csv_length_mismatch: int
    csv_length_mismatches: tuple[dict[str, int | str], ...]
    production_training_eligible: bool
    dataset_role: str
    reason_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _cycle_size(rows: list[tuple[str, ...]]) -> int | None:
    for size in range(1, len(rows)):
        if len(rows) % size == 0 and all(row == rows[index % size] for index, row in enumerate(rows)):
            return size
    return None


def audit_synthetic(path: str | Path) -> SyntheticAudit:
    source = Path(path)
    with source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        rows = [tuple(value.strip() for value in row) for row in reader if row]
    unique = len(set(rows))
    cycle = _cycle_size(rows)
    reasons = ["synthetic_source"]
    if unique < len(rows):
        reasons.append("duplicate_rows")
    if cycle is not None:
        reasons.append("repeated_cycle")
    return SyntheticAudit(
        path=str(source),
        total_rows=len(rows),
        unique_rows=unique,
        duplicate_rows=len(rows) - unique,
        repeated_cycle_size=cycle,
        training_eligible=False,
        reason_codes=tuple(reasons),
    )


def _galaxy_sensor(path: Path) -> str | None:
    name = "".join(character for character in path.stem.lower() if character.isalnum())
    if name.startswith("hr") or "heartrate" in name:
        return "heart_rate"
    if name.startswith("acc") or "accelerometer" in name:
        return "motion"
    if "temp" in name:
        return "temperature"
    if "ppg" in name:
        return "ppg"
    return None


def audit_galaxy(root: str | Path) -> GalaxyAudit:
    source = Path(root)
    files = [path for path in source.rglob("*.csv") if "galaxywatch" in path.as_posix().lower()]
    participants = {
        part
        for path in files
        for part in path.parts
        if len(part) == 3 and part[0].upper() == "P" and part[1:].isdigit()
    }
    streams = {_galaxy_sensor(path) for path in files} - {None}
    required = {"heart_rate", "motion", "temperature", "spo2"}
    return GalaxyAudit(
        path=str(source),
        participant_count=len(participants),
        csv_file_count=len(files),
        streams=tuple(sorted(streams)),
        missing_required_streams=tuple(sorted(required - streams)),
        production_training_eligible=False,
        dataset_role="engineering_only",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_pulse_transit(path: str | Path) -> PulseTransitAudit:
    source = Path(path)
    with PulseTransitArchive(source) as dataset:
        records = dataset.records()
        scans = tuple(dataset.scan_csv(record.name) for record in records)

    people = {}
    for record in records:
        people.setdefault(record.subject_id, record.metadata)
    ages = [int(person["age"]) for person in people.values()]
    genders = sorted({person["gender"] for person in people.values()})
    endpoint_spo2 = [
        float(record.metadata[name])
        for record in records
        for name in ("spo2_start", "spo2_end")
    ]
    activity_counts = {
        activity: sum(record.activity == activity for record in records)
        for activity in sorted({record.activity for record in records})
    }
    malformed = tuple(item for scan in scans for item in scan.malformed_rows)
    scans_by_record = {scan.record: scan for scan in scans}
    length_mismatches = []
    for record in records:
        scan = scans_by_record[record.name]
        actual_rows = scan.valid_rows + len(scan.malformed_rows)
        if actual_rows != record.sample_count:
            length_mismatches.append(
                {
                    "record": record.name,
                    "wfdb_sample_count": record.sample_count,
                    "csv_row_count": actual_rows,
                    "difference": actual_rows - record.sample_count,
                }
            )
    channels = tuple(sorted({channel for record in records for channel in record.channels}))
    return PulseTransitAudit(
        path=str(source),
        archive_sha256=_sha256(source),
        participant_count=len(people),
        recording_count=len(records),
        activity_counts=activity_counts,
        total_duration_hours=round(
            sum(record.duration_seconds for record in records) / 3600.0, 6
        ),
        total_waveform_samples=sum(record.sample_count for record in records),
        channel_names=channels,
        feature_availability={
            "heart_rate_bpm": "ecg" in channels and "pleth_1" in channels,
            "spo2_percent": False,
            "temperature_c": any(name.startswith("temp_") for name in channels),
            "motion_intensity": all(name in channels for name in ("a_x", "a_y", "a_z")),
        },
        endpoint_spo2_count=len(endpoint_spo2),
        endpoint_spo2_range=(min(endpoint_spo2), max(endpoint_spo2)),
        gender_counts={
            gender: sum(person["gender"] == gender for person in people.values())
            for gender in genders
        },
        age_range=(min(ages), max(ages)),
        mean_age=round(statistics.mean(ages), 6),
        csv_valid_rows=sum(scan.valid_rows for scan in scans),
        csv_malformed_rows=len(malformed),
        malformed_rows=malformed,
        recordings_with_csv_length_mismatch=len(length_mismatches),
        csv_length_mismatches=tuple(length_mismatches),
        production_training_eligible=False,
        dataset_role="real_data_development_candidate",
        reason_codes=(
            "no_continuous_spo2",
            "short_activity_recordings",
            "not_target_wearable",
            "not_target_population",
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit external home-monitor datasets")
    parser.add_argument("kind", choices=("synthetic", "galaxyppg", "pulse-transit"))
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.kind == "synthetic":
        report = audit_synthetic(args.path)
    elif args.kind == "galaxyppg":
        report = audit_galaxy(args.path)
    else:
        report = audit_pulse_transit(args.path)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
