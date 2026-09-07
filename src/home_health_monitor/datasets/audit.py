from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit external home-monitor datasets")
    parser.add_argument("kind", choices=("synthetic", "galaxyppg"))
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    report = audit_synthetic(args.path) if args.kind == "synthetic" else audit_galaxy(args.path)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
