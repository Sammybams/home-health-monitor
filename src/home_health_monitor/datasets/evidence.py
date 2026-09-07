from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path
import zipfile

from .audit import audit_galaxy, audit_synthetic
from .development import load_supplied_monitoring_rows
from .galaxyppg import convert_galaxy


class _Range:
    def __init__(self) -> None:
        self.count = 0
        self.minimum = math.inf
        self.maximum = -math.inf
        self.total = 0.0

    def add(self, value: object) -> None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return
        if not math.isfinite(number):
            return
        self.count += 1
        self.minimum = min(self.minimum, number)
        self.maximum = max(self.maximum, number)
        self.total += number

    def report(self) -> dict[str, float | int | None]:
        return {
            "count": self.count,
            "minimum": self.minimum if self.count else None,
            "mean": self.total / self.count if self.count else None,
            "maximum": self.maximum if self.count else None,
        }


def profile_supplied(path: str | Path) -> dict[str, object]:
    audit = audit_synthetic(path)
    rows = load_supplied_monitoring_rows(path)
    return {
        "dataset": "supplied_monitoring_fall_csv",
        "role": "development_demo_seed_and_visual_check",
        "rows": audit.total_rows,
        "unique_rows": audit.unique_rows,
        "duplicate_rows": audit.duplicate_rows,
        "normal_unique_rows": sum(row.is_normal for row in rows),
        "feature_availability": {
            "heart_rate_bpm": True,
            "spo2_percent": True,
            "temperature_c": True,
            "motion_intensity": True,
        },
    }


def profile_galaxy(root: str | Path) -> dict[str, object]:
    audit = audit_galaxy(root)
    ranges = {name: _Range() for name in ("heart_rate", "motion", "skin_temperature")}
    for record in convert_galaxy(root):
        sensor = record["sensor"]
        if sensor == "heart_rate":
            value = record["heart_rate_bpm"]
            if 20 <= float(value) <= 250:
                ranges["heart_rate"].add(value)
        elif sensor == "accelerometer":
            ranges["motion"].add(record["motion_intensity"])
        elif sensor == "skin_temperature" and "skin_temperature_c" in record:
            ranges["skin_temperature"].add(record["skin_temperature_c"])
    return {
        "dataset": "GalaxyPPG",
        "role": "engineering_reference",
        "participants": audit.participant_count,
        "heart_rate": ranges["heart_rate"].report(),
        "motion": ranges["motion"].report(),
        "skin_temperature": ranges["skin_temperature"].report(),
        "feature_availability": {
            "heart_rate_bpm": ranges["heart_rate"].count > 0,
            "spo2_percent": False,
            "temperature_c": ranges["skin_temperature"].count > 0,
            "motion_intensity": ranges["motion"].count > 0,
        },
    }


def profile_bidmc(path: str | Path) -> dict[str, object]:
    heart_rate = _Range()
    spo2 = _Range()
    participants = set()
    with zipfile.ZipFile(path) as archive:
        names = sorted(name for name in archive.namelist() if name.endswith("_Numerics.csv"))
        for name in names:
            participants.add(Path(name).stem.split("_")[1])
            content = io.TextIOWrapper(archive.open(name), encoding="utf-8-sig", newline="")
            for row in csv.DictReader(content):
                heart_rate.add(row.get(" HR"))
                spo2.add(row.get(" SpO2"))
    return {
        "dataset": "BIDMC_PPG_and_Respiration",
        "role": "engineering_reference",
        "participants": len(participants),
        "heart_rate": heart_rate.report(),
        "spo2": spo2.report(),
        "feature_availability": {
            "heart_rate_bpm": heart_rate.count > 0,
            "spo2_percent": spo2.count > 0,
            "temperature_c": False,
            "motion_intensity": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile development data evidence")
    parser.add_argument("--supplied-csv", type=Path, required=True)
    parser.add_argument("--galaxy-root", type=Path, required=True)
    parser.add_argument("--bidmc-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {
        "schema_version": 1,
        "production_training_eligible": False,
        "sources": [
            profile_supplied(args.supplied_csv),
            profile_galaxy(args.galaxy_root),
            profile_bidmc(args.bidmc_zip),
        ],
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
