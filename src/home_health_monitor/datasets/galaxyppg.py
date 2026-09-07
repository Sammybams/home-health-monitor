from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterator


def _normalized(name: str) -> str:
    return "".join(character for character in name.lower() if character.isalnum())


def _column(row: dict[str, str], *aliases: str) -> str | None:
    normalized = {_normalized(name): value for name, value in row.items()}
    for alias in aliases:
        value = normalized.get(_normalized(alias))
        if value not in (None, ""):
            return value
    return None


def _number(row: dict[str, str], *aliases: str) -> float | None:
    value = _column(row, *aliases)
    if value is None:
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _participant(path: Path) -> str:
    for part in reversed(path.parts):
        if len(part) == 3 and part[0].upper() == "P" and part[1:].isdigit():
            return part.upper()
    return "unknown"


def _sensor(path: Path) -> str | None:
    name = _normalized(path.stem)
    if name.startswith("hr") or "heartrate" in name:
        return "heart_rate"
    if name.startswith("acc") or "accelerometer" in name:
        return "accelerometer"
    if "temp" in name:
        return "skin_temperature"
    if "ppg" in name:
        return "ppg"
    return None


def convert_galaxy(root: str | Path) -> Iterator[dict[str, object]]:
    source = Path(root)
    files = sorted(
        path
        for path in source.rglob("*.csv")
        if "galaxywatch" in path.as_posix().lower() and _sensor(path) is not None
    )
    for path in files:
        sensor = _sensor(path)
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row_number, row in enumerate(csv.DictReader(handle), 2):
                timestamp = _column(row, "timestamp", "time", "timestamp_ms")
                record: dict[str, object] = {
                    "dataset_role": "engineering_only",
                    "production_training_eligible": False,
                    "subject_id": _participant(path),
                    "sensor": sensor,
                    "timestamp": timestamp,
                    "source_file": str(path.relative_to(source)),
                    "source_row": row_number,
                }
                if sensor == "heart_rate":
                    heart_rate = _number(row, "heartRate", "heart_rate", "hr", "bpm")
                    if heart_rate is None:
                        continue
                    record["heart_rate_bpm"] = heart_rate
                    status = _column(row, "status", "hr_status")
                    if status is not None:
                        record["sensor_status"] = status
                elif sensor == "accelerometer":
                    axes = tuple(_number(row, name) for name in ("x", "y", "z"))
                    if any(value is None for value in axes):
                        continue
                    magnitude = math.sqrt(sum(float(value) ** 2 for value in axes))
                    record["motion_intensity"] = abs(magnitude - 9.81)
                    record["acceleration_m_s2"] = {
                        name: value for name, value in zip(("x", "y", "z"), axes)
                    }
                elif sensor == "skin_temperature":
                    skin = _number(row, "bodyTemperature", "skinTemperature", "body", "skin")
                    ambient = _number(row, "ambientTemperature", "ambient")
                    if skin is None and ambient is None:
                        continue
                    if skin is not None:
                        record["skin_temperature_c"] = skin
                    if ambient is not None:
                        record["ambient_temperature_c"] = ambient
                elif sensor == "ppg":
                    ppg = _number(row, "ppg", "green", "ppgGreen", "value")
                    if ppg is None:
                        continue
                    record["ppg"] = ppg
                yield record
