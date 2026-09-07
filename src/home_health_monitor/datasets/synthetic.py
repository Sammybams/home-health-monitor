from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator


def _is_corrected_v2(path: Path) -> bool:
    name = path.stem.lower().replace("-", "_").replace(" ", "_")
    return "corrected" in name and ("v2" in name or "version_2" in name or "version2" in name)


def convert_synthetic(path: str | Path) -> Iterator[dict[str, object]]:
    source = Path(path)
    if not _is_corrected_v2(source):
        raise ValueError("only the corrected Version 2 synthetic file may be converted")
    with source.open(newline="", encoding="utf-8-sig") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), 2):
            yield {
                "dataset_role": "synthetic_demo",
                "production_training_eligible": False,
                "source_row": row_number,
                "values": dict(row),
            }
