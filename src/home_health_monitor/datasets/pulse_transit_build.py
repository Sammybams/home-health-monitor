from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from .pulse_transit import PulseTransitArchive
from .pulse_transit_features import DATASET_FEATURE_NAMES, extract_segment_features


DATASET_ID = "pulse-transit-time-ppg-1.1.0"
FEATURE_MANIFEST_ID = "real-ppg-core-v2-candidate-1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _error_summary(errors: list[float]) -> dict[str, float | int | None]:
    if not errors:
        return {"count": 0, "mae_bpm": None, "rmse_bpm": None, "bias_bpm": None, "p95_absolute_error_bpm": None}
    values = np.asarray(errors, dtype=np.float64)
    absolute = np.abs(values)
    return {
        "count": int(values.size),
        "mae_bpm": float(np.mean(absolute)),
        "rmse_bpm": float(np.sqrt(np.mean(values**2))),
        "bias_bpm": float(np.mean(values)),
        "p95_absolute_error_bpm": float(np.percentile(absolute, 95)),
    }


def build_feature_dataset(
    archive_path: str | Path,
    output_path: str | Path,
    *,
    segment_seconds: float = 5.0,
    stride_seconds: float = 30.0,
) -> dict[str, Any]:
    """Stream the ZIP into an atomic JSONL feature dataset and return its audit."""
    source = Path(archive_path)
    destination = Path(output_path)
    if not math.isfinite(segment_seconds) or segment_seconds <= 0:
        raise ValueError("segment_seconds must be finite and positive")
    if not math.isfinite(stride_seconds) or stride_seconds < segment_seconds:
        raise ValueError("stride_seconds must be finite and at least segment_seconds")
    destination.parent.mkdir(parents=True, exist_ok=True)
    activity_counts: dict[str, int] = defaultdict(int)
    activity_errors: dict[str, list[float]] = defaultdict(list)
    all_errors: list[float] = []
    extraction_failures: list[dict[str, Any]] = []
    feature_rows = 0
    heart_rate_valid = 0
    ecg_reference_valid = 0

    temporary = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary_path = Path(temporary.name)
    try:
        with temporary, PulseTransitArchive(source) as archive:
            for record in archive.records():
                segment_rows = int(round(record.sample_rate_hz * segment_seconds))
                stride_rows = int(round(record.sample_rate_hz * stride_seconds))
                for segment_index, segment in enumerate(
                    archive.iter_csv_segments(
                        record.name,
                        segment_rows,
                        stride_rows=stride_rows,
                        malformed="skip",
                    )
                ):
                    try:
                        result = extract_segment_features(
                            [row.values for row in segment],
                            sample_rate_hz=record.sample_rate_hz,
                        )
                    except ValueError as exc:
                        extraction_failures.append(
                            {
                                "record": record.name,
                                "segment_index": segment_index,
                                "first_csv_line": segment[0].line_number,
                                "reason": str(exc),
                            }
                        )
                        continue
                    first_sample = segment[0].line_number - 2
                    item = {
                        "schema_version": 1,
                        "dataset_id": DATASET_ID,
                        "feature_manifest_id": FEATURE_MANIFEST_ID,
                        "feature_names": list(DATASET_FEATURE_NAMES),
                        "subject_id": record.subject_id,
                        "record": record.name,
                        "activity": record.activity,
                        "segment_index": segment_index,
                        "start_seconds": first_sample / record.sample_rate_hz,
                        "duration_seconds": segment_seconds,
                        "values": [result.values[name] for name in DATASET_FEATURE_NAMES],
                        "ecg_reference_hr_bpm": result.ecg_reference_hr_bpm,
                        "demographics": {
                            "age": int(record.metadata["age"]),
                            "gender": record.metadata["gender"],
                        },
                    }
                    temporary.write(json.dumps(item, separators=(",", ":"), sort_keys=True) + "\n")
                    feature_rows += 1
                    activity_counts[record.activity] += 1
                    if result.values["heart_rate_valid"] == 1.0:
                        heart_rate_valid += 1
                    if result.ecg_reference_hr_bpm is not None:
                        ecg_reference_valid += 1
                    if result.values["heart_rate_valid"] == 1.0 and result.ecg_reference_hr_bpm is not None:
                        error = result.values["heart_rate_bpm"] - result.ecg_reference_hr_bpm
                        all_errors.append(error)
                        activity_errors[record.activity].append(error)
        temporary_path.replace(destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    return {
        "schema_version": 1,
        "dataset_id": DATASET_ID,
        "feature_manifest_id": FEATURE_MANIFEST_ID,
        "source_archive_name": source.name,
        "source_archive_sha256": _sha256_file(source),
        "output_name": destination.name,
        "output_sha256": _sha256_file(destination),
        "segment_seconds": segment_seconds,
        "stride_seconds": stride_seconds,
        "feature_names": list(DATASET_FEATURE_NAMES),
        "feature_rows": feature_rows,
        "activity_counts": dict(sorted(activity_counts.items())),
        "extraction_failure_count": len(extraction_failures),
        "extraction_failures": extraction_failures,
        "heart_rate_valid_fraction": heart_rate_valid / feature_rows if feature_rows else 0.0,
        "ecg_reference_valid_fraction": ecg_reference_valid / feature_rows if feature_rows else 0.0,
        "ppg_vs_ecg_heart_rate": _error_summary(all_errors),
        "ppg_vs_ecg_heart_rate_by_activity": {
            name: _error_summary(errors) for name, errors in sorted(activity_errors.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build short real-PPG feature vectors directly from the source ZIP"
    )
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--segment-seconds", type=float, default=5.0)
    parser.add_argument("--stride-seconds", type=float, default=30.0)
    args = parser.parse_args()
    report = build_feature_dataset(
        args.archive,
        args.output,
        segment_seconds=args.segment_seconds,
        stride_seconds=args.stride_seconds,
    )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
