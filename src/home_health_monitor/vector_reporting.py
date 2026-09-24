from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from home_health_monitor.datasets.pulse_transit_features import DATASET_FEATURE_NAMES
from home_health_monitor.gateway.vector_training import load_real_ppg_vectors


def _style(plt: Any) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#FFFFFF",
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#475569",
            "axes.labelcolor": "#1E293B",
            "text.color": "#0F172A",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
            "grid.color": "#E2E8F0",
            "font.size": 10,
        }
    )


def _save(fig: Any, path: Path) -> None:
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")


def render_real_ppg_plots(
    feature_report_path: str | Path,
    training_report_path: str | Path,
    vectors_path: str | Path,
    output_directory: str | Path,
) -> tuple[Path, ...]:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise SystemExit("plotting requires: python -m pip install -e '.[analysis,gateway]'") from exc
    feature_report = json.loads(Path(feature_report_path).read_text(encoding="utf-8"))
    training_report = json.loads(Path(training_report_path).read_text(encoding="utf-8"))
    rows = load_real_ppg_vectors(vectors_path)
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    _style(plt)
    created = []

    activities = ("sit", "walk", "run")
    validation = feature_report["ppg_vs_ecg_heart_rate_by_activity"]
    x = np.arange(len(activities))
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.bar(x - 0.18, [validation[name]["mae_bpm"] for name in activities], 0.36, label="MAE", color="#2563EB")
    ax.bar(x + 0.18, [validation[name]["p95_absolute_error_bpm"] for name in activities], 0.36, label="95th percentile", color="#D97706")
    ax.set_xticks(x, [name.title() for name in activities])
    ax.set_ylabel("Absolute heart-rate error (BPM)")
    ax.set_title("PPG-derived heart rate checked against ECG", loc="left", fontweight="bold")
    ax.grid(axis="y", linewidth=0.7)
    ax.legend(frameon=False)
    path = destination / "ppg-heart-rate-validation.png"
    _save(fig, path)
    plt.close(fig)
    created.append(path)

    history = training_report["training"]["final_loss"]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.plot(range(1, len(history) + 1), history, color="#2563EB", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean squared reconstruction loss")
    ax.set_title("Final autoencoder training loss", loc="left", fontweight="bold")
    ax.grid(axis="y", linewidth=0.7)
    path = destination / "real-ppg-training-loss.png"
    _save(fig, path)
    plt.close(fig)
    created.append(path)

    threshold = training_report["thresholds"]["persistent_error"]
    oof = [item["p95_vector_error"] for item in training_report["cross_validation"]["intervals"]]
    locked = [item["p95_vector_error"] for item in training_report["locked_normal_test"]["intervals"]]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.boxplot([oof, locked], tick_labels=["Cross-validation", "Locked test"], patch_artist=True)
    ax.axhline(threshold, color="#DC2626", linestyle="--", linewidth=2, label="Threshold")
    ax.set_ylabel("Eight-minute anomaly score")
    ax.set_title("Healthy interval reconstruction scores", loc="left", fontweight="bold")
    ax.grid(axis="y", linewidth=0.7)
    ax.legend(frameon=False)
    path = destination / "real-ppg-normal-score-distribution.png"
    _save(fig, path)
    plt.close(fig)
    created.append(path)

    cv_activity = training_report["cross_validation"]["intervals_by_activity"]
    test_activity = training_report["locked_normal_test"]["intervals_by_activity"]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.bar(x - 0.18, [cv_activity[name]["anomaly_fraction"] for name in activities], 0.36, label="Cross-validation", color="#2563EB")
    ax.bar(x + 0.18, [test_activity[name]["anomaly_fraction"] for name in activities], 0.36, label="Locked test", color="#0F766E")
    ax.set_xticks(x, [name.title() for name in activities])
    ax.set_ylabel("Healthy intervals flagged")
    ax.set_ylim(0, max(0.15, ax.get_ylim()[1]))
    ax.set_title("False anomaly fraction by activity", loc="left", fontweight="bold")
    ax.grid(axis="y", linewidth=0.7)
    ax.legend(frameon=False)
    path = destination / "real-ppg-false-anomalies-by-activity.png"
    _save(fig, path)
    plt.close(fig)
    created.append(path)

    controlled = training_report["controlled_sensitivity"]
    scenario_names = sorted(controlled)
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    fractions = [controlled[name]["summary"]["anomaly_fraction"] for name in scenario_names]
    bars = ax.barh(
        [name.replace("_", " ").title() for name in scenario_names],
        fractions,
        color="#D97706",
    )
    for bar, value in zip(bars, fractions):
        ax.text(value + 0.015, bar.get_y() + bar.get_height() / 2, f"{value:.0%}", va="center")
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Controlled intervals flagged")
    ax.set_title("Controlled sensitivity checks (not clinical labels)", loc="left", fontweight="bold")
    ax.grid(axis="x", linewidth=0.7)
    path = destination / "real-ppg-controlled-sensitivity.png"
    _save(fig, path)
    plt.close(fig)
    created.append(path)

    values = np.asarray([row.values for row in rows], dtype=np.float64)
    correlation = np.corrcoef(values, rowvar=False)
    labels = [name.replace("_", " ") for name in DATASET_FEATURE_NAMES]
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(correlation, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=55, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_title("Extracted feature correlation", loc="left", fontweight="bold")
    fig.colorbar(image, ax=ax, shrink=0.78, label="Pearson correlation")
    path = destination / "real-ppg-feature-correlation.png"
    _save(fig, path)
    plt.close(fig)
    created.append(path)

    return tuple(created)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render real-PPG model evaluation plots")
    parser.add_argument("feature_report", type=Path)
    parser.add_argument("training_report", type=Path)
    parser.add_argument("vectors", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    for path in render_real_ppg_plots(
        args.feature_report, args.training_report, args.vectors, args.output_directory
    ):
        print(path)


if __name__ == "__main__":
    main()
