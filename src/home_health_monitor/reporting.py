from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


FEATURES = (
    ("heart_rate_bpm", "Heart rate"),
    ("spo2_percent", "SpO₂"),
    ("temperature_c", "Temperature"),
    ("motion_intensity", "Motion"),
)


def availability_matrix(
    evidence: dict[str, Any],
) -> tuple[list[str], list[str], list[list[int]]]:
    sources = [str(item["dataset"]) for item in evidence["sources"]]
    labels = [label for _, label in FEATURES]
    matrix = [
        [int(bool(source["feature_availability"].get(name))) for name, _ in FEATURES]
        for source in evidence["sources"]
    ]
    return sources, labels, matrix


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


def render_development_plots(
    evidence_path: str | Path,
    training_report_path: str | Path,
    output_directory: str | Path,
) -> tuple[Path, ...]:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
    except ImportError as exc:
        raise SystemExit("plotting requires: python -m pip install -e '.[analysis]'") from exc

    evidence = json.loads(Path(evidence_path).read_text(encoding="utf-8"))
    report = json.loads(Path(training_report_path).read_text(encoding="utf-8"))
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    _style(plt)
    created = []

    sources, features, matrix = availability_matrix(evidence)
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    ax.imshow(matrix, cmap=ListedColormap(["#E2E8F0", "#2563EB"]), vmin=0, vmax=1)
    ax.set_xticks(range(len(features)), features)
    ax.set_yticks(range(len(sources)), sources)
    for row, values in enumerate(matrix):
        for column, value in enumerate(values):
            ax.text(column, row, "Available" if value else "Missing", ha="center", va="center", color="white" if value else "#475569", fontsize=9)
    ax.set_title("Feature availability by reviewed dataset", loc="left", fontweight="bold")
    ax.set_xlabel("Target home-gateway input")
    ax.tick_params(length=0)
    path = destination / "dataset-feature-availability.png"
    _save(fig, path)
    plt.close(fig)
    created.append(path)

    history = report["history"]
    epochs = range(1, len(history["loss"]) + 1)
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.plot(epochs, history["loss"], color="#2563EB", linewidth=2, label="Training")
    ax.plot(epochs, history["val_loss"], color="#D97706", linewidth=2, linestyle="--", label="Validation")
    ax.set_title("Autoencoder training loss", loc="left", fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean squared reconstruction loss")
    ax.grid(axis="y", linewidth=0.7)
    ax.legend(frameon=False)
    path = destination / "autoencoder-training-loss.png"
    _save(fig, path)
    plt.close(fig)
    created.append(path)

    normal = report["normal_test_scores"]
    simulated = report["simulated_anomaly_scores"]
    threshold = float(report["threshold"])
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    bins = 16
    ax.hist(normal, bins=bins, color="#2563EB", alpha=0.72, label="Held-out normal", edgecolor="#1E3A8A")
    ax.hist(simulated, bins=bins, color="#D97706", alpha=0.55, label="Controlled anomaly", edgecolor="#92400E", hatch="//")
    ax.axvline(threshold, color="#0F172A", linewidth=2, linestyle=":", label="Decision threshold")
    ax.set_title("Reconstruction-error distributions", loc="left", fontweight="bold")
    ax.set_xlabel("Quantized-model mean squared error")
    ax.set_ylabel("Windows")
    ax.grid(axis="y", linewidth=0.7)
    ax.legend(frameon=False)
    path = destination / "autoencoder-score-distribution.png"
    _save(fig, path)
    plt.close(fig)
    created.append(path)

    example = report["example"]
    labels = example["feature_names"][:4]
    observed = example["simulated_input"]
    reconstructed = example["simulated_reconstruction"]
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    for feature, (ax, label) in enumerate(zip(axes, labels)):
        ax.plot([row[feature] for row in observed], color="#2563EB", linewidth=1.4, label="Input")
        ax.plot([row[feature] for row in reconstructed], color="#D97706", linewidth=1.4, linestyle="--", label="Reconstruction")
        ax.axvspan(120, 168, color="#FDE68A", alpha=0.35)
        ax.set_ylabel(label)
        ax.grid(axis="y", linewidth=0.7)
    axes[0].set_title("Input and autoencoder reconstruction", loc="left", fontweight="bold")
    axes[0].legend(frameon=False, ncol=2)
    axes[-1].set_xlabel("Five-minute step across a 24-hour window")
    path = destination / "autoencoder-reconstruction-example.png"
    _save(fig, path)
    plt.close(fig)
    created.append(path)

    return tuple(created)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render development autoencoder plots")
    parser.add_argument("evidence", type=Path)
    parser.add_argument("training_report", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    for path in render_development_plots(
        args.evidence, args.training_report, args.output_directory
    ):
        print(path)


if __name__ == "__main__":
    main()
