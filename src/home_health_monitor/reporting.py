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


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def binary_performance(
    normal_scores: list[float], anomaly_scores: list[float], *, threshold: float
) -> dict[str, Any]:
    """Summarize a controlled binary evaluation at one fixed threshold."""
    tn = sum(score < threshold for score in normal_scores)
    fp = len(normal_scores) - tn
    tp = sum(score >= threshold for score in anomaly_scores)
    fn = len(anomaly_scores) - tp
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    return {
        "threshold": threshold,
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "accuracy": _ratio(tp + tn, tp + tn + fp + fn),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "balanced_accuracy": (recall + specificity) / 2,
        "f1": _ratio(2 * precision * recall, precision + recall),
        "normal_windows": len(normal_scores),
        "controlled_anomaly_windows": len(anomaly_scores),
    }


def scenario_score_summary(
    scenarios: list[str], scores: list[float], *, threshold: float
) -> dict[str, dict[str, float | int]]:
    if len(scenarios) != len(scores):
        raise ValueError("scenario names and scores must have equal lengths")
    grouped: dict[str, list[float]] = {}
    for scenario, score in zip(scenarios, scores):
        grouped.setdefault(scenario, []).append(float(score))
    return {
        name: {
            "count": len(values),
            "mean_score": sum(values) / len(values),
            "minimum_score": min(values),
            "maximum_score": max(values),
            "detection_fraction": sum(value >= threshold for value in values)
            / len(values),
        }
        for name, values in sorted(grouped.items())
    }


def performance_summary(report: dict[str, Any]) -> dict[str, Any]:
    threshold = float(report["threshold"])
    return {
        "artifact_role": report["artifact_role"],
        "model_id": report["model_id"],
        "evaluation_scope": "held_out_generated_normal_vs_controlled_simulation",
        "binary": binary_performance(
            report["normal_test_scores"],
            report["simulated_anomaly_scores"],
            threshold=threshold,
        ),
        "by_scenario": scenario_score_summary(
            report["simulated_anomaly_scenarios"],
            report["simulated_anomaly_scores"],
            threshold=threshold,
        ),
    }


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
    threshold = float(report["threshold"])
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

    performance = performance_summary(report)
    confusion = performance["binary"]["confusion"]
    confusion_values = [
        [confusion["tn"], confusion["fp"]],
        [confusion["fn"], confusion["tp"]],
    ]
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.imshow(confusion_values, cmap="Blues", vmin=0)
    ax.set_xticks((0, 1), ("Predicted normal", "Predicted anomaly"))
    ax.set_yticks((0, 1), ("Actual normal", "Controlled anomaly"))
    for row, values in enumerate(confusion_values):
        for column, value in enumerate(values):
            ax.text(column, row, str(value), ha="center", va="center", fontsize=18, fontweight="bold", color="white" if value > max(map(max, confusion_values)) / 2 else "#0F172A")
    ax.set_title("Development evaluation confusion matrix", loc="left", fontweight="bold")
    ax.tick_params(length=0)
    path = destination / "autoencoder-confusion-matrix.png"
    _save(fig, path)
    plt.close(fig)
    created.append(path)

    grouped_scores: dict[str, list[float]] = {}
    for scenario, score in zip(
        report["simulated_anomaly_scenarios"], report["simulated_anomaly_scores"]
    ):
        grouped_scores.setdefault(scenario, []).append(float(score))
    scenario_names = sorted(grouped_scores)
    display_scenarios = [name.replace("sustained_", "").replace("_", " ").title() for name in scenario_names]
    fig, ax = plt.subplots(figsize=(9.4, 4.8))
    boxes = ax.boxplot(
        [grouped_scores[name] for name in scenario_names],
        tick_labels=display_scenarios,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#0F172A", "linewidth": 2},
    )
    for box in boxes["boxes"]:
        box.set(facecolor="#F4B76E", edgecolor="#92400E")
    for index, name in enumerate(scenario_names, 1):
        values = grouped_scores[name]
        offsets = [index + (position - (len(values) - 1) / 2) * 0.018 for position in range(len(values))]
        ax.scatter(offsets, values, color="#2563EB", edgecolor="white", linewidth=0.5, zorder=3)
    ax.axhline(threshold, color="#0F172A", linewidth=2, linestyle=":", label="Decision threshold")
    ax.set_title("Reconstruction error by controlled anomaly", loc="left", fontweight="bold")
    ax.set_ylabel("Quantized-model mean squared error")
    ax.grid(axis="y", linewidth=0.7)
    ax.legend(frameon=False)
    path = destination / "autoencoder-scenario-performance.png"
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
    display_names = dict(FEATURES)
    labels = [display_names.get(name, name) for name in example["feature_names"][:4]]
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
    report = json.loads(args.training_report.read_text(encoding="utf-8"))
    summary_path = args.training_report.with_name("performance-summary.json")
    summary_path.write_text(
        json.dumps(performance_summary(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(summary_path)
    for path in render_development_plots(
        args.evidence, args.training_report, args.output_directory
    ):
        print(path)


if __name__ == "__main__":
    main()
