from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from home_health_monitor.datasets.pulse_transit_features import DATASET_FEATURE_NAMES
from home_health_monitor.gateway.vector_training import load_real_ppg_vectors
from home_health_monitor.gateway.vector_training import controlled_anomalies


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


def _reconstruct(model_path: Path, values: Any, np: Any) -> Any:
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            try:
                from tensorflow.lite.python.interpreter import Interpreter
            except ImportError as exc:
                raise SystemExit("reconstruction plots require LiteRT or TensorFlow") from exc
    # Preserving tensors keeps this evidence path on the reference kernels and
    # avoids plotting-library/OpenMP delegate conflicts on development Macs.
    interpreter = Interpreter(
        model_path=str(model_path), experimental_preserve_all_tensors=True
    )
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    input_scale, input_zero = input_detail["quantization"]
    output_scale, output_zero = output_detail["quantization"]
    reconstructed = []
    for row in values:
        quantized = np.clip(np.rint(row / input_scale) + input_zero, -128, 127).astype(np.int8)
        interpreter.set_tensor(input_detail["index"], quantized[np.newaxis, :])
        interpreter.invoke()
        output = interpreter.get_tensor(output_detail["index"])[0]
        reconstructed.append((output.astype(np.float32) - output_zero) * output_scale)
    return np.asarray(reconstructed, dtype=np.float64)


def build_reconstruction_evidence(
    training_report_path: str | Path,
    vectors_path: str | Path,
    model_path: str | Path,
    metadata_path: str | Path,
    output_path: str | Path,
) -> Path:
    import numpy as np

    report = json.loads(Path(training_report_path).read_text(encoding="utf-8"))
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    locked_subjects = set(report["participant_plan"]["locked_test_subjects"])
    rows = [row for row in load_real_ppg_vectors(vectors_path) if row.subject_id in locked_subjects]
    raw = np.asarray([row.values for row in rows], dtype=np.float64)
    normalization = metadata["input"]["normalization"]
    center = np.asarray(normalization["center"], dtype=np.float64)
    scale = np.asarray(normalization["scale"], dtype=np.float64)
    actual = (raw - center) / scale
    invalid = raw[:, DATASET_FEATURE_NAMES.index("heart_rate_valid")] == 0
    for name in ("heart_rate_bpm", "ppg_rr_interval_std_ms"):
        actual[invalid, DATASET_FEATURE_NAMES.index(name)] = 0
    reconstructed = _reconstruct(Path(model_path), actual, np)
    residual = actual - reconstructed
    controlled_values, scenario_labels = controlled_anomalies(actual, np)
    controlled_reconstructed = _reconstruct(Path(model_path), controlled_values, np)
    scenario_order = ["normal", "heart_rate_shift", "temperature_shift", "motion_shift", "combined_drift"]
    error_rows = [np.mean(np.abs(residual), axis=0)]
    for scenario in scenario_order[1:]:
        mask = np.asarray(scenario_labels) == scenario
        error_rows.append(np.mean(np.abs(controlled_values[mask] - controlled_reconstructed[mask]), axis=0))
    destination = Path(output_path)
    destination.write_text(json.dumps({
        "schema_version": 1,
        "scope": "locked healthy participants and controlled shifts",
        "locked_subjects": sorted(locked_subjects),
        "locked_vectors": len(rows),
        "units": "robust-standardized feature units",
        "feature_names": list(DATASET_FEATURE_NAMES),
        "actual": actual.tolist(),
        "reconstructed": reconstructed.tolist(),
        "scenario_names": scenario_order,
        "scenario_mean_absolute_residuals": np.asarray(error_rows).tolist(),
    }, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    return destination


def render_real_ppg_plots(
    feature_report_path: str | Path,
    training_report_path: str | Path,
    vectors_path: str | Path,
    output_directory: str | Path,
    reconstruction_evidence_path: str | Path | None = None,
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

    if reconstruction_evidence_path is not None:
        evidence = json.loads(Path(reconstruction_evidence_path).read_text(encoding="utf-8"))
        actual = np.asarray(evidence["actual"], dtype=np.float64)
        reconstructed = np.asarray(evidence["reconstructed"], dtype=np.float64)
        residual = actual - reconstructed
        labels = [name.replace("_", " ") for name in DATASET_FEATURE_NAMES]

        fig, axes = plt.subplots(3, 4, figsize=(13, 10))
        for index, ax in enumerate(axes.flat):
            ax.scatter(actual[:, index], reconstructed[:, index], s=9, alpha=0.45, color="#2563EB")
            low = min(actual[:, index].min(), reconstructed[:, index].min())
            high = max(actual[:, index].max(), reconstructed[:, index].max())
            ax.plot([low, high], [low, high], "--", color="#DC2626", linewidth=1)
            ax.set_title(labels[index], fontsize=9)
            ax.grid(linewidth=0.5)
        fig.supxlabel("Actual standardized value")
        fig.supylabel("Reconstructed standardized value")
        fig.suptitle("Locked participants: actual versus reconstructed features", fontweight="bold")
        path = destination / "real-ppg-actual-vs-reconstructed.png"
        _save(fig, path); plt.close(fig); created.append(path)

        mae = np.mean(np.abs(residual), axis=0)
        rmse = np.sqrt(np.mean(residual**2, axis=0))
        fig, ax = plt.subplots(figsize=(10, 5.5))
        y = np.arange(len(labels))
        ax.barh(y + 0.18, rmse, 0.36, label="RMSE", color="#D97706")
        ax.barh(y - 0.18, mae, 0.36, label="MAE", color="#2563EB")
        ax.set_yticks(y, labels); ax.invert_yaxis()
        ax.set_xlabel("Error in robust-standardized units")
        ax.set_title("Locked-participant reconstruction error by feature", loc="left", fontweight="bold")
        ax.grid(axis="x", linewidth=0.7); ax.legend(frameon=False)
        path = destination / "real-ppg-reconstruction-error-by-feature.png"
        _save(fig, path); plt.close(fig); created.append(path)

        scores = np.mean(residual**2, axis=1)
        example_indices = [int(np.argsort(scores)[len(scores)//2]), int(np.argmax(scores))]
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        for ax, row_index, title in zip(axes, example_indices, ("Typical locked normal vector", "Highest-error locked normal vector")):
            ax.plot(actual[row_index], marker="o", label="Actual", color="#2563EB")
            ax.plot(reconstructed[row_index], marker="s", label="Reconstructed", color="#D97706")
            ax.set_title(f"{title} — score {scores[row_index]:.3f}", loc="left")
            ax.grid(axis="y", linewidth=0.7); ax.legend(frameon=False)
        axes[-1].set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        fig.supylabel("Robust-standardized value")
        path = destination / "real-ppg-reconstruction-examples.png"
        _save(fig, path); plt.close(fig); created.append(path)

        scenario_order = evidence["scenario_names"]
        error_rows = evidence["scenario_mean_absolute_residuals"]
        fig, ax = plt.subplots(figsize=(12, 4.8))
        image = ax.imshow(np.asarray(error_rows), aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(labels)), labels, rotation=50, ha="right", fontsize=8)
        ax.set_yticks(range(len(scenario_order)), [name.replace("_", " ").title() for name in scenario_order])
        ax.set_title("Where reconstruction error grows under controlled shifts", loc="left", fontweight="bold")
        fig.colorbar(image, ax=ax, label="Mean absolute standardized residual")
        path = destination / "real-ppg-normal-vs-controlled-residuals.png"
        _save(fig, path); plt.close(fig); created.append(path)


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
    ax.boxplot([oof, locked], labels=["Cross-validation", "Locked test"], patch_artist=True)
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
    parser.add_argument("--reconstruction-evidence", type=Path)
    args = parser.parse_args()
    for path in render_real_ppg_plots(
        args.feature_report, args.training_report, args.vectors, args.output_directory,
        args.reconstruction_evidence,
    ):
        print(path)


if __name__ == "__main__":
    main()
