from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

from .contracts import InputError, parse_request
from .features import extract_features, feature_names


def _load(path: Path):
    try:
        import numpy as np
        from sklearn.model_selection import GroupShuffleSplit
    except ImportError as exc:
        raise SystemExit("training requires: python -m pip install -e '.[train]'") from exc

    names = feature_names()
    rows, current, future, groups = [], [], [], []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                labels = item.pop("labels")
                request = parse_request(item)
                if request.subject_id is None:
                    raise InputError("subject_id is required for grouped evaluation")
                if labels["current_unhealthy"] not in (0, 1) or labels["unhealthy_within_horizon"] not in (0, 1):
                    raise InputError("labels must be 0 or 1")
                values, _ = extract_features(request)
            except (json.JSONDecodeError, KeyError, InputError) as exc:
                raise SystemExit(f"invalid training row {line_number}: {exc}") from exc
            rows.append([values[name] for name in names])
            current.append(labels["current_unhealthy"])
            future.append(labels["unhealthy_within_horizon"])
            groups.append(request.subject_id)
    if len(rows) < 100 or len(set(groups)) < 10:
        raise SystemExit("need at least 100 windows from at least 10 distinct subjects")
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_indices, test_indices = next(splitter.split(rows, current, groups))
    return np.asarray(rows), np.asarray(current), np.asarray(future), train_indices, test_indices, names


def train(input_path: Path, output_path: Path, horizon: int) -> None:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    x, y_current, y_future, train_idx, test_idx, names = _load(input_path)
    if any(len(np.unique(target[train_idx])) != 2 or len(np.unique(target[test_idx])) != 2 for target in (y_current, y_future)):
        raise SystemExit("both labels must contain both classes in grouped train and test sets")
    scaler = StandardScaler().fit(x[train_idx])
    x_train, x_test = scaler.transform(x[train_idx]), scaler.transform(x[test_idx])
    heads = {}
    metrics = {}
    for name, target in (("current", y_current), ("future", y_future)):
        # Preserve observed prevalence so predict_proba remains useful for
        # calibration assessment. Sensitivity trade-offs belong in a reviewed
        # threshold, not an implicit class-weight distortion.
        classifier = LogisticRegression(max_iter=2000, random_state=42)
        classifier.fit(x_train, target[train_idx])
        probabilities = classifier.predict_proba(x_test)[:, 1]
        auc = float(roc_auc_score(target[test_idx], probabilities))
        brier = float(brier_score_loss(target[test_idx], probabilities))
        print(f"{name}: roc_auc={auc:.4f} brier={brier:.4f}")
        metrics[name] = {"test_roc_auc": auc, "test_brier": brier}
        heads[name] = {
            "coefficients": classifier.coef_[0].tolist(),
            "intercept": float(classifier.intercept_[0]),
            "threshold": 0.5,
        }
    artifact = {
        "schema_version": 1,
        "model_id": f"hhm-{uuid.uuid4().hex[:12]}",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "future_horizon_hours": horizon,
        "training_summary": {
            "windows": int(len(x)),
            "train_windows": int(len(train_idx)),
            "test_windows": int(len(test_idx)),
            "metrics": metrics,
        },
        "features": list(names),
        "scaler": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()},
        "heads": heads,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and export a tiny model")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--future-horizon-hours", type=int, default=24)
    args = parser.parse_args()
    if not 1 <= args.future_horizon_hours <= 168:
        parser.error("future horizon must be between 1 and 168 hours")
    train(args.input, args.output, args.future_horizon_hours)


if __name__ == "__main__":
    main()
