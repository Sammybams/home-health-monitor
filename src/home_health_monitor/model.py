from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

from .features import feature_names


class ModelError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class LinearHead:
    coefficients: tuple[float, ...]
    intercept: float
    threshold: float


@dataclass(frozen=True, slots=True)
class TinyModel:
    model_id: str
    trained_at: str
    future_horizon_hours: int
    names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    current: LinearHead
    future: LinearHead

    @classmethod
    def load(cls, path: str | Path) -> "TinyModel":
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            if data.get("schema_version") != 1:
                raise ModelError("unsupported model schema_version")
            names = tuple(data["features"])
            if names != feature_names():
                raise ModelError("model feature contract does not match this service")
            means = tuple(float(value) for value in data["scaler"]["mean"])
            scales = tuple(float(value) for value in data["scaler"]["scale"])
            if len(means) != len(names) or len(scales) != len(names) or any(v <= 0 for v in scales):
                raise ModelError("invalid scaler dimensions")
            def head(name: str) -> LinearHead:
                item = data["heads"][name]
                coefficients = tuple(float(value) for value in item["coefficients"])
                if len(coefficients) != len(names):
                    raise ModelError(f"invalid {name} coefficient dimensions")
                threshold = float(item["threshold"])
                if not 0 < threshold < 1:
                    raise ModelError(f"invalid {name} threshold")
                return LinearHead(coefficients, float(item["intercept"]), threshold)
            return cls(
                model_id=str(data["model_id"]),
                trained_at=str(data["trained_at"]),
                future_horizon_hours=int(data["future_horizon_hours"]),
                names=names,
                means=means,
                scales=scales,
                current=head("current"),
                future=head("future"),
            )
        except ModelError:
            raise
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ModelError(f"could not load model: {exc}") from exc

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value >= 0:
            return 1 / (1 + math.exp(-value))
        exp_value = math.exp(value)
        return exp_value / (1 + exp_value)

    def _predict_head(self, vector: tuple[float, ...], head: LinearHead) -> float:
        score = head.intercept + sum(value * weight for value, weight in zip(vector, head.coefficients))
        return self._sigmoid(score)

    def predict(self, features: dict[str, float], warnings: list[str]) -> dict[str, Any]:
        raw = tuple(features[name] for name in self.names)
        vector = tuple((value - mean) / scale for value, mean, scale in zip(raw, self.means, self.scales))
        current_probability = self._predict_head(vector, self.current)
        future_probability = self._predict_head(vector, self.future)
        return {
            "prediction": {
                "current_risk": {
                    "classification": "higher_risk" if current_probability >= self.current.threshold else "lower_risk",
                    "probability": round(current_probability, 6),
                    "threshold": self.current.threshold,
                },
                "future_risk": {
                    "classification": "higher_risk" if future_probability >= self.future.threshold else "lower_risk",
                    "probability": round(future_probability, 6),
                    "horizon_hours": self.future_horizon_hours,
                    "threshold": self.future.threshold,
                },
            },
            "model": {"id": self.model_id, "trained_at": self.trained_at},
            "data_quality": {"warnings": warnings},
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "disclaimer": "Risk-screening output only; not a diagnosis or emergency service.",
        }
