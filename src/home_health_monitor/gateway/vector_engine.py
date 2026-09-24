from __future__ import annotations

from datetime import datetime, timedelta
from .store import GatewayStore
from .vector_autoencoder import (
    MINIMUM_PERSONAL_CALIBRATION_INTERVALS,
    VectorAutoencoderModel,
    calibrate_personal_reconstruction_threshold,
)
from .vector_contracts import VectorInterval


class VectorGatewayEngine:
    def __init__(self, store: GatewayStore, model: VectorAutoencoderModel) -> None:
        self.store = store
        self.model = model

    def ingest(self, interval: VectorInterval) -> dict:
        duplicate = self.store.vector_interval_by_sequence(interval.device_id, interval.sequence)
        if duplicate is not None:
            return duplicate
        history = self.store.vector_history(interval.subject_id)
        elapsed_hours = 0.0 if not history else max(
            0.0,
            (interval.interval_end - datetime.fromisoformat(history[0]["interval_start"])).total_seconds() / 3600,
        )
        calibration_scores = [row["score"] for row in history if not row["above_threshold"]]
        personal_threshold = None
        if elapsed_hours >= 48 and len(calibration_scores) >= MINIMUM_PERSONAL_CALIBRATION_INTERVALS:
            personal_threshold = calibrate_personal_reconstruction_threshold(
                calibration_scores[:MINIMUM_PERSONAL_CALIBRATION_INTERVALS], elapsed_hours=elapsed_hours
            )
        previous_above = bool(history[-1]["above_threshold"]) if history else False
        reconstructions = [self.model.predict(vector) for vector in interval.vectors]
        prediction = self.model.aggregate(
            reconstructions,
            previous_interval_above_threshold=previous_above,
            personal_threshold=personal_threshold,
        )
        result = {
            "decision": prediction.decision,
            "model_id": self.model.model_id,
            "subject_id": interval.subject_id,
            "interval_start": interval.interval_start.isoformat(),
            "interval_end": interval.interval_end.isoformat(),
            "score": prediction.score,
            "threshold": prediction.threshold,
            "threshold_source": "personal" if personal_threshold is not None else "population",
            "above_threshold": prediction.above_threshold,
            "severe": prediction.severe,
            "persistent": prediction.persistent,
            "vector_count": prediction.vector_count,
            "expected_vector_count": prediction.expected_vector_count,
            "coverage": prediction.coverage,
            "contributing_signals": list(prediction.contributing_signals),
            "calibration": {
                "status": "ready" if personal_threshold is not None else "collecting",
                "elapsed_hours": elapsed_hours,
                "accepted_intervals": len(calibration_scores),
                "required_hours": 48,
                "required_intervals": MINIMUM_PERSONAL_CALIBRATION_INTERVALS,
            },
        }
        self.store.save_vector_interval(
            subject_id=interval.subject_id, device_id=interval.device_id,
            sequence=interval.sequence, interval_start=interval.interval_start,
            interval_end=interval.interval_end, score=prediction.score,
            above_threshold=prediction.above_threshold, payload=result,
        )
        self.store.delete_vector_intervals_before(interval.interval_end - timedelta(days=365))
        return result
