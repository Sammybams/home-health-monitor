from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest

from home_health_monitor.gateway.autoencoder import ReconstructionResult
from home_health_monitor.gateway.calibration import (
    CalibrationProfile,
    CalibrationSignal,
    profile_to_dict,
)
from home_health_monitor.gateway.contracts import parse_packet
from home_health_monitor.gateway.engine import GatewayEngine
from home_health_monitor.gateway.store import GatewayStore
from tests.test_gateway_contracts import valid_packet


NOW = datetime(2026, 9, 7, 20, tzinfo=timezone.utc)


def packet(
    sequence: int = 1,
    *,
    when: datetime = NOW,
    wearable_decision: str = "normal",
    quality: float = 0.95,
    heart_rate_bpm: float = 70.0,
):
    payload = valid_packet()
    payload["sequence"] = sequence
    payload["timestamp"] = when.isoformat()
    payload["heart_rate_bpm"] = heart_rate_bpm
    payload["quality"] = {name: quality for name in payload["quality"]}
    payload["wearable"]["decision"] = wearable_decision
    payload["wearable"]["reason_codes"] = (
        ["wearable_heart_rate_deviation"] if wearable_decision == "anomaly" else []
    )
    return parse_packet(payload)


def ready_profile() -> CalibrationProfile:
    return CalibrationProfile(
        schema_version=1,
        subject_id="subject-1",
        feature_manifest_id="features-v1",
        sensor_config_id="sensor-set-a",
        temperature_type="skin",
        temperature_site="wrist",
        started_at=NOW - timedelta(days=2),
        ready_at=NOW,
        expected_interval_seconds=60.0,
        packet_count=2881,
        coverage=1.0,
        low_motion_hours=12.0,
        signals={
            "heart_rate_bpm": CalibrationSignal(70.0, 3.0),
            "spo2_percent": CalibrationSignal(97.2, 0.5),
            "temperature_c": CalibrationSignal(33.8, 0.1),
            "motion_intensity": CalibrationSignal(0.14, 0.05),
        },
    )


class FixedModel:
    model_id = "fixed-test"
    consecutive_windows = 2

    def __init__(self, error: float, severe: bool = False) -> None:
        self.error = error
        self.severe = severe

    def predict(self, window):
        return ReconstructionResult(
            overall_error=self.error,
            feature_errors={name: self.error for name in window.feature_names},
            persistent_threshold=0.25,
            severe_threshold=0.75,
            is_above_threshold=self.error >= 0.25,
            is_severe=self.severe,
        )


class GatewayEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.store = GatewayStore(Path(self.temporary.name) / "gateway.db")

    def tearDown(self) -> None:
        self.store.close()
        self.temporary.cleanup()

    def test_wearable_anomaly_wins_immediately(self) -> None:
        decision = GatewayEngine(self.store).ingest(
            packet(wearable_decision="anomaly")
        )

        self.assertEqual("anomaly", decision.decision)
        self.assertEqual(("wearable",), decision.triggered_by)

    def test_first_packet_always_returns_normal_prediction(self) -> None:
        body = GatewayEngine(self.store).ingest(packet()).to_dict()

        self.assertEqual("normal", body["decision"])
        self.assertNotIn("disclaimer", body)
        self.assertEqual("collecting", body["calibration"]["status"])

    def test_persistent_sensor_failure_is_anomaly(self) -> None:
        engine = GatewayEngine(self.store)
        engine.ingest(packet(1, when=NOW, quality=0.2))
        engine.ingest(packet(2, when=NOW + timedelta(minutes=1), quality=0.2))

        decision = engine.ingest(
            packet(3, when=NOW + timedelta(minutes=2), quality=0.2)
        )

        self.assertEqual("anomaly", decision.decision)
        self.assertIn("data_quality", decision.triggered_by)
        self.assertIn("persistent_ppg_quality_failure", decision.reason_codes)

    def test_ready_baseline_detects_large_personal_deviation(self) -> None:
        self.store.save_calibration("subject-1", profile_to_dict(ready_profile()))

        decision = GatewayEngine(self.store).ingest(packet(1, heart_rate_bpm=85.0))

        self.assertEqual("anomaly", decision.decision)
        self.assertIn("gateway_baseline", decision.triggered_by)
        self.assertIn("heart_rate_bpm", decision.contributing_signals)

    def test_model_requires_persistent_error(self) -> None:
        self.store.save_calibration("subject-1", profile_to_dict(ready_profile()))
        engine = GatewayEngine(self.store, model=FixedModel(0.4))

        first = engine.ingest(packet(1, when=NOW))
        second = engine.ingest(packet(2, when=NOW + timedelta(minutes=5)))

        self.assertEqual("normal", first.decision)
        self.assertEqual("anomaly", second.decision)
        self.assertIn("gateway_autoencoder", second.triggered_by)

    def test_severe_model_error_is_immediate(self) -> None:
        self.store.save_calibration("subject-1", profile_to_dict(ready_profile()))

        decision = GatewayEngine(self.store, model=FixedModel(0.9, severe=True)).ingest(
            packet()
        )

        self.assertEqual("anomaly", decision.decision)
        self.assertIn("gateway_autoencoder", decision.triggered_by)

    def test_output_contains_only_compact_prediction_fields(self) -> None:
        body = GatewayEngine(self.store).ingest(packet()).to_dict()

        self.assertEqual(
            {
                "decision",
                "triggered_by",
                "reason_codes",
                "scores",
                "contributing_signals",
                "measurements",
                "calibration",
                "timestamp",
            },
            set(body),
        )


if __name__ == "__main__":
    unittest.main()
