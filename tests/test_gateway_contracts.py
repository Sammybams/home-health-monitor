from __future__ import annotations

import unittest

from home_health_monitor.contracts import InputError
from home_health_monitor.gateway.contracts import parse_packet


def valid_packet() -> dict:
    return {
        "schema_version": 1,
        "subject_id": "subject-1",
        "device_id": "wearable-1",
        "sequence": 42,
        "timestamp": "2026-09-07T20:00:00Z",
        "sample_duration_seconds": 3.0,
        "firmware_version": "1.0.0",
        "sensor_config_id": "sensor-set-a",
        "feature_manifest_id": "features-v1",
        "heart_rate_bpm": 71.5,
        "spo2_percent": 97.2,
        "temperature_c": 33.8,
        "temperature_type": "skin",
        "temperature_site": "wrist",
        "motion_intensity": 0.14,
        "motion": 0,
        "quality": {
            "ppg": 0.96,
            "spo2": 0.94,
            "temperature": 1.0,
            "motion": 0.99,
        },
        "wearable": {
            "decision": "normal",
            "score": 0.42,
            "deviations": {
                "heart_rate_bpm": 0.42,
                "spo2_percent": -0.2,
                "temperature_c": 0.1,
                "motion_intensity": 0.05,
            },
            "reason_codes": [],
        },
    }


class GatewayContractTests(unittest.TestCase):
    def test_parse_packet_accepts_complete_v1_packet(self) -> None:
        packet = parse_packet(valid_packet())

        self.assertEqual("subject-1", packet.subject_id)
        self.assertEqual(97.2, packet.spo2_percent)
        self.assertEqual("normal", packet.wearable.decision)
        self.assertEqual("2026-09-07T20:00:00+00:00", packet.timestamp.isoformat())

    def test_parse_packet_rejects_unknown_fields(self) -> None:
        payload = valid_packet()
        payload["sms"] = True

        with self.assertRaisesRegex(InputError, "unknown packet fields: sms"):
            parse_packet(payload)

    def test_parse_packet_rejects_missing_fields(self) -> None:
        payload = valid_packet()
        del payload["spo2_percent"]

        with self.assertRaisesRegex(InputError, "missing packet fields: spo2_percent"):
            parse_packet(payload)

    def test_parse_packet_rejects_timestamp_without_timezone(self) -> None:
        payload = valid_packet()
        payload["timestamp"] = "2026-09-07T20:00:00"

        with self.assertRaisesRegex(InputError, "timestamp must include a timezone"):
            parse_packet(payload)

    def test_parse_packet_rejects_invalid_quality(self) -> None:
        payload = valid_packet()
        payload["quality"]["ppg"] = 1.01

        with self.assertRaisesRegex(InputError, "quality.ppg must be between 0 and 1"):
            parse_packet(payload)

    def test_parse_packet_requires_anomaly_reason(self) -> None:
        payload = valid_packet()
        payload["wearable"]["decision"] = "anomaly"

        with self.assertRaisesRegex(InputError, "anomaly decision requires a reason code"):
            parse_packet(payload)

    def test_parse_packet_rejects_non_finite_deviation(self) -> None:
        payload = valid_packet()
        payload["wearable"]["deviations"]["spo2_percent"] = float("nan")

        with self.assertRaisesRegex(InputError, "wearable deviation spo2_percent must be finite"):
            parse_packet(payload)


if __name__ == "__main__":
    unittest.main()
