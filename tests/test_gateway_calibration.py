from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest

from home_health_monitor.contracts import InputError
from home_health_monitor.gateway.calibration import (
    build_calibration,
    profile_from_dict,
    profile_to_dict,
    robust_deviations,
)
from home_health_monitor.gateway.contracts import parse_packet
from tests.test_gateway_contracts import valid_packet


START = datetime(2026, 9, 1, tzinfo=timezone.utc)


def packets_for(
    *,
    hours: int,
    coverage: float,
    low_motion_hours: float | None = None,
    outlier: bool = False,
):
    total = hours * 60 + 1
    selected = max(2, round(total * coverage))
    indices = sorted({round(i * (total - 1) / (selected - 1)) for i in range(selected)})
    low_motion_packets = selected if low_motion_hours is None else round(low_motion_hours * 60)
    packets = []
    for position, minute in enumerate(indices):
        payload = valid_packet()
        payload["sequence"] = position
        payload["timestamp"] = (START + timedelta(minutes=minute)).isoformat()
        payload["heart_rate_bpm"] = 70.0
        payload["spo2_percent"] = 97.0
        payload["temperature_c"] = 33.5
        payload["motion"] = 0 if position < low_motion_packets else 1
        payload["motion_intensity"] = 0.1 if payload["motion"] == 0 else 2.0
        packets.append(parse_packet(payload))
    if outlier:
        payload = valid_packet()
        payload["sequence"] = len(packets)
        payload["timestamp"] = (START + timedelta(hours=hours, seconds=1)).isoformat()
        payload["heart_rate_bpm"] = 200.0
        payload["spo2_percent"] = 50.0
        payload["temperature_c"] = 44.0
        payload["motion_intensity"] = 100.0
        packets.append(parse_packet(payload))
    return packets


class GatewayCalibrationTests(unittest.TestCase):
    def test_calibration_requires_48_hours(self) -> None:
        result = build_calibration(packets_for(hours=47, coverage=1.0))

        self.assertEqual("collecting", result.status)
        self.assertIn("need_48_elapsed_hours", result.reason_codes)

    def test_calibration_requires_80_percent_coverage(self) -> None:
        result = build_calibration(packets_for(hours=48, coverage=0.79))

        self.assertEqual("collecting", result.status)
        self.assertIn("need_80_percent_valid_coverage", result.reason_codes)

    def test_calibration_requires_eight_low_motion_hours(self) -> None:
        result = build_calibration(
            packets_for(hours=48, coverage=0.9, low_motion_hours=7.9)
        )

        self.assertEqual("collecting", result.status)
        self.assertIn("need_eight_low_motion_hours", result.reason_codes)

    def test_calibration_uses_median_and_mad(self) -> None:
        result = build_calibration(
            packets_for(hours=49, coverage=0.9, outlier=True)
        )

        self.assertEqual("ready", result.status)
        self.assertIsNotNone(result.profile)
        self.assertAlmostEqual(97.0, result.profile.signals["spo2_percent"].median)
        self.assertEqual(0.5, result.profile.signals["spo2_percent"].mad_scale)

    def test_low_quality_packets_do_not_count_towards_coverage(self) -> None:
        packets = packets_for(hours=48, coverage=0.81)
        low_quality = []
        for packet in packets[:100]:
            payload = valid_packet()
            payload["sequence"] = packet.sequence
            payload["timestamp"] = packet.timestamp.isoformat()
            payload["quality"]["ppg"] = 0.2
            low_quality.append(parse_packet(payload))
        result = build_calibration(low_quality + packets[100:])

        self.assertEqual("collecting", result.status)
        self.assertIn("need_80_percent_valid_coverage", result.reason_codes)

    def test_robust_deviations_use_profile_scale(self) -> None:
        ready = build_calibration(packets_for(hours=49, coverage=0.9))
        payload = valid_packet()
        payload["heart_rate_bpm"] = 76.0
        current = parse_packet(payload)

        deviations = robust_deviations(current, ready.profile)

        self.assertAlmostEqual(2.0, deviations["heart_rate_bpm"])

    def test_calibration_rejects_mixed_subjects(self) -> None:
        packets = packets_for(hours=49, coverage=0.9)
        payload = valid_packet()
        payload["subject_id"] = "subject-2"
        payload["sequence"] = len(packets)
        payload["timestamp"] = (START + timedelta(hours=49, seconds=1)).isoformat()
        packets.append(parse_packet(payload))

        with self.assertRaisesRegex(InputError, "same subject"):
            build_calibration(packets)

    def test_ready_profile_round_trips_through_json_shape(self) -> None:
        ready = build_calibration(packets_for(hours=49, coverage=0.9))

        restored = profile_from_dict(profile_to_dict(ready.profile))

        self.assertEqual(ready.profile, restored)


if __name__ == "__main__":
    unittest.main()
