from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest

from home_health_monitor.contracts import InputError
from home_health_monitor.gateway.calibration import CalibrationProfile, CalibrationSignal
from home_health_monitor.gateway.contracts import parse_packet
from home_health_monitor.gateway.windowing import FEATURE_NAMES, build_window
from tests.test_gateway_contracts import valid_packet


START = datetime(2026, 9, 1, tzinfo=timezone.utc)
END = START + timedelta(hours=23, minutes=55)


def profile() -> CalibrationProfile:
    return CalibrationProfile(
        schema_version=1,
        subject_id="subject-1",
        feature_manifest_id="features-v1",
        sensor_config_id="sensor-set-a",
        temperature_type="skin",
        temperature_site="wrist",
        started_at=START - timedelta(days=2),
        ready_at=START,
        expected_interval_seconds=60.0,
        packet_count=2880,
        coverage=1.0,
        low_motion_hours=12.0,
        signals={
            "heart_rate_bpm": CalibrationSignal(70.0, 3.0),
            "spo2_percent": CalibrationSignal(97.0, 0.5),
            "temperature_c": CalibrationSignal(33.5, 0.1),
            "motion_intensity": CalibrationSignal(0.1, 0.05),
        },
    )


def packet_at(when: datetime, sequence: int, **changes):
    payload = valid_packet()
    payload["sequence"] = sequence
    payload["timestamp"] = when.isoformat()
    payload.update(changes)
    return parse_packet(payload)


class GatewayWindowingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.packets = [
            packet_at(START + timedelta(minutes=5 * index), index)
            for index in range(288)
        ]

    def test_window_has_288_five_minute_steps(self) -> None:
        window = build_window(self.packets, profile(), END)

        self.assertEqual(288, len(window.values))
        self.assertEqual(288, len(window.masks))
        self.assertEqual(len(FEATURE_NAMES), len(window.values[0]))
        self.assertEqual(START, window.started_at)
        self.assertEqual(END + timedelta(minutes=5), window.ended_at)

    def test_missing_bins_have_zero_values_and_zero_mask(self) -> None:
        packets = [packet for packet in self.packets if packet.sequence != 36]

        window = build_window(packets, profile(), END)

        self.assertTrue(all(value == 0.0 for value in window.values[36]))
        self.assertTrue(all(mask == 0.0 for mask in window.masks[36]))

    def test_values_are_median_aggregated_and_robustly_normalized(self) -> None:
        packets = self.packets + [
            packet_at(
                START + timedelta(minutes=1),
                500,
                heart_rate_bpm=76.0,
                spo2_percent=96.0,
            ),
            packet_at(
                START + timedelta(minutes=2),
                501,
                heart_rate_bpm=73.0,
                spo2_percent=97.0,
            ),
        ]

        window = build_window(packets, profile(), END)

        self.assertAlmostEqual(1.0, window.values[0][FEATURE_NAMES.index("heart_rate_bpm")])
        self.assertAlmostEqual(0.0, window.values[0][FEATURE_NAMES.index("spo2_percent")])

    def test_quality_channels_are_part_of_each_observed_step(self) -> None:
        window = build_window(self.packets, profile(), END)

        ppg_index = FEATURE_NAMES.index("quality_ppg")
        self.assertAlmostEqual(0.96, window.values[0][ppg_index])
        self.assertEqual(1.0, window.masks[0][ppg_index])

    def test_low_signal_quality_masks_only_that_measurement(self) -> None:
        payload = valid_packet()
        payload["sequence"] = 700
        payload["timestamp"] = START.isoformat()
        payload["quality"]["ppg"] = 0.2

        window = build_window([parse_packet(payload)], profile(), END)

        heart_rate_index = FEATURE_NAMES.index("heart_rate_bpm")
        quality_index = FEATURE_NAMES.index("quality_ppg")
        self.assertEqual(0.0, window.masks[0][heart_rate_index])
        self.assertEqual(1.0, window.masks[0][quality_index])
        self.assertAlmostEqual(0.2, window.values[0][quality_index])

    def test_packets_outside_window_are_ignored(self) -> None:
        outside = packet_at(START - timedelta(seconds=1), 800, heart_rate_bpm=200.0)

        window = build_window([outside] + self.packets, profile(), END)

        self.assertAlmostEqual(0.5, window.values[0][0])

    def test_profile_and_packets_must_match(self) -> None:
        mismatch = packet_at(START, 900)
        payload = valid_packet()
        payload["subject_id"] = "subject-2"
        payload["sequence"] = 901
        payload["timestamp"] = (START + timedelta(minutes=1)).isoformat()

        with self.assertRaisesRegex(InputError, "same subject"):
            build_window([mismatch, parse_packet(payload)], profile(), END)


if __name__ == "__main__":
    unittest.main()
