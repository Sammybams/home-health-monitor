from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest

from home_health_monitor.contracts import InputError, parse_request


def payload(count: int = 12) -> dict:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return {"observations": [
        {
            "timestamp": (start + timedelta(minutes=3 * index)).isoformat(),
            "body_temperature_c": 36.5 + index / 100,
            "ambient_temperature_c": 25.0,
            "heart_rate_bpm": 70 + index % 3,
            "motion": index % 2,
        }
        for index in range(count)
    ]}


class ContractTests(unittest.TestCase):
    def test_parses_valid_request(self) -> None:
        request = parse_request(payload())
        self.assertEqual(12, len(request.observations))

    def test_parses_optional_resting_flag(self) -> None:
        item = payload()
        item["observations"][0]["resting"] = True
        request = parse_request(item)
        self.assertTrue(request.observations[0].resting)

    def test_rejects_unsorted_timestamps(self) -> None:
        item = payload()
        item["observations"][2]["timestamp"] = item["observations"][1]["timestamp"]
        with self.assertRaisesRegex(InputError, "strictly increasing"):
            parse_request(item)

    def test_rejects_implausible_sensor_value(self) -> None:
        item = payload()
        item["observations"][0]["heart_rate_bpm"] = 900
        with self.assertRaisesRegex(InputError, "sensor range"):
            parse_request(item)
