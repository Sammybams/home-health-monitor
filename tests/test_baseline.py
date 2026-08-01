from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest

from home_health_monitor.baseline import baseline_to_dict, build_baseline
from home_health_monitor.change import assess_change
from home_health_monitor.contracts import InputError, parse_request


def daily_payload(day: int, *, body_offset: float = 0.0) -> dict:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=day)
    return {
        "subject_id": "person-1",
        "observations": [
            {
                "timestamp": (start + timedelta(minutes=15 * index)).isoformat(),
                "body_temperature_c": 36.5 + body_offset + (index % 3) * 0.01,
                "ambient_temperature_c": 25.0 + (index % 4) * 0.05,
                "heart_rate_bpm": 68 + index % 5,
                "motion": index % 2,
                "resting": index % 2 == 0,
            }
            for index in range(97)
        ],
    }


def profile_payload() -> dict:
    requests = [parse_request(daily_payload(day)) for day in range(7)]
    profile = build_baseline(
        requests,
        created_at=datetime(2026, 1, 10, tzinfo=timezone.utc),
    )
    return baseline_to_dict(profile)


class BaselineTests(unittest.TestCase):
    def test_builds_and_round_trips_profile(self) -> None:
        profile = profile_payload()
        current = daily_payload(8)
        current["baseline"] = profile
        request = parse_request(current)
        self.assertEqual(7, request.baseline.healthy_days)  # type: ignore[union-attr]
        self.assertEqual(679, request.baseline.sample_count)  # type: ignore[union-attr]

    def test_rejects_mismatched_subject(self) -> None:
        current = daily_payload(8)
        current["subject_id"] = "somebody-else"
        current["baseline"] = profile_payload()
        with self.assertRaisesRegex(InputError, "does not match"):
            parse_request(current)

    def test_requires_seven_healthy_days(self) -> None:
        with self.assertRaisesRegex(InputError, "between 7 and 30"):
            build_baseline(parse_request(daily_payload(day)) for day in range(6))

    def test_assessment_recognizes_normal_window(self) -> None:
        current = daily_payload(8)
        current["baseline"] = profile_payload()
        result, warnings = assess_change(parse_request(current))
        self.assertEqual("within_personal_baseline", result["status"])
        self.assertEqual([], warnings)

    def test_assessment_recognizes_large_temperature_change(self) -> None:
        current = daily_payload(8, body_offset=1.0)
        current["baseline"] = profile_payload()
        result, _ = assess_change(parse_request(current))
        self.assertEqual("unusual_change", result["status"])
        self.assertGreater(result["score"], result["threshold"])

    def test_assessment_requires_profile(self) -> None:
        result, warnings = assess_change(parse_request(daily_payload(8)))
        self.assertEqual("insufficient_data", result["status"])
        self.assertIn("personal_baseline_missing", warnings)
