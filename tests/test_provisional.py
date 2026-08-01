from __future__ import annotations

import unittest

from home_health_monitor.contracts import parse_request
from home_health_monitor.provisional import provisional_prediction
from tests.test_baseline import daily_payload, profile_payload


class ProvisionalPredictionTests(unittest.TestCase):
    def test_first_day_always_returns_two_predictions(self) -> None:
        prediction, warnings = provisional_prediction(parse_request(daily_payload(8)))
        self.assertEqual("within_day_trend", prediction["method"])
        self.assertEqual("low", prediction["confidence"])
        self.assertIn(prediction["current_risk"]["classification"], {"lower_risk", "higher_risk"})
        self.assertIn(prediction["future_risk"]["classification"], {"lower_risk", "higher_risk"})
        self.assertEqual(24, prediction["future_risk"]["horizon_hours"])
        self.assertIn("personal_baseline_missing_used_within_day_reference", warnings)

    def test_personal_baseline_is_preferred(self) -> None:
        current = daily_payload(8)
        current["baseline"] = profile_payload()
        prediction, _ = provisional_prediction(parse_request(current))
        self.assertEqual("personal_baseline_trend", prediction["method"])
        self.assertEqual("moderate", prediction["confidence"])

    def test_large_recent_change_produces_higher_risk(self) -> None:
        current = daily_payload(8)
        for observation in current["observations"][-32:]:
            observation["body_temperature_c"] += 1.0
        prediction, _ = provisional_prediction(parse_request(current))
        self.assertEqual("higher_risk", prediction["current_risk"]["classification"])
        self.assertEqual("higher_risk", prediction["future_risk"]["classification"])
        self.assertEqual("uncalibrated_risk_score", prediction["future_risk"]["score_type"])
