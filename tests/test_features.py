from __future__ import annotations

import unittest

from home_health_monitor.contracts import parse_request
from home_health_monitor.features import extract_features, feature_names
from tests.test_contracts import payload


class FeatureTests(unittest.TestCase):
    def test_feature_contract_is_complete_and_finite(self) -> None:
        values, warnings = extract_features(parse_request(payload()))
        self.assertEqual(set(feature_names()), set(values))
        self.assertIn("history_shorter_than_recommended_20_hours", warnings)
        self.assertAlmostEqual(71.0, values["1h_heart_rate_bpm_mean"])

    def test_slope_uses_elapsed_time(self) -> None:
        values, _ = extract_features(parse_request(payload()))
        self.assertGreater(values["1h_body_temperature_c_slope_per_hour"], 0)
