from __future__ import annotations

import unittest

from home_health_monitor.contracts import InputError
from home_health_monitor.datasets.pulse_transit_build import FEATURE_MANIFEST_ID
from home_health_monitor.datasets.pulse_transit_features import DATASET_FEATURE_NAMES
from home_health_monitor.gateway.vector_contracts import parse_vector_interval


def valid_interval() -> dict:
    vector = {name: 0.1 for name in DATASET_FEATURE_NAMES}
    vector["heart_rate_valid"] = 1
    return {
        "schema_version": 1,
        "subject_id": "subject-1",
        "device_id": "wearable-1",
        "sequence": 1,
        "interval_start": "2026-09-24T08:00:00+01:00",
        "interval_end": "2026-09-24T08:08:00+01:00",
        "feature_manifest_id": FEATURE_MANIFEST_ID,
        "vectors": [vector] * 16,
    }


class VectorContractTests(unittest.TestCase):
    def test_accepts_exact_versioned_contract(self) -> None:
        interval = parse_vector_interval(valid_interval())
        self.assertEqual(16, len(interval.vectors))

    def test_rejects_unknown_fields(self) -> None:
        payload = valid_interval()
        payload["sms"] = True
        with self.assertRaises(InputError):
            parse_vector_interval(payload)

    def test_rejects_wrong_feature_manifest(self) -> None:
        payload = valid_interval()
        payload["vectors"][0] = {"heart_rate_bpm": 70}
        with self.assertRaisesRegex(InputError, "feature manifest"):
            parse_vector_interval(payload)


if __name__ == "__main__":
    unittest.main()
