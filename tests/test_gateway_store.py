from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest

from home_health_monitor.gateway.contracts import parse_packet
from home_health_monitor.gateway.store import GatewayStore
from tests.test_gateway_contracts import valid_packet


START = datetime(2026, 9, 7, 20, 0, tzinfo=timezone.utc)
END = START + timedelta(hours=1)


def packet(*, sequence: int, minutes: int = 0):
    payload = valid_packet()
    payload["sequence"] = sequence
    payload["timestamp"] = (START + timedelta(minutes=minutes)).isoformat()
    return parse_packet(payload)


class GatewayStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "gateway.sqlite3"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_duplicate_device_sequence_is_idempotent(self) -> None:
        with GatewayStore(self.path) as store:
            self.assertTrue(store.add_packet(packet(sequence=4)))
            self.assertFalse(store.add_packet(packet(sequence=4)))

            saved = store.packets_between("subject-1", START, END)

        self.assertEqual(1, len(saved))

    def test_packets_are_returned_in_timestamp_order(self) -> None:
        with GatewayStore(self.path) as store:
            store.add_packet(packet(sequence=2, minutes=2))
            store.add_packet(packet(sequence=1, minutes=1))

            saved = store.packets_between("subject-1", START, END)

        self.assertEqual([1, 2], [item.sequence for item in saved])

    def test_state_survives_reopen(self) -> None:
        with GatewayStore(self.path) as store:
            store.add_packet(packet(sequence=1))

        with GatewayStore(self.path) as reopened:
            latest = reopened.latest_packet("subject-1")

        self.assertIsNotNone(latest)
        self.assertEqual(1, latest.sequence)
        self.assertEqual(97.2, latest.spo2_percent)

    def test_calibration_round_trips(self) -> None:
        profile = {"schema_version": 1, "status": "ready", "signals": {"spo2_percent": 97.1}}
        with GatewayStore(self.path) as store:
            store.save_calibration("subject-1", profile)

        with GatewayStore(self.path) as reopened:
            self.assertEqual(profile, reopened.load_calibration("subject-1"))

    def test_recent_events_are_newest_first(self) -> None:
        with GatewayStore(self.path) as store:
            store.save_event("subject-1", {"decision": "anomaly", "sequence": 1}, START)
            store.save_event(
                "subject-1",
                {"decision": "anomaly", "sequence": 2},
                START + timedelta(minutes=1),
            )

            events = store.recent_events("subject-1", limit=1)

        self.assertEqual([2], [event["sequence"] for event in events])

    def test_retention_removes_only_old_packets(self) -> None:
        with GatewayStore(self.path) as store:
            store.add_packet(packet(sequence=1, minutes=1))
            store.add_packet(packet(sequence=2, minutes=20))

            removed = store.delete_packets_before(START + timedelta(minutes=10))
            saved = store.packets_between("subject-1", START, END)

        self.assertEqual(1, removed)
        self.assertEqual([2], [item.sequence for item in saved])

    def test_event_retention_removes_only_old_events(self) -> None:
        old = datetime(2026, 1, 1, tzinfo=timezone.utc)
        recent = datetime(2026, 2, 1, tzinfo=timezone.utc)
        with GatewayStore(self.path) as store:
            store.save_event("subject-1", {"decision": "normal", "id": "old"}, old)
            store.save_event("subject-1", {"decision": "normal", "id": "recent"}, recent)

            removed = store.delete_events_before(datetime(2026, 1, 15, tzinfo=timezone.utc))
            events = store.recent_events("subject-1")

        self.assertEqual(1, removed)
        self.assertEqual(["recent"], [event["id"] for event in events])


if __name__ == "__main__":
    unittest.main()
