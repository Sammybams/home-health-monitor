from __future__ import annotations

import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest

from home_health_monitor.gateway.server import create_server
from tests.test_gateway_contracts import valid_packet


def request_json(server, method: str, path: str, payload: dict | None = None):
    connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=2)
    body = None if payload is None else json.dumps(payload)
    headers = {} if payload is None else {"Content-Type": "application/json"}
    connection.request(method, path, body, headers)
    response = connection.getresponse()
    result = json.loads(response.read())
    connection.close()
    return response.status, result


class GatewayServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        directory = Path(self.temporary.name)
        self.server = create_server(
            "127.0.0.1",
            0,
            database_path=directory / "gateway.db",
            model_path=directory / "missing.tflite",
            metadata_path=directory / "missing-metadata.json",
            max_body_bytes=64 * 1024,
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temporary.cleanup()

    def test_packet_endpoint_returns_binary_prediction(self) -> None:
        status, body = request_json(self.server, "POST", "/v2/packets", valid_packet())

        self.assertEqual(200, status)
        self.assertIn(body["decision"], {"normal", "anomaly"})
        self.assertNotIn("disclaimer", body)

    def test_health_reports_gateway_components(self) -> None:
        status, body = request_json(self.server, "GET", "/health")

        self.assertEqual(200, status)
        self.assertEqual("ready", body["status"])
        self.assertFalse(body["model_loaded"])
        self.assertEqual("ready", body["database"])

    def test_latest_prediction_and_calibration_are_available(self) -> None:
        request_json(self.server, "POST", "/v2/packets", valid_packet())

        status, prediction = request_json(
            self.server, "GET", "/v2/prediction?subject_id=subject-1"
        )
        calibration_status, calibration = request_json(
            self.server, "GET", "/v2/calibration?subject_id=subject-1"
        )

        self.assertEqual(200, status)
        self.assertEqual("normal", prediction["decision"])
        self.assertEqual(200, calibration_status)
        self.assertEqual("collecting", calibration["status"])

    def test_unknown_packet_fields_are_rejected(self) -> None:
        payload = valid_packet()
        payload["sms"] = True

        status, body = request_json(self.server, "POST", "/v2/packets", payload)

        self.assertEqual(422, status)
        self.assertEqual("invalid_input", body["error"])

    def test_prediction_requires_subject_id(self) -> None:
        status, body = request_json(self.server, "GET", "/v2/prediction")

        self.assertEqual(400, status)
        self.assertEqual("subject_id_required", body["error"])


if __name__ == "__main__":
    unittest.main()
