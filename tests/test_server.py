from __future__ import annotations

import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest

from home_health_monitor.features import feature_names
from home_health_monitor.server import create_server
from tests.test_contracts import payload


def artifact() -> dict:
    names = feature_names()
    head = {"coefficients": [0] * len(names), "intercept": 0, "threshold": 0.5}
    return {
        "schema_version": 1,
        "model_id": "server-test",
        "trained_at": "2026-01-01T00:00:00Z",
        "future_horizon_hours": 24,
        "features": list(names),
        "scaler": {"mean": [0] * len(names), "scale": [1] * len(names)},
        "heads": {"current": head, "future": head},
    }


class ServerTests(unittest.TestCase):
    def test_health_and_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            path.write_text(json.dumps(artifact()), encoding="utf-8")
            server = create_server("127.0.0.1", 0, str(path), 1024 * 1024)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=2)
                connection.request("GET", "/health")
                response = connection.getresponse()
                self.assertEqual(200, response.status)
                self.assertEqual("ready", json.loads(response.read())["status"])

                body = json.dumps(payload())
                connection.request("POST", "/v1/predict", body, {"Content-Type": "application/json"})
                response = connection.getresponse()
                result = json.loads(response.read())
                self.assertEqual(200, response.status)
                self.assertEqual("server-test", result["model"]["id"])
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_missing_model_still_supports_change_service(self) -> None:
        server = create_server("127.0.0.1", 0, "/definitely/missing/model.json", 1024 * 1024)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            self.assertIsNone(server.model)
            self.assertIn("could not load model", server.model_error)
            connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=2)
            connection.request("GET", "/health")
            response = connection.getresponse()
            health = json.loads(response.read())
            self.assertEqual(200, response.status)
            self.assertFalse(health["illness_model_loaded"])

            body = json.dumps(payload())
            connection.request("POST", "/v1/predict", body, {"Content-Type": "application/json"})
            response = connection.getresponse()
            result = json.loads(response.read())
            self.assertEqual(200, response.status)
            self.assertEqual("within_day_trend", result["prediction"]["method"])
            self.assertIn(result["prediction"]["current_risk"]["classification"], {"lower_risk", "higher_risk"})
            self.assertIn(result["prediction"]["future_risk"]["classification"], {"lower_risk", "higher_risk"})
            self.assertEqual("insufficient_data", result["change_assessment"]["status"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
