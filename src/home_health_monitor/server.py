from __future__ import annotations

from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

from .change import assess_change
from .contracts import InputError, parse_request
from .features import extract_features
from .model import ModelError, TinyModel

LOGGER = logging.getLogger("home_health_monitor")


class PredictionServer(HTTPServer):
    model: TinyModel | None
    model_error: str | None
    max_body_bytes: int


class Handler(BaseHTTPRequestHandler):
    server: PredictionServer

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/health":
            self._json(404, {"error": "not_found"})
            return
        self._json(200, {
            "status": "ready",
            "change_assessment_available": True,
            "illness_model_loaded": self.server.model is not None,
            "illness_model_reason": self.server.model_error,
        })

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/predict":
            self._json(404, {"error": "not_found"})
            return
        if self.headers.get_content_type() != "application/json":
            self._json(415, {"error": "content_type_must_be_application_json"})
            return
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self._json(400, {"error": "invalid_content_length"})
            return
        if length <= 0 or length > self.server.max_body_bytes:
            self._json(413, {"error": "request_body_size_out_of_range"})
            return
        try:
            payload = json.loads(self.rfile.read(length))
            request = parse_request(payload)
            features, warnings = extract_features(request)
            change_assessment, change_warnings = assess_change(request)
            warnings = sorted(set(warnings + change_warnings))
            if self.server.model is not None:
                response = self.server.model.predict(features, warnings)
                response["change_assessment"] = change_assessment
            else:
                response = {
                    "change_assessment": change_assessment,
                    "prediction": None,
                    "model": {
                        "status": "not_loaded",
                        "reason": self.server.model_error,
                    },
                    "data_quality": {"warnings": warnings},
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "disclaimer": "Change screening only; not a diagnosis or emergency service.",
                }
        except json.JSONDecodeError:
            self._json(400, {"error": "invalid_json"})
            return
        except InputError as exc:
            self._json(422, {"error": "invalid_input", "detail": str(exc)})
            return
        self._json(200, response)

    def log_message(self, format: str, *args: object) -> None:
        LOGGER.info("client=%s " + format, self.client_address[0], *args)


def create_server(host: str, port: int, model_path: str, max_body_bytes: int) -> PredictionServer:
    server = PredictionServer((host, port), Handler)
    server.max_body_bytes = max_body_bytes
    try:
        server.model = TinyModel.load(Path(model_path))
        server.model_error = None
    except ModelError as exc:
        server.model = None
        server.model_error = str(exc)
    return server


def run(host: str, port: int, model_path: str, max_body_bytes: int) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    server = create_server(host, port, model_path, max_body_bytes)
    LOGGER.info("listening on http://%s:%s model_loaded=%s", host, port, server.model is not None)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
