from __future__ import annotations

from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import logging
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from ..contracts import InputError
from .autoencoder import AutoencoderModel, ModelError
from .contracts import parse_packet
from .engine import GatewayEngine
from .store import GatewayStore


LOGGER = logging.getLogger("home_health_monitor.gateway")


class GatewayServer(HTTPServer):
    engine: GatewayEngine
    store: GatewayStore
    model_loaded: bool
    model_error: str | None
    max_body_bytes: int

    def server_close(self) -> None:
        try:
            if hasattr(self, "store"):
                self.store.close()
        finally:
            super().server_close()


class Handler(BaseHTTPRequestHandler):
    server: GatewayServer

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _subject_id(self, query: str) -> str | None:
        values = parse_qs(query, keep_blank_values=True).get("subject_id", [])
        if len(values) != 1 or not values[0] or len(values[0]) > 128:
            return None
        return values[0]

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlsplit(self.path)
        if parsed.path == "/health":
            self._json(
                200,
                {
                    "status": "ready",
                    "database": "ready",
                    "model_loaded": self.server.model_loaded,
                    "model_reason": self.server.model_error,
                },
            )
            return
        if parsed.path not in {"/v2/prediction", "/v2/calibration"}:
            self._json(404, {"error": "not_found"})
            return
        subject_id = self._subject_id(parsed.query)
        if subject_id is None:
            self._json(400, {"error": "subject_id_required"})
            return
        events = self.server.store.recent_events(subject_id, limit=1)
        if not events:
            self._json(404, {"error": "prediction_not_found"})
            return
        if parsed.path == "/v2/prediction":
            self._json(200, events[0])
        else:
            self._json(200, events[0]["calibration"])

    def do_POST(self) -> None:  # noqa: N802
        if urlsplit(self.path).path != "/v2/packets":
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
            packet = parse_packet(payload)
            result = self.server.engine.ingest(packet).to_dict()
        except json.JSONDecodeError:
            self._json(400, {"error": "invalid_json"})
            return
        except InputError as exc:
            self._json(422, {"error": "invalid_input", "detail": str(exc)})
            return
        self._json(200, result)

    def log_message(self, format: str, *args: object) -> None:
        LOGGER.info("client=%s " + format, self.client_address[0], *args)


def create_server(
    host: str,
    port: int,
    *,
    database_path: str | Path,
    model_path: str | Path,
    metadata_path: str | Path,
    max_body_bytes: int,
) -> GatewayServer:
    server = GatewayServer((host, port), Handler)
    server.max_body_bytes = max_body_bytes
    server.store = GatewayStore(database_path)
    try:
        model = AutoencoderModel.load(model_path, metadata_path)
        server.model_loaded = True
        server.model_error = None
    except ModelError as exc:
        model = None
        server.model_loaded = False
        server.model_error = str(exc)
    server.engine = GatewayEngine(server.store, model=model)
    return server


def run(
    host: str,
    port: int,
    *,
    database_path: str | Path,
    model_path: str | Path,
    metadata_path: str | Path,
    max_body_bytes: int,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    server = create_server(
        host,
        port,
        database_path=database_path,
        model_path=model_path,
        metadata_path=metadata_path,
        max_body_bytes=max_body_bytes,
    )
    LOGGER.info(
        "listening on http://%s:%s model_loaded=%s",
        host,
        port,
        server.model_loaded,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
