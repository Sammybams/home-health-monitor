from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sqlite3
from typing import Any

from .contracts import FeaturePacket, Quality, WearableAssessment


SCHEMA = """
CREATE TABLE IF NOT EXISTS packets (
    id INTEGER PRIMARY KEY,
    schema_version INTEGER NOT NULL,
    subject_id TEXT NOT NULL,
    device_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    sample_duration_seconds REAL NOT NULL,
    firmware_version TEXT NOT NULL,
    sensor_config_id TEXT NOT NULL,
    feature_manifest_id TEXT NOT NULL,
    heart_rate_bpm REAL NOT NULL,
    spo2_percent REAL NOT NULL,
    temperature_c REAL NOT NULL,
    temperature_type TEXT NOT NULL,
    temperature_site TEXT NOT NULL,
    motion_intensity REAL NOT NULL,
    motion INTEGER NOT NULL,
    quality_ppg REAL NOT NULL,
    quality_spo2 REAL NOT NULL,
    quality_temperature REAL NOT NULL,
    quality_motion REAL NOT NULL,
    wearable_decision TEXT NOT NULL,
    wearable_score REAL NOT NULL,
    wearable_deviations TEXT NOT NULL,
    wearable_reason_codes TEXT NOT NULL,
    UNIQUE(device_id, sequence)
);
CREATE INDEX IF NOT EXISTS packets_subject_timestamp
    ON packets(subject_id, timestamp);

CREATE TABLE IF NOT EXISTS calibrations (
    subject_id TEXT PRIMARY KEY,
    profile TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,
    subject_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_subject_timestamp
    ON events(subject_id, timestamp DESC);
"""


class GatewayStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.executescript(SCHEMA)

    def __enter__(self) -> "GatewayStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()

    def add_packet(self, packet: FeaturePacket) -> bool:
        values = (
            packet.schema_version,
            packet.subject_id,
            packet.device_id,
            packet.sequence,
            packet.timestamp.isoformat(),
            packet.sample_duration_seconds,
            packet.firmware_version,
            packet.sensor_config_id,
            packet.feature_manifest_id,
            packet.heart_rate_bpm,
            packet.spo2_percent,
            packet.temperature_c,
            packet.temperature_type,
            packet.temperature_site,
            packet.motion_intensity,
            packet.motion,
            packet.quality.ppg,
            packet.quality.spo2,
            packet.quality.temperature,
            packet.quality.motion,
            packet.wearable.decision,
            packet.wearable.score,
            json.dumps(packet.wearable.deviations, separators=(",", ":"), allow_nan=False),
            json.dumps(packet.wearable.reason_codes, separators=(",", ":")),
        )
        try:
            with self.connection:
                self.connection.execute(
                    """
                    INSERT INTO packets (
                        schema_version, subject_id, device_id, sequence, timestamp,
                        sample_duration_seconds, firmware_version, sensor_config_id,
                        feature_manifest_id, heart_rate_bpm, spo2_percent, temperature_c,
                        temperature_type, temperature_site, motion_intensity, motion,
                        quality_ppg, quality_spo2, quality_temperature, quality_motion,
                        wearable_decision, wearable_score, wearable_deviations,
                        wearable_reason_codes
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    values,
                )
        except sqlite3.IntegrityError:
            return False
        return True

    @staticmethod
    def _packet(row: sqlite3.Row) -> FeaturePacket:
        return FeaturePacket(
            schema_version=row["schema_version"],
            subject_id=row["subject_id"],
            device_id=row["device_id"],
            sequence=row["sequence"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            sample_duration_seconds=row["sample_duration_seconds"],
            firmware_version=row["firmware_version"],
            sensor_config_id=row["sensor_config_id"],
            feature_manifest_id=row["feature_manifest_id"],
            heart_rate_bpm=row["heart_rate_bpm"],
            spo2_percent=row["spo2_percent"],
            temperature_c=row["temperature_c"],
            temperature_type=row["temperature_type"],
            temperature_site=row["temperature_site"],
            motion_intensity=row["motion_intensity"],
            motion=row["motion"],
            quality=Quality(
                ppg=row["quality_ppg"],
                spo2=row["quality_spo2"],
                temperature=row["quality_temperature"],
                motion=row["quality_motion"],
            ),
            wearable=WearableAssessment(
                decision=row["wearable_decision"],
                score=row["wearable_score"],
                deviations=json.loads(row["wearable_deviations"]),
                reason_codes=tuple(json.loads(row["wearable_reason_codes"])),
            ),
        )

    def packets_between(
        self,
        subject_id: str,
        start: datetime,
        end: datetime,
    ) -> tuple[FeaturePacket, ...]:
        rows = self.connection.execute(
            """
            SELECT * FROM packets
            WHERE subject_id = ? AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp ASC, sequence ASC
            """,
            (subject_id, start.isoformat(), end.isoformat()),
        ).fetchall()
        return tuple(self._packet(row) for row in rows)

    def latest_packet(self, subject_id: str) -> FeaturePacket | None:
        row = self.connection.execute(
            """
            SELECT * FROM packets WHERE subject_id = ?
            ORDER BY timestamp DESC, sequence DESC LIMIT 1
            """,
            (subject_id,),
        ).fetchone()
        return None if row is None else self._packet(row)

    def save_calibration(self, subject_id: str, profile: dict[str, Any]) -> None:
        encoded = json.dumps(profile, separators=(",", ":"), sort_keys=True, allow_nan=False)
        updated_at = datetime.now().astimezone().isoformat()
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO calibrations (subject_id, profile, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(subject_id) DO UPDATE SET
                    profile = excluded.profile,
                    updated_at = excluded.updated_at
                """,
                (subject_id, encoded, updated_at),
            )

    def load_calibration(self, subject_id: str) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT profile FROM calibrations WHERE subject_id = ?",
            (subject_id,),
        ).fetchone()
        return None if row is None else json.loads(row["profile"])

    def save_event(self, subject_id: str, event: dict[str, Any], timestamp: datetime) -> int:
        encoded = json.dumps(event, separators=(",", ":"), sort_keys=True, allow_nan=False)
        with self.connection:
            cursor = self.connection.execute(
                "INSERT INTO events (subject_id, timestamp, payload) VALUES (?, ?, ?)",
                (subject_id, timestamp.isoformat(), encoded),
            )
        return int(cursor.lastrowid)

    def recent_events(self, subject_id: str, *, limit: int = 100) -> tuple[dict[str, Any], ...]:
        if not 1 <= limit <= 1_000:
            raise ValueError("limit must be between 1 and 1000")
        rows = self.connection.execute(
            """
            SELECT payload FROM events WHERE subject_id = ?
            ORDER BY timestamp DESC, id DESC LIMIT ?
            """,
            (subject_id, limit),
        ).fetchall()
        return tuple(json.loads(row["payload"]) for row in rows)

    def delete_packets_before(self, cutoff: datetime) -> int:
        with self.connection:
            cursor = self.connection.execute(
                "DELETE FROM packets WHERE timestamp < ?",
                (cutoff.isoformat(),),
            )
        return cursor.rowcount
