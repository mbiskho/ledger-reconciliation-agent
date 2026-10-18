"""Immutable audit log — append-only record of every state change."""

import json
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()


@dataclass
class AuditEntry:
    id: str
    timestamp: str
    event_type: str
    break_id: str
    actor: str
    action: str
    details: dict[str, Any]
    result: str
    integrity_hash: str


class AuditLog:
    def __init__(self, db_path: str = "audit.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                break_id TEXT NOT NULL,
                actor TEXT NOT NULL,
                action TEXT NOT NULL,
                details TEXT NOT NULL,
                result TEXT NOT NULL,
                integrity_hash TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS integrity_chain (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                previous_hash TEXT NOT NULL,
                entry_id TEXT NOT NULL,
                entry_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()

    def log(
        self,
        event_type: str,
        break_id: str,
        actor: str,
        action: str,
        details: dict[str, Any],
        result: str = "success",
    ) -> AuditEntry:
        import hashlib
        import uuid

        entry_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()

        prev_hash = self._get_latest_hash()

        raw = f"{entry_id}|{timestamp}|{event_type}|{break_id}|{actor}|{action}|{json.dumps(details, sort_keys=True)}|{result}|{prev_hash}"
        integrity_hash = hashlib.sha256(raw.encode()).hexdigest()

        entry = AuditEntry(
            id=entry_id, timestamp=timestamp, event_type=event_type,
            break_id=break_id, actor=actor, action=action,
            details=details, result=result, integrity_hash=integrity_hash,
        )

        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO audit_log VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (entry.id, entry.timestamp, entry.event_type, entry.break_id,
             entry.actor, entry.action, json.dumps(entry.details),
             entry.result, entry.integrity_hash),
        )
        conn.execute(
            "INSERT INTO integrity_chain (previous_hash, entry_id, entry_hash, created_at) VALUES (?, ?, ?, ?)",
            (prev_hash, entry.id, integrity_hash, timestamp),
        )
        conn.commit()
        conn.close()

        logger.debug("audit entry logged", event_type=event_type, break_id=break_id, entry_id=entry.id)
        return entry

    def _get_latest_hash(self) -> str:
        conn = sqlite3.connect(self.db_path)
        row = conn.execute("SELECT entry_hash FROM integrity_chain ORDER BY id DESC LIMIT 1").fetchone()
        conn.close()
        return row[0] if row else "genesis"

    def get_entries_for_break(self, break_id: str) -> list[AuditEntry]:
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT * FROM audit_log WHERE break_id = ? ORDER BY timestamp",
            (break_id,),
        ).fetchall()
        conn.close()

        return [
            AuditEntry(
                id=r[0], timestamp=r[1], event_type=r[2], break_id=r[3],
                actor=r[4], action=r[5], details=json.loads(r[6]),
                result=r[7], integrity_hash=r[8],
            )
            for r in rows
        ]

    def get_all_entries(self, limit: int = 100) -> list[AuditEntry]:
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        conn.close()

        return [
            AuditEntry(
                id=r[0], timestamp=r[1], event_type=r[2], break_id=r[3],
                actor=r[4], action=r[5], details=json.loads(r[6]),
                result=r[7], integrity_hash=r[8],
            )
            for r in rows
        ]

    def verify_integrity(self) -> tuple[bool, list[str]]:
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT entry_id, entry_hash, previous_hash FROM integrity_chain ORDER BY id"
        ).fetchall()
        conn.close()

        import hashlib

        issues = []
        prev_expected = "genesis"

        for entry_id, entry_hash, prev_hash in rows:
            if prev_hash != prev_expected:
                issues.append(f"Chain break at {entry_id}: expected {prev_expected}, got {prev_hash}")
            prev_expected = entry_hash

        return len(issues) == 0, issues

    def export_to_json(self, path: Path) -> None:
        entries = self.get_all_entries(limit=100000)
        with open(path, "w") as f:
            json.dump([asdict(e) for e in entries], f, indent=2, default=str)
        logger.info("audit log exported", path=str(path), entries=len(entries))
