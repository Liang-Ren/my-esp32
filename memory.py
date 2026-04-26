import json
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "memory.db"
SHORT_TERM_LIMIT = 10


class Memory:
    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self._init_db()

    def _init_db(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS messages (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id TEXT    NOT NULL,
                role      TEXT    NOT NULL,
                content   TEXT    NOT NULL,
                ts        DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS device_memory (
                device_id   TEXT PRIMARY KEY,
                preferences TEXT DEFAULT '{}',
                summary     TEXT DEFAULT '',
                facts       TEXT DEFAULT '[]',
                updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        """)
        self.conn.commit()

    # ── short-term ──────────────────────────────────────────────────────────

    def add_message(self, device_id: str, role: str, content: str):
        self.conn.execute(
            "INSERT INTO messages (device_id, role, content) VALUES (?, ?, ?)",
            (device_id, role, content),
        )
        self.conn.commit()

    def get_recent(self, device_id: str, limit: int = SHORT_TERM_LIMIT) -> list[dict]:
        rows = self.conn.execute(
            "SELECT role, content FROM messages "
            "WHERE device_id=? ORDER BY ts DESC LIMIT ?",
            (device_id, limit),
        ).fetchall()
        return [{"role": r[0], "content": r[1]} for r in reversed(rows)]

    # ── long-term ────────────────────────────────────────────────────────────

    def get_long_term(self, device_id: str) -> dict:
        row = self.conn.execute(
            "SELECT preferences, summary, facts FROM device_memory WHERE device_id=?",
            (device_id,),
        ).fetchone()
        if not row:
            return {"preferences": {}, "summary": "", "facts": []}
        return {
            "preferences": json.loads(row[0]),
            "summary": row[1],
            "facts": json.loads(row[2]),
        }

    def update_long_term(
        self,
        device_id: str,
        summary: str = None,
        facts: list = None,
        preferences: dict = None,
    ):
        existing = self.get_long_term(device_id)
        self.conn.execute(
            """
            INSERT INTO device_memory (device_id, preferences, summary, facts, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(device_id) DO UPDATE SET
                preferences = excluded.preferences,
                summary     = excluded.summary,
                facts       = excluded.facts,
                updated_at  = excluded.updated_at
            """,
            (
                device_id,
                json.dumps(preferences if preferences is not None else existing["preferences"]),
                summary if summary is not None else existing["summary"],
                json.dumps(facts if facts is not None else existing["facts"]),
            ),
        )
        self.conn.commit()