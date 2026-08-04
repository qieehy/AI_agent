import sqlite3
import json
from pathlib import Path


class SessionStore:
    def __init__(self, db_path: str = "data/sessions.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row   #返回Row对象, 当dict用
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions
            (
                id
                TEXT
                PRIMARY
                KEY,
                system_message
                TEXT,
                messages
                TEXT,
                created_at
                TEXT
                DEFAULT CURRENT_TIMESTAMP,
                updated_at
                TEXT
                DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.commit()

    def save(self, session_id: str, system_message: dict | None, messages: list[dict]) -> None:
        self._conn.execute("INSERT OR REPLACE INTO sessions (id, system_message, messages, updated_at) VALUES (?, ?, ?, datetime('now'))",
                           (session_id,
                            json.dumps(system_message) if system_message is not None else None,
                            json.dumps(messages))  #list[dict] -> json
                           )
        self._conn.commit()

    def load(self, session_id: str):
        row = self._conn.execute("SELECT system_message, messages FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if row is None:
            return None
        return(
            json.loads(row["system_message"]) if row["system_message"] is not None else None,
            json.loads(row["messages"]),
        )

    def list_sessions(self):
        rows = self._conn.execute("SELECT id FROM sessions ORDER BY updated_at DESC").fetchall()
        return [row["id"] for row in rows]

    def delete(self, session_id):
        self._conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        self._conn.commit()