"""SQLite response cache for extractor and judge calls (raw response text, resumable)."""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from types import TracebackType
from typing import Any

from amazon_recsys.utils.hashing import hash_obj

_SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
    role        TEXT NOT NULL,
    review_id   TEXT NOT NULL,
    prompt_hash TEXT NOT NULL,
    model       TEXT NOT NULL,
    input_hash  TEXT NOT NULL,
    response    TEXT NOT NULL,
    created_at  REAL NOT NULL,
    PRIMARY KEY (role, review_id, prompt_hash, model, input_hash)
)
"""

Key = tuple[str, str, str, str, str]


def make_input_hash(*parts: Any) -> str:
    """Hash the inputs that determine a response. For the judge, include the extraction hash."""
    return hash_obj(list(parts))


class ResponseCache:
    """Persistent key -> raw response store. Safe for concurrent use from threads."""

    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(_SCHEMA)
            self._conn.commit()

    def get(
        self, role: str, review_id: str, prompt_hash: str, model: str, input_hash: str
    ) -> str | None:
        key: Key = (role, review_id, prompt_hash, model, input_hash)
        with self._lock:
            row = self._conn.execute(
                "SELECT response FROM responses WHERE role=? AND review_id=? AND prompt_hash=? "
                "AND model=? AND input_hash=?",
                key,
            ).fetchone()
        return None if row is None else row[0]

    def put(
        self,
        role: str,
        review_id: str,
        prompt_hash: str,
        model: str,
        input_hash: str,
        response: str,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO responses VALUES (?, ?, ?, ?, ?, ?, ?)",
                (role, review_id, prompt_hash, model, input_hash, response, time.time()),
            )
            self._conn.commit()

    def count(self, role: str | None = None) -> int:
        query, args = "SELECT COUNT(*) FROM responses", ()
        if role is not None:
            query, args = query + " WHERE role=?", (role,)
        with self._lock:
            return self._conn.execute(query, args).fetchone()[0]

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> ResponseCache:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
