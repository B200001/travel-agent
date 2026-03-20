"""SQLite cache for repeated/similar travel queries."""

import difflib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple


class TravelQueryCache:
    """Simple SQLite cache for demo-friendly response reuse."""

    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS query_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                query_raw TEXT NOT NULL,
                query_norm TEXT NOT NULL,
                response_text TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_query_cache_session ON query_cache(session_id)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_query_cache_norm ON query_cache(query_norm)"
        )
        self._conn.commit()

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    def make_cache_key(self, session_id: str, user_query: str) -> str:
        return f"{session_id}:{self._normalize(user_query)}"

    def put(self, session_id: str, user_query: str, response_text: str) -> None:
        query_norm = self._normalize(user_query)
        timestamp = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            INSERT INTO query_cache (session_id, query_raw, query_norm, response_text, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, user_query, query_norm, response_text, timestamp),
        )
        self._conn.commit()

    def lookup(self, session_id: str, user_query: str, similarity_threshold: float = 0.92) -> Optional[Tuple[str, float]]:
        query_norm = self._normalize(user_query)

        exact = self._conn.execute(
            """
            SELECT response_text
            FROM query_cache
            WHERE session_id = ? AND query_norm = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (session_id, query_norm),
        ).fetchone()
        if exact:
            return str(exact["response_text"]), 1.0

        # Similar-query matching for demo convenience.
        rows = self._conn.execute(
            """
            SELECT query_norm, response_text
            FROM query_cache
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT 100
            """,
            (session_id,),
        ).fetchall()
        best_score = 0.0
        best_response = None
        for row in rows:
            score = difflib.SequenceMatcher(None, query_norm, str(row["query_norm"])).ratio()
            if score > best_score:
                best_score = score
                best_response = str(row["response_text"])
        if best_response and best_score >= similarity_threshold:
            return best_response, best_score
        return None

    def _normalize(self, text: str) -> str:
        lowered = (text or "").lower().strip()
        return " ".join(lowered.split())
