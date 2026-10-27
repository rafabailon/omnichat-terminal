from __future__ import annotations

import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from models import ChatSession, MessageMatch, StoredMessage

SCHEMA_VERSION = 1


def _timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _like_pattern(query: str) -> str:
    """Build a literal, case-insensitive SQLite LIKE pattern."""
    escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


class ChatStore:
    """SQLite persistence for chat sessions and their complete message history."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser()
        if str(self.path) == ":memory:":
            raise ValueError(
                "OmniChat Terminal requires a persistent SQLite file path, not :memory:."
            )
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._initialize()
        if os.name == "posix":
            self.path.chmod(0o600)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version > SCHEMA_VERSION:
                raise RuntimeError(
                    "The chat database was created by a newer OmniChat Terminal version."
                )
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_messages_session_id_id
                    ON messages(session_id, id);
                CREATE INDEX IF NOT EXISTS idx_chat_sessions_updated_at
                    ON chat_sessions(updated_at DESC);
                """
            )
            if version < SCHEMA_VERSION:
                connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    @staticmethod
    def _session(row: sqlite3.Row) -> ChatSession:
        return ChatSession(
            id=row["id"],
            title=row["title"],
            provider=row["provider"],
            model=row["model"],
            created_at=_timestamp(row["created_at"]),
            updated_at=_timestamp(row["updated_at"]),
        )

    def create_session(self, provider: str, model: str) -> ChatSession:
        now = datetime.now(timezone.utc).isoformat()
        session_id = str(uuid.uuid4())
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO chat_sessions (id, title, provider, model, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, "New conversation", provider, model, now, now),
            )
            row = connection.execute(
                "SELECT * FROM chat_sessions WHERE id = ?", (session_id,)
            ).fetchone()
        assert row is not None
        return self._session(row)

    def add_turn(
        self,
        session_id: str,
        user_prompt: str,
        assistant_response: str,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            session = connection.execute(
                "SELECT title FROM chat_sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if session is None:
                raise KeyError(f"Chat session {session_id!r} does not exist.")
            title = session["title"]
            if title == "New conversation":
                title = " ".join(user_prompt.split())[:80] or "New conversation"
            connection.execute(
                """
                UPDATE chat_sessions SET title = ?, updated_at = ? WHERE id = ?
                """,
                (title, now, session_id),
            )
            connection.executemany(
                """
                INSERT INTO messages (session_id, role, content, created_at)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (session_id, "user", user_prompt, now),
                    (session_id, "assistant", assistant_response, now),
                ],
            )

    def list_sessions(self, limit: int = 100) -> list[ChatSession]:
        if limit < 1:
            raise ValueError("Session limit must be positive.")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM chat_sessions ORDER BY updated_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._session(row) for row in rows]

    def get_session(self, session_id: str) -> ChatSession | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM chat_sessions WHERE id = ?", (session_id,)
            ).fetchone()
        return self._session(row) if row else None

    def get_messages(self, session_id: str) -> list[StoredMessage]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY id", (session_id,)
            ).fetchall()
        return [
            StoredMessage(
                id=row["id"],
                session_id=row["session_id"],
                role=row["role"],
                content=row["content"],
                created_at=_timestamp(row["created_at"]),
            )
            for row in rows
        ]

    def search_messages(self, query: str, limit: int = 20) -> list[MessageMatch]:
        """Search message text across every saved conversation."""
        needle = query.strip()
        if not needle:
            raise ValueError("Search query cannot be blank.")
        if not 1 <= limit <= 500:
            raise ValueError("Search limit must be between 1 and 500.")
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    m.id AS message_id,
                    m.session_id,
                    s.title AS session_title,
                    s.provider,
                    m.role,
                    m.content,
                    m.created_at
                FROM messages AS m
                JOIN chat_sessions AS s ON s.id = m.session_id
                WHERE m.content LIKE ? ESCAPE '\\'
                ORDER BY m.id DESC
                LIMIT ?
                """,
                (_like_pattern(needle), limit),
            ).fetchall()
        return [
            MessageMatch(
                message_id=row["message_id"],
                session_id=row["session_id"],
                session_title=row["session_title"],
                provider=row["provider"],
                role=row["role"],
                content=row["content"],
                created_at=_timestamp(row["created_at"]),
            )
            for row in rows
        ]

    def delete_session(self, session_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
        return cursor.rowcount > 0
