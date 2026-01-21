"""
SQLite-backed conversation memory store.

Stores conversations, messages, and summaries for long-term recall.
"""

from __future__ import annotations

import os
import sqlite3
import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MemoryStore:
    """Lightweight SQLite storage for conversation memory."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        """Close the SQLite connection."""
        self.conn.close()

    def _init_schema(self) -> None:
        """Initialize tables and FTS indexes if available."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                is_persistent INTEGER NOT NULL DEFAULT 1
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(conversation_id) REFERENCES conversations(id)
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS summaries (
                conversation_id TEXT PRIMARY KEY,
                summary TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(conversation_id) REFERENCES conversations(id)
            )
            """
        )

        try:
            cursor.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts
                USING fts5(content, conversation_id UNINDEXED, content='messages', content_rowid='id')
                """
            )
            cursor.executescript(
                """
                CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
                    INSERT INTO messages_fts(rowid, content, conversation_id)
                    VALUES (new.id, new.content, new.conversation_id);
                END;
                CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
                    INSERT INTO messages_fts(messages_fts, rowid, content, conversation_id)
                    VALUES ('delete', old.id, old.content, old.conversation_id);
                END;
                CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
                    INSERT INTO messages_fts(messages_fts, rowid, content, conversation_id)
                    VALUES ('delete', old.id, old.content, old.conversation_id);
                    INSERT INTO messages_fts(rowid, content, conversation_id)
                    VALUES (new.id, new.content, new.conversation_id);
                END;
                """
            )
        except sqlite3.OperationalError:
            # FTS5 may be disabled in some builds. Fallback to LIKE search.
            pass

        self.conn.commit()

    def create_conversation(self, title: Optional[str] = None, is_persistent: bool = True) -> str:
        """Create a new conversation record and return its ID."""
        conversation_id = str(uuid.uuid4())
        now = _utc_now()
        self.conn.execute(
            """
            INSERT INTO conversations (id, title, created_at, updated_at, is_persistent)
            VALUES (?, ?, ?, ?, ?)
            """,
            (conversation_id, title or "", now, now, 1 if is_persistent else 0)
        )
        self.conn.commit()
        return conversation_id

    def update_title(self, conversation_id: str, title: str) -> None:
        """Update a conversation title."""
        self.conn.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
            (title, _utc_now(), conversation_id)
        )
        self.conn.commit()

    def get_title(self, conversation_id: str) -> Optional[str]:
        """Fetch the current title for a conversation."""
        row = self.conn.execute(
            "SELECT title FROM conversations WHERE id = ?",
            (conversation_id,)
        ).fetchone()
        return row[0] if row else None

    def touch_conversation(self, conversation_id: str) -> None:
        """Update the conversation updated_at timestamp."""
        self.conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (_utc_now(), conversation_id)
        )
        self.conn.commit()

    def list_conversations(self, search: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """List conversations, optionally filtered by a search term."""
        if search:
            search_like = f"%{search}%"
            try:
                rows = self.conn.execute(
                    """
                    SELECT c.id, c.title, c.created_at, c.updated_at
                    FROM conversations c
                    WHERE c.title LIKE ?
                       OR c.id IN (
                            SELECT conversation_id FROM messages_fts WHERE messages_fts MATCH ?
                       )
                    ORDER BY c.updated_at DESC
                    LIMIT ?
                    """,
                    (search_like, f"{search}*", limit)
                ).fetchall()
            except sqlite3.OperationalError:
                rows = self.conn.execute(
                    """
                    SELECT DISTINCT c.id, c.title, c.created_at, c.updated_at
                    FROM conversations c
                    LEFT JOIN messages m ON m.conversation_id = c.id
                    WHERE c.title LIKE ? OR m.content LIKE ?
                    ORDER BY c.updated_at DESC
                    LIMIT ?
                    """,
                    (search_like, search_like, limit)
                ).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT id, title, created_at, updated_at
                FROM conversations
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,)
            ).fetchall()

        return [dict(row) for row in rows]

    def get_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Return messages for a conversation in chronological order."""
        rows = self.conn.execute(
            """
            SELECT role, content, created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id ASC
            """,
            (conversation_id,)
        ).fetchall()
        return [dict(row) for row in rows]

    def add_message(self, conversation_id: str, role: str, content: str) -> None:
        """Insert a new message for a conversation."""
        now = _utc_now()
        self.conn.execute(
            """
            INSERT INTO messages (conversation_id, role, content, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (conversation_id, role, content, now)
        )
        self.conn.commit()
        self.touch_conversation(conversation_id)

    def get_summary(self, conversation_id: str) -> Optional[str]:
        """Return the stored summary for a conversation, if any."""
        row = self.conn.execute(
            "SELECT summary FROM summaries WHERE conversation_id = ?",
            (conversation_id,)
        ).fetchone()
        return row[0] if row else None

    def save_summary(self, conversation_id: str, summary: str) -> None:
        """Upsert a summary for a conversation."""
        now = _utc_now()
        self.conn.execute(
            """
            INSERT INTO summaries (conversation_id, summary, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(conversation_id) DO UPDATE SET
                summary = excluded.summary,
                updated_at = excluded.updated_at
            """,
            (conversation_id, summary, now)
        )
        self.conn.commit()

    def delete_conversation(self, conversation_id: str) -> None:
        """Delete a conversation and all related records."""
        self.conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        self.conn.execute("DELETE FROM summaries WHERE conversation_id = ?", (conversation_id,))
        self.conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        self.conn.commit()
