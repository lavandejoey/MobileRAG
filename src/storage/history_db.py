#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chat history database using SQLite.
src/storage/history_db.py

@author: LIU Ziyi
@date: 2025-12-30
@license: Apache-2.0
"""
from __future__ import annotations

import re
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ChatRow:
    chat_id: str
    title: str
    created_at: float
    updated_at: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chat_id": self.chat_id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class MessageRow:
    msg_id: int
    chat_id: str
    role: str
    content: str
    turn_id: Optional[str]
    created_at: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "msg_id": self.msg_id,
            "chat_id": self.chat_id,
            "role": self.role,
            "content": self.content,
            "turn_id": self.turn_id,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class UploadedFileRow:
    upload_id: int
    chat_id: str
    original_name: str
    stored_name: str
    rel_path: str
    processed: int
    attached_msg_id: Optional[int]
    created_at: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "upload_id": self.upload_id,
            "chat_id": self.chat_id,
            "original_name": self.original_name,
            "stored_name": self.stored_name,
            "rel_path": self.rel_path,
            "processed": bool(self.processed),
            "attached_msg_id": self.attached_msg_id,
            "created_at": self.created_at,
        }


def _now() -> float:
    return time.time()


_TITLE_SPLIT_RE = re.compile(r"[。\.\!\?？！，,;；:\n]+")
_TITLE_SPACE_RE = re.compile(r"\s+")


def _title_from_first_user_text(text: str, max_len: int = 36) -> str:
    raw = (text or "").strip()
    if not raw:
        return "New chat"

    cleaned = _TITLE_SPACE_RE.sub(" ", raw)
    segments = [seg.strip(" -_[](){}\"'`") for seg in _TITLE_SPLIT_RE.split(cleaned)]
    segments = [seg for seg in segments if seg]

    title = segments[0] if segments else cleaned

    if len(title) > max_len:
        title = title[:max_len].rstrip(" -_.,;:!?！？，。")

    if not title:
        return "New chat"
    if len(title) == len(cleaned) or len(title) >= max_len:
        return title
    return title


DEFAULT_CHAT_TITLES = {"New chat", "Uploaded files"}


class HistoryDB:
    def __init__(self, db_path: str):
        p = Path(db_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = str(p)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # enable foreign key support immediately on this connection
        cur = self._conn.cursor()
        cur.execute("PRAGMA foreign_keys = ON;")
        self._conn.commit()
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS chats
                (
                    chat_id    TEXT PRIMARY KEY,
                    title      TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS messages
                (
                    msg_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id    TEXT NOT NULL,
                    role       TEXT NOT NULL,
                    content    TEXT NOT NULL,
                    turn_id    TEXT,
                    created_at REAL NOT NULL,
                    FOREIGN KEY (chat_id) REFERENCES chats (chat_id) ON DELETE CASCADE
                );
                """
            )
            cols = {
                str(row["name"])
                for row in cur.execute("PRAGMA table_info(messages)").fetchall()
            }
            if "turn_id" not in cols:
                cur.execute("ALTER TABLE messages ADD COLUMN turn_id TEXT")
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id, msg_id);"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_turn ON messages(chat_id, turn_id, msg_id);"
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS uploaded_files
                (
                    upload_id      INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id        TEXT NOT NULL,
                    original_name  TEXT NOT NULL,
                    stored_name    TEXT NOT NULL,
                    rel_path       TEXT NOT NULL,
                    processed      INTEGER NOT NULL DEFAULT 0,
                    attached_msg_id INTEGER,
                    created_at     REAL NOT NULL,
                    FOREIGN KEY (chat_id) REFERENCES chats (chat_id) ON DELETE CASCADE
                );
                """
            )
            cols = {
                str(row["name"])
                for row in cur.execute("PRAGMA table_info(uploaded_files)").fetchall()
            }
            if "attached_msg_id" not in cols:
                cur.execute("ALTER TABLE uploaded_files ADD COLUMN attached_msg_id INTEGER")
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_uploaded_files_chat ON uploaded_files(chat_id, upload_id);"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_uploaded_files_msg ON uploaded_files(attached_msg_id, upload_id);"
            )
            self._conn.commit()

    def create_chat(self, first_user_text: str) -> str:
        chat_id = str(uuid.uuid4())
        self.ensure_chat(chat_id, first_user_text)
        return chat_id

    def list_chats(self, limit: int = 100) -> List[ChatRow]:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT chat_id, title, created_at, updated_at FROM chats ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            )
            rows = cur.fetchall()
            return [ChatRow(**dict(r)) for r in rows]

    def get_messages(self, chat_id: str, limit: int = 2000) -> List[MessageRow]:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                SELECT msg_id, chat_id, role, content, turn_id, created_at
                FROM messages
                WHERE chat_id = ?
                ORDER BY msg_id ASC
                LIMIT ?
                """,
                (chat_id, limit),
            )
            rows = cur.fetchall()
            return [MessageRow(**dict(r)) for r in rows]

    def create_empty_chat(self, title: str = "Uploaded files") -> str:
        chat_id = str(uuid.uuid4())
        now = _now()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO chats(chat_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (chat_id, title, now, now),
            )
            self._conn.commit()
        return chat_id

    def get_chat(self, chat_id: str) -> ChatRow | None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT chat_id, title, created_at, updated_at FROM chats WHERE chat_id = ?",
                (chat_id,),
            )
            row = cur.fetchone()
        return ChatRow(**dict(row)) if row else None

    def delete_chat(self, chat_id: str) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("DELETE FROM chats WHERE chat_id = ?", (chat_id,))
            self._conn.commit()

    def ensure_chat(self, chat_id: str, first_user_text: str) -> None:
        now = _now()
        title = _title_from_first_user_text(first_user_text)
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT OR IGNORE INTO chats(chat_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (chat_id, title, now, now),
            )
            self._conn.commit()

    def maybe_update_title_from_first_user_text(self, chat_id: str, first_user_text: str) -> None:
        title = _title_from_first_user_text(first_user_text)
        if not title:
            return
        with self._lock:
            cur = self._conn.cursor()
            row = cur.execute(
                "SELECT title FROM chats WHERE chat_id = ?",
                (chat_id,),
            ).fetchone()
            if row is None or row["title"] not in DEFAULT_CHAT_TITLES:
                return
            count_row = cur.execute(
                "SELECT COUNT(*) AS n FROM messages WHERE chat_id = ? AND role = 'user' AND TRIM(content) <> ''",
                (chat_id,),
            ).fetchone()
            if count_row and int(count_row["n"]) > 0:
                return
            cur.execute(
                "UPDATE chats SET title = ? WHERE chat_id = ?",
                (title, chat_id),
            )
            self._conn.commit()

    def touch_chat(self, chat_id: str) -> None:
        now = _now()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "UPDATE chats SET updated_at = ? WHERE chat_id = ?", (now, chat_id)
            )
            self._conn.commit()

    def add_message(self, chat_id: str, role: str, content: str, turn_id: str | None = None) -> int:
        now = _now()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "INSERT INTO messages(chat_id, role, content, turn_id, created_at) VALUES (?, ?, ?, ?, ?)",
                (chat_id, role, content, turn_id, now),
            )
            cur.execute(
                "UPDATE chats SET updated_at = ? WHERE chat_id = ?", (now, chat_id)
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def add_uploaded_file(self, chat_id: str, original_name: str, stored_name: str, rel_path: str) -> int:
        now = _now()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO uploaded_files(chat_id, original_name, stored_name, rel_path, processed, created_at)
                VALUES (?, ?, ?, ?, 0, ?)
                """,
                (chat_id, original_name, stored_name, rel_path, now),
            )
            cur.execute(
                "UPDATE chats SET updated_at = ? WHERE chat_id = ?",
                (now, chat_id),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def list_uploaded_files(self, chat_id: str) -> List[UploadedFileRow]:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                SELECT upload_id, chat_id, original_name, stored_name, rel_path, processed, attached_msg_id, created_at
                FROM uploaded_files
                WHERE chat_id = ?
                ORDER BY upload_id ASC
                """,
                (chat_id,),
            )
            rows = cur.fetchall()
        return [UploadedFileRow(**dict(row)) for row in rows]

    def list_pending_uploaded_files(self, chat_id: str) -> List[UploadedFileRow]:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                SELECT upload_id, chat_id, original_name, stored_name, rel_path, processed, attached_msg_id, created_at
                FROM uploaded_files
                WHERE chat_id = ? AND attached_msg_id IS NULL
                ORDER BY upload_id ASC
                """,
                (chat_id,),
            )
            rows = cur.fetchall()
        return [UploadedFileRow(**dict(row)) for row in rows]

    def attach_pending_uploads_to_message(self, chat_id: str, msg_id: int) -> List[UploadedFileRow]:
        with self._lock:
            cur = self._conn.cursor()
            rows = cur.execute(
                """
                SELECT upload_id, chat_id, original_name, stored_name, rel_path, processed, attached_msg_id, created_at
                FROM uploaded_files
                WHERE chat_id = ? AND attached_msg_id IS NULL
                ORDER BY upload_id ASC
                """,
                (chat_id,),
            ).fetchall()
            cur.execute(
                """
                UPDATE uploaded_files
                SET attached_msg_id = ?
                WHERE chat_id = ? AND attached_msg_id IS NULL
                """,
                (msg_id, chat_id),
            )
            self._conn.commit()
        return [UploadedFileRow(**dict(row)) for row in rows]

    def mark_uploaded_files_processed(self, chat_id: str, msg_id: int) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                UPDATE uploaded_files
                SET processed = 1
                WHERE chat_id = ? AND processed = 0 AND attached_msg_id = ?
                """,
                (chat_id, msg_id),
            )
            self._conn.commit()

    def delete_uploaded_file(self, chat_id: str, upload_id: int) -> UploadedFileRow | None:
        with self._lock:
            cur = self._conn.cursor()
            row = cur.execute(
                """
                SELECT upload_id, chat_id, original_name, stored_name, rel_path, processed, attached_msg_id, created_at
                FROM uploaded_files
                WHERE chat_id = ? AND upload_id = ?
                """,
                (chat_id, upload_id),
            ).fetchone()
            if row is None:
                return None
            cur.execute(
                "DELETE FROM uploaded_files WHERE chat_id = ? AND upload_id = ?",
                (chat_id, upload_id),
            )
            self._conn.commit()
        return UploadedFileRow(**dict(row))
