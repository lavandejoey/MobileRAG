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

import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


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
    created_at: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "msg_id": self.msg_id,
            "chat_id": self.chat_id,
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at,
        }


def _now() -> float:
    return time.time()


def _title_from_first_user_text(text: str, max_len: int = 48) -> str:
    s = " ".join((text or "").strip().split())
    if not s:
        return "New chat"
    if len(s) <= max_len:
        return s
    if max_len <= 1:
        return "…"
    return s[: max_len - 1] + "…"


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
                    created_at REAL NOT NULL,
                    FOREIGN KEY (chat_id) REFERENCES chats (chat_id) ON DELETE CASCADE
                );
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id, msg_id);"
            )
            self._conn.commit()

    def create_chat(self, title: str) -> str:
        chat_id = str(uuid.uuid4())
        self.ensure_chat(chat_id, title)
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
                SELECT msg_id, chat_id, role, content, created_at
                FROM messages
                WHERE chat_id = ?
                ORDER BY msg_id ASC
                LIMIT ?
                """,
                (chat_id, limit),
            )
            rows = cur.fetchall()
            return [MessageRow(**dict(r)) for r in rows]

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

    def touch_chat(self, chat_id: str) -> None:
        now = _now()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "UPDATE chats SET updated_at = ? WHERE chat_id = ?", (now, chat_id)
            )
            self._conn.commit()

    def add_message(self, chat_id: str, role: str, content: str) -> None:
        now = _now()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "INSERT INTO messages(chat_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (chat_id, role, content, now),
            )
            cur.execute(
                "UPDATE chats SET updated_at = ? WHERE chat_id = ?", (now, chat_id)
            )
            self._conn.commit()
