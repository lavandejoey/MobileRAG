# -*- coding: utf-8 -*-
"""
@file: core/history/store.py
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/14
@version: 0.8.0
"""

import json
import sqlite3
from typing import Any, Dict, List, Optional


class ChatHistoryStore:
    def __init__(self, db_path: str = "./chat_history.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                turn_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                token_count INTEGER NOT NULL,
                evidence_json TEXT,  -- New column for serialized evidence
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(session_id, turn_id)
            )
        """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL UNIQUE,
                summary TEXT NOT NULL,
                token_count INTEGER NOT NULL,
                last_turn_id INTEGER NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        conn.commit()
        conn.close()

    def append_message(
        self,
        session_id: str,
        turn_id: int,
        role: str,
        content: str,
        token_count: int,
        evidence: Optional[List[Dict[str, Any]]] = None,
    ):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        evidence_json = json.dumps(evidence) if evidence is not None else None
        cursor.execute(
            "INSERT INTO chat_messages "
            "(session_id, turn_id, role, content, token_count, evidence_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, turn_id, role, content, token_count, evidence_json),
        )
        conn.commit()
        conn.close()

    def load_messages(self, session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        query = (
            "SELECT session_id, turn_id, role, content, token_count, evidence_json, timestamp "
            "FROM chat_messages WHERE session_id = ? ORDER BY turn_id ASC"
        )
        params = [session_id]
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        cursor.execute(query, params)
        messages = []
        for row in cursor.fetchall():
            evidence = json.loads(row[5]) if row[5] else None
            messages.append(
                {
                    "session_id": row[0],
                    "turn_id": row[1],
                    "role": row[2],
                    "content": row[3],
                    "token_count": row[4],
                    "evidence": evidence,
                    "timestamp": row[6],
                }
            )
        conn.close()
        return messages

    def get_message_by_turn_id(self, session_id: str, turn_id: int) -> Optional[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT session_id, turn_id, role, content, token_count, evidence_json, timestamp "
            "FROM chat_messages WHERE session_id = ? AND turn_id = ?",
            (session_id, turn_id),
        )
        row = cursor.fetchone()
        conn.close()
        if row:
            evidence = json.loads(row[5]) if row[5] else None
            return {
                "session_id": row[0],
                "turn_id": row[1],
                "role": row[2],
                "content": row[3],
                "token_count": row[4],
                "evidence": evidence,
                "timestamp": row[6],
            }
        return None

    def get_last_turn_id(self, session_id: str) -> Optional[int]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(turn_id) FROM chat_messages WHERE session_id = ?", (session_id,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result and result[0] is not None else None

    def save_summary(self, session_id: str, summary: str, token_count: int, last_turn_id: int):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO chat_summaries (session_id, summary, token_count, last_turn_id)
            VALUES (?, ?, ?, ?)
            """,
            (session_id, summary, token_count, last_turn_id),
        )
        conn.commit()
        conn.close()

    def load_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT session_id, summary, token_count, last_turn_id, timestamp "
            "FROM chat_summaries WHERE session_id = ?",
            (session_id,),
        )
        row = cursor.fetchone()
        conn.close()
        if row:
            return {
                "session_id": row[0],
                "summary": row[1],
                "token_count": row[2],
                "last_turn_id": row[3],
                "timestamp": row[4],
            }
        return None

    def delete_session_history(self, session_id: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM chat_summaries WHERE session_id = ?", (session_id,))
        conn.commit()
        conn.close()
