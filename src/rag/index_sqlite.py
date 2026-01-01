#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SQLite-based vector index for RAG systems.
src/rag/index_sqlite.py

@author: LIU Ziyi
@date: 2026-01-01
@license: Apache-2.0
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import List, Optional

from src.rag.types import ChunkRecord, DocRecord


class RagSqliteStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS docs
                (
                    doc_id TEXT PRIMARY KEY,
                    path   TEXT NOT NULL,
                    mtime  REAL NOT NULL,
                    sha1   TEXT NOT NULL,
                    mime   TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks
                (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id   TEXT    NOT NULL,
                    path     TEXT    NOT NULL,
                    idx      INTEGER NOT NULL,
                    start    INTEGER NOT NULL,
                    end      INTEGER NOT NULL,
                    text     TEXT    NOT NULL,
                    FOREIGN KEY (doc_id) REFERENCES docs (doc_id)
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);")

    def get_doc_by_path(self, path: str) -> Optional[DocRecord]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM docs WHERE path=?", (path,)).fetchone()
        if not row:
            return None
        return DocRecord(
            doc_id=row["doc_id"],
            path=row["path"],
            mtime=float(row["mtime"]),
            sha1=row["sha1"],
            mime=row["mime"],
        )

    def upsert_doc(self, doc: DocRecord) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO docs(doc_id, path, mtime, sha1, mime)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET path=excluded.path,
                                                  mtime=excluded.mtime,
                                                  sha1=excluded.sha1,
                                                  mime=excluded.mime
                """,
                (doc.doc_id, doc.path, doc.mtime, doc.sha1, doc.mime),
            )

    def delete_chunks_for_doc(self, doc_id: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))

    def insert_chunks(self, chunks: List[ChunkRecord]) -> None:
        if not chunks:
            return
        with self._conn() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO chunks(chunk_id, doc_id, path, idx, start, end, text)
                VALUES(?,?,?,?,?,?,?)
                """,
                [(c.chunk_id, c.doc_id, c.path, c.idx, c.start, c.end, c.text) for c in chunks],
            )

    def get_all_chunks(self) -> List[ChunkRecord]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM chunks ORDER BY chunk_id").fetchall()
        out: List[ChunkRecord] = []
        for r in rows:
            out.append(
                ChunkRecord(
                    chunk_id=r["chunk_id"],
                    doc_id=r["doc_id"],
                    path=r["path"],
                    idx=int(r["idx"]),
                    start=int(r["start"]),
                    end=int(r["end"]),
                    text=r["text"],
                )
            )
        return out

    def get_chunk_text_by_ids(self, chunk_ids: List[str]) -> List[ChunkRecord]:
        if not chunk_ids:
            return []
        q = "SELECT * FROM chunks WHERE chunk_id IN (%s)" % (",".join(["?"] * len(chunk_ids)))
        with self._conn() as conn:
            rows = conn.execute(q, tuple(chunk_ids)).fetchall()
        by_id = {r["chunk_id"]: r for r in rows}
        out: List[ChunkRecord] = []
        for cid in chunk_ids:
            r = by_id.get(cid)
            if not r:
                continue
            out.append(
                ChunkRecord(
                    chunk_id=r["chunk_id"],
                    doc_id=r["doc_id"],
                    path=r["path"],
                    idx=int(r["idx"]),
                    start=int(r["start"]),
                    end=int(r["end"]),
                    text=r["text"],
                )
            )
        return out
