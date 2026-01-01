#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG data types.
src/rag/types.py

@author: LIU Ziyi
@date: 2026-01-01
@license: Apache-2.0
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RagSnippet:
    chunk_id: str
    doc_id: str
    path: str
    score: float
    text: str


@dataclass(frozen=True)
class DocRecord:
    doc_id: str
    path: str
    mtime: float
    sha1: str
    mime: str


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    doc_id: str
    path: str
    idx: int
    start: int
    end: int
    text: str
