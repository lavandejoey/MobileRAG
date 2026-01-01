#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chunker for RAG systems.
src/rag/chunker.py

@author: LIU Ziyi
@date: 2026-01-01
@license: Apache-2.0
"""
from __future__ import annotations

from typing import List


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[tuple[int, int, str]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    overlap = max(0, min(overlap, chunk_size - 1))

    t = text
    n = len(t)
    out: List[tuple[int, int, str]] = []

    start = 0
    while start < n:
        end = min(n, start + chunk_size)
        chunk = t[start:end].strip()
        if chunk:
            out.append((start, end, chunk))
        if end >= n:
            break
        start = max(0, end - overlap)

    return out
