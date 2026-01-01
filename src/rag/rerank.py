#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG snippet reranker implementations.
src/rag/rerank.py

@author: LIU Ziyi
@date: 2026-01-01
@license: Apache-2.0
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from src.rag.types import RagSnippet

_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)


def _token_set(text: str) -> set[str]:
    return {m.group(0).lower() for m in _WORD_RE.finditer(text)}


@dataclass
class HybridReranker:
    alpha: float = 0.10

    def rerank(self, query: str, snippets: List[RagSnippet]) -> List[RagSnippet]:
        if not snippets:
            return snippets
        q = _token_set(query)
        if not q:
            return snippets

        rescored: List[RagSnippet] = []
        for s in snippets:
            t = _token_set(s.text)
            overlap = 0.0
            if t:
                overlap = len(q & t) / max(1, len(q))
            score = float(s.score) + self.alpha * float(overlap)
            rescored.append(RagSnippet(s.chunk_id, s.doc_id, s.path, score, s.text))

        rescored.sort(key=lambda x: x.score, reverse=True)
        return rescored


def create_reranker(backend: str, alpha: float) -> HybridReranker:
    b = (backend or "").lower()
    if b in ("hybrid", "overlap", "lexical"):
        return HybridReranker(alpha=alpha)
    raise ValueError(f"unknown rerank backend: {backend}")
