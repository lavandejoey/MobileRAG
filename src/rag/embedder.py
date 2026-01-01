#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embedding backends.
src/rag/embedder.py

@author: LIU Ziyi
@date: 2026-01-01
@license: Apache-2.0
"""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from typing import List

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


class Embedder:
    def embed(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError


@dataclass
class HashingEmbedder(Embedder):
    dim: int = 2048

    def __post_init__(self) -> None:
        self._vec = HashingVectorizer(
            n_features=self.dim,
            alternate_sign=False,
            norm=None,
            lowercase=True,
            analyzer="word",
            token_pattern=r"(?u)\b\w+\b",
        )

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        X = self._vec.transform(texts)
        dense = X.toarray().astype(np.float32, copy=False)
        return _l2_normalize(dense)


@dataclass
class OllamaEmbedder(Embedder):
    base_url: str = "http://localhost:11434"
    model: str = "nomic-embed-text"
    timeout_s: int = 60

    def _post_json(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            url=self.base_url.rstrip("/") + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
            return json.loads(body)

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        vecs: List[List[float]] = []
        for t in texts:
            payload = {"model": self.model, "prompt": t}

            out = None
            last_error: Exception | None = None
            for endpoint in ("/api/embeddings", "/api/embed"):
                try:
                    out = self._post_json(endpoint, payload)
                    if isinstance(out, dict) and ("embedding" in out or "embeddings" in out):
                        break
                except Exception as e:
                    last_error = e
                    out = None

            if out is None:
                raise RuntimeError(f"ollama embed request failed: {last_error}")

            if "embedding" in out and isinstance(out["embedding"], list):
                vec = out["embedding"]
            elif "embeddings" in out and isinstance(out["embeddings"], list) and out["embeddings"]:
                vec = out["embeddings"][0]
            else:
                raise RuntimeError(f"unexpected ollama embed response keys: {list(out.keys())}")

            vecs.append([float(x) for x in vec])

        arr = np.asarray(vecs, dtype=np.float32)
        return _l2_normalize(arr)


def create_embedder(backend: str, dim: int, ollama_url: str, ollama_model: str) -> Embedder:
    b = (backend or "").lower()
    if b in ("hashing", "hash", "hashing_vectorizer", "bow"):
        return HashingEmbedder(dim=dim)
    if b in ("ollama", "ollama_embed"):
        return OllamaEmbedder(base_url=ollama_url, model=ollama_model)
    raise ValueError(f"unknown embedder backend: {backend}")
