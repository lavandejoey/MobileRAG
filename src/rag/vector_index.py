#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vector index with FAISS backend or numpy fallback.
src/rag/vector_index.py

@author: LIU Ziyi
@date: 2026-01-01
@license: Apache-2.0
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np


def _try_import_faiss():
    try:
        import faiss  # type: ignore
        return faiss
    except Exception:
        return None


class VectorIndex:
    """
    If faiss is available -> store/load a real FAISS index.
    Else -> store/load a numpy matrix and brute-force cosine (still as a separate index file).

    File name is kept as *.faiss for forward compatibility.
    """

    def __init__(self, index_path: str, dim: int, metric: str = "ip") -> None:
        self.index_path = Path(index_path)
        self.meta_path = self.index_path.with_suffix(self.index_path.suffix + ".meta.json")
        self.dim = int(dim)
        self.metric = metric
        self._faiss = _try_import_faiss()

        self._ids: List[str] = []
        self._mat: np.ndarray | None = None
        self._index = None

    def exists(self) -> bool:
        return self.index_path.exists() and self.meta_path.exists()

    def save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "dim": self.dim,
            "metric": self.metric,
            "backend": "faiss" if self._faiss and self._index is not None else "numpy",
            "count": len(self._ids),
        }
        self.meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        if self._faiss and self._index is not None:
            self._faiss.write_index(self._index, str(self.index_path))
            (self.index_path.with_suffix(self.index_path.suffix + ".ids.txt")).write_text(
                "\n".join(self._ids), encoding="utf-8"
            )
            return

        if self._mat is None:
            raise RuntimeError("cannot save: numpy matrix is empty")
        np.savez_compressed(str(self.index_path), mat=self._mat, ids=np.asarray(self._ids, dtype=object))

    def load(self) -> None:
        if not self.exists():
            raise FileNotFoundError(f"missing index files under {self.index_path}")

        meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self.dim = int(meta.get("dim", self.dim))
        self.metric = str(meta.get("metric", self.metric))

        if self._faiss and meta.get("backend") == "faiss":
            self._index = self._faiss.read_index(str(self.index_path))
            ids_path = self.index_path.with_suffix(self.index_path.suffix + ".ids.txt")
            self._ids = ids_path.read_text(encoding="utf-8").splitlines() if ids_path.exists() else []
            self._mat = None
            return

        data = np.load(str(self.index_path), allow_pickle=True)
        self._mat = data["mat"].astype(np.float32, copy=False)
        self._ids = [str(x) for x in data["ids"].tolist()]
        self._index = None

    def build(self, vectors: np.ndarray, ids: List[str]) -> None:
        if vectors.ndim != 2:
            raise ValueError("vectors must be 2D")
        if vectors.shape[1] != self.dim:
            raise ValueError(f"vector dim mismatch: got {vectors.shape[1]}, expected {self.dim}")
        if len(ids) != vectors.shape[0]:
            raise ValueError("ids length mismatch")

        self._ids = list(ids)

        if self._faiss:
            if self.metric == "ip":
                idx = self._faiss.IndexFlatIP(self.dim)
            else:
                idx = self._faiss.IndexFlatL2(self.dim)
            idx.add(vectors.astype(np.float32, copy=False))
            self._index = idx
            self._mat = None
        else:
            self._mat = vectors.astype(np.float32, copy=False)
            self._index = None

    def search(self, query_vectors: np.ndarray, k: int) -> Tuple[np.ndarray, List[List[str]]]:
        if query_vectors.ndim != 2:
            raise ValueError("query_vectors must be 2D")
        if query_vectors.shape[1] != self.dim:
            raise ValueError(f"query dim mismatch: got {query_vectors.shape[1]}, expected {self.dim}")
        if k <= 0:
            raise ValueError("k must be > 0")

        if self._faiss and self._index is not None:
            scores, idxs = self._index.search(query_vectors.astype(np.float32, copy=False), k)
            id_lists: List[List[str]] = []
            for row in idxs:
                row_ids = [self._ids[i] for i in row if 0 <= i < len(self._ids)]
                id_lists.append(row_ids)
            return scores, id_lists

        if self._mat is None:
            raise RuntimeError("index not loaded")

        sims = query_vectors.astype(np.float32, copy=False) @ self._mat.T
        k_eff = min(k, sims.shape[1])
        idxs = np.argpartition(-sims, kth=k_eff - 1, axis=1)[:, :k_eff]

        sorted_rows = []
        sorted_scores = []
        for qi in range(idxs.shape[0]):
            row = idxs[qi]
            row_scores = sims[qi, row]
            order = np.argsort(-row_scores)
            row = row[order]
            row_scores = row_scores[order]
            sorted_rows.append(row)
            sorted_scores.append(row_scores)

        idxs2 = np.stack(sorted_rows, axis=0)
        scores2 = np.stack(sorted_scores, axis=0)
        id_lists = [[self._ids[i] for i in row] for row in idxs2]
        return scores2, id_lists
