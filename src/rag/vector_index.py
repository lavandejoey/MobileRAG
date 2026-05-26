#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vector index with FAISS backend or numpy fallback.
src/rag/vector_index.py

Supports incremental add/remove by stable string ids.

@author: LIU Ziyi
@date: 2026-01-01
@license: Apache-2.0
"""
from __future__ import annotations

import hashlib
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
    If faiss is available -> store/load a mutable FAISS index with stable int64 ids.
    Else -> store/load a numpy matrix and mutate by filtering/appending rows.

    File name is kept as *.faiss for forward compatibility.
    """

    def __init__(self, index_path: str, dim: int, metric: str = "ip") -> None:
        self.index_path = Path(index_path)
        self.meta_path = self.index_path.with_suffix(self.index_path.suffix + ".meta.json")
        self.ids_path = self.index_path.with_suffix(self.index_path.suffix + ".ids.txt")
        self.dim = int(dim)
        self.metric = metric
        self._faiss = _try_import_faiss()

        self._ids: List[str] = []
        self._mat: np.ndarray | None = None
        self._index = None
        self._legacy_positional_ids = False
        self._int_to_string: dict[int, str] = {}
        self._string_to_int: dict[str, int] = {}

    def exists(self) -> bool:
        return self.index_path.exists() and self.meta_path.exists()

    def is_mutable(self) -> bool:
        if self._faiss and self._index is not None:
            return not self._legacy_positional_ids
        return True

    def _stable_int_id(self, string_id: str) -> int:
        existing = self._string_to_int.get(string_id)
        if existing is not None:
            return existing

        digest = hashlib.blake2b(string_id.encode("utf-8", errors="ignore"), digest_size=8).digest()
        candidate = int.from_bytes(digest, "big") & ((1 << 63) - 1)
        if candidate == 0:
            candidate = 1
        while True:
            seen = self._int_to_string.get(candidate)
            if seen is None or seen == string_id:
                self._int_to_string[candidate] = string_id
                self._string_to_int[string_id] = candidate
                return candidate
            candidate = (candidate + 1) & ((1 << 63) - 1)
            if candidate == 0:
                candidate = 1

    def _reset_mappings(self) -> None:
        self._ids = []
        self._int_to_string = {}
        self._string_to_int = {}
        self._legacy_positional_ids = False

    def _make_empty_faiss_index(self):
        if not self._faiss:
            return None
        if self.metric == "ip":
            base = self._faiss.IndexFlatIP(self.dim)
        else:
            base = self._faiss.IndexFlatL2(self.dim)
        return self._faiss.IndexIDMap2(base)

    def save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "dim": self.dim,
            "metric": self.metric,
            "backend": "faiss" if self._faiss and self._index is not None else "numpy",
            "count": len(self._ids),
            "format_version": 2,
        }
        self.meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        if self._faiss and self._index is not None:
            self._faiss.write_index(self._index, str(self.index_path))
            lines = [f"{self._string_to_int[sid]}\t{sid}" for sid in self._ids]
            self.ids_path.write_text("\n".join(lines), encoding="utf-8")
            return

        if self._mat is None:
            raise RuntimeError("cannot save: numpy matrix is empty")
        with self.index_path.open("wb") as f:
            np.savez_compressed(f, mat=self._mat, ids=np.asarray(self._ids, dtype=object))

    def load(self) -> None:
        if not self.exists():
            raise FileNotFoundError(f"missing index files under {self.index_path}")

        meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self.dim = int(meta.get("dim", self.dim))
        self.metric = str(meta.get("metric", self.metric))
        backend = str(meta.get("backend", "numpy"))

        self._reset_mappings()

        if self._faiss and backend == "faiss":
            self._index = self._faiss.read_index(str(self.index_path))
            self._mat = None
            raw_lines = self.ids_path.read_text(encoding="utf-8").splitlines() if self.ids_path.exists() else []
            if raw_lines and all("\t" in line for line in raw_lines):
                ordered: list[tuple[int, str]] = []
                for line in raw_lines:
                    raw_int, sid = line.split("\t", 1)
                    int_id = int(raw_int)
                    ordered.append((int_id, sid))
                    self._int_to_string[int_id] = sid
                    self._string_to_int[sid] = int_id
                self._ids = [sid for _, sid in ordered]
                self._legacy_positional_ids = False
            else:
                self._ids = [line for line in raw_lines if line]
                self._legacy_positional_ids = True
            return

        with self.index_path.open("rb") as f:
            data = np.load(f, allow_pickle=True)
            mat = data["mat"].astype(np.float32, copy=False)
            ids = [str(x) for x in data["ids"].tolist()]
        self._mat = mat
        self._index = None
        self._ids = ids
        for sid in self._ids:
            self._stable_int_id(sid)

    def build(self, vectors: np.ndarray, ids: List[str]) -> None:
        if vectors.ndim != 2:
            raise ValueError("vectors must be 2D")
        if vectors.shape[1] != self.dim:
            raise ValueError(f"vector dim mismatch: got {vectors.shape[1]}, expected {self.dim}")
        if len(ids) != vectors.shape[0]:
            raise ValueError("ids length mismatch")

        self._reset_mappings()
        self._ids = list(ids)

        if self._faiss:
            idx = self._make_empty_faiss_index()
            if idx is None:
                raise RuntimeError("failed to create faiss index")
            if ids:
                int_ids = np.asarray([self._stable_int_id(sid) for sid in ids], dtype=np.int64)
                idx.add_with_ids(vectors.astype(np.float32, copy=False), int_ids)
            self._index = idx
            self._mat = None
            return

        self._mat = vectors.astype(np.float32, copy=False)
        self._index = None
        for sid in self._ids:
            self._stable_int_id(sid)

    def add(self, vectors: np.ndarray, ids: List[str]) -> None:
        if not ids:
            return
        if vectors.ndim != 2:
            raise ValueError("vectors must be 2D")
        if vectors.shape[1] != self.dim:
            raise ValueError(f"vector dim mismatch: got {vectors.shape[1]}, expected {self.dim}")
        if len(ids) != vectors.shape[0]:
            raise ValueError("ids length mismatch")

        if self._faiss:
            if self._index is None:
                self._index = self._make_empty_faiss_index()
            if self._legacy_positional_ids:
                raise RuntimeError("legacy faiss index does not support incremental mutation")
            int_ids = np.asarray([self._stable_int_id(sid) for sid in ids], dtype=np.int64)
            self._index.add_with_ids(vectors.astype(np.float32, copy=False), int_ids)
            self._ids.extend(ids)
            return

        if self._mat is None:
            self._mat = np.empty((0, self.dim), dtype=np.float32)
        self._mat = np.concatenate([self._mat, vectors.astype(np.float32, copy=False)], axis=0)
        self._ids.extend(ids)
        for sid in ids:
            self._stable_int_id(sid)

    def remove_ids(self, ids: List[str]) -> None:
        if not ids:
            return
        id_set = set(ids)

        if self._faiss:
            if self._index is None:
                return
            if self._legacy_positional_ids:
                raise RuntimeError("legacy faiss index does not support incremental mutation")
            int_ids = [self._string_to_int[sid] for sid in ids if sid in self._string_to_int]
            if int_ids:
                arr = np.asarray(int_ids, dtype=np.int64)
                self._index.remove_ids(arr)
            self._ids = [sid for sid in self._ids if sid not in id_set]
            for sid in ids:
                int_id = self._string_to_int.pop(sid, None)
                if int_id is not None:
                    self._int_to_string.pop(int_id, None)
            return

        if self._mat is None or not self._ids:
            return
        keep_idx = [i for i, sid in enumerate(self._ids) if sid not in id_set]
        self._mat = self._mat[np.asarray(keep_idx, dtype=np.int64)] if keep_idx else np.empty((0, self.dim), dtype=np.float32)
        self._ids = [self._ids[i] for i in keep_idx]
        for sid in ids:
            int_id = self._string_to_int.pop(sid, None)
            if int_id is not None:
                self._int_to_string.pop(int_id, None)

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
                row_ids: List[str] = []
                for raw_id in row:
                    idx = int(raw_id)
                    if idx < 0:
                        continue
                    if self._legacy_positional_ids:
                        if 0 <= idx < len(self._ids):
                            row_ids.append(self._ids[idx])
                        continue
                    sid = self._int_to_string.get(idx)
                    if sid:
                        row_ids.append(sid)
                id_lists.append(row_ids)
            return scores, id_lists

        if self._mat is None:
            raise RuntimeError("index not loaded")
        if self._mat.shape[0] == 0:
            return np.zeros((query_vectors.shape[0], 0), dtype=np.float32), [[] for _ in range(query_vectors.shape[0])]

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
