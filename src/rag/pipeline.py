from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.config import AppConfig
from src.rag.chunker import chunk_text
from src.rag.embedder import create_embedder
from src.rag.fs_scan import list_doc_paths
from src.rag.index_sqlite import RagSqliteStore
from src.rag.parsers import file_sha1, parse_file_sections
from src.rag.rerank import create_reranker
from src.rag.types import ChunkRecord, DocRecord, RagSnippet
from src.rag.vector_index import VectorIndex


def _stable_doc_id(path: str) -> str:
    return hashlib.sha1(path.encode("utf-8", errors="ignore")).hexdigest()


def _embed_chunks(embedder, chunks: List[ChunkRecord]):
    if not chunks:
        return None, []
    ids = [c.chunk_id for c in chunks]
    texts = [c.text for c in chunks]
    vecs = embedder.embed(texts)
    if vecs.shape[0] != len(ids):
        raise RuntimeError("embedding count mismatch")
    return vecs, ids


@dataclass
class BuildStats:
    scanned: int = 0
    updated_docs: int = 0
    removed_docs: int = 0
    updated_chunks: int = 0
    rebuilt_index: bool = False
    ms: int = 0


class RagPipeline:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.enabled = bool(cfg.RAG.ENABLED)

        base = Path(cfg.RAG.INDEX_DIR).expanduser()
        base.mkdir(parents=True, exist_ok=True)

        self.sqlite_path = str(base / cfg.RAG.SQLITE_FILE)
        self.index_path = str(base / cfg.RAG.INDEX_FILE)

        self.store = RagSqliteStore(self.sqlite_path)
        self.embedder = create_embedder(
            backend=cfg.RAG.EMBEDDER_BACKEND,
            dim=cfg.RAG.EMBED_DIM,
            ollama_url=cfg.RAG.OLLAMA_URL,
            ollama_model=cfg.RAG.OLLAMA_EMBED_MODEL,
        )
        self.reranker = create_reranker(cfg.RAG.RERANK_BACKEND, cfg.RAG.RERANK_ALPHA)
        self.vindex = VectorIndex(index_path=self.index_path, dim=cfg.RAG.EMBED_DIM, metric="ip")

        self._loaded = False

    def warmup(self, build_if_missing: bool = True) -> Dict[str, int | bool]:
        if not self.enabled:
            return {"ok": True, "enabled": False, "loaded": False, "rebuilt_index": False}

        if self.vindex.exists():
            self._ensure_loaded()
            return {"ok": True, "enabled": True, "loaded": True, "rebuilt_index": False}

        if not build_if_missing:
            return {"ok": True, "enabled": True, "loaded": False, "rebuilt_index": False}

        return self.build_or_update_index()

    def _ensure_loaded(self) -> None:
        if not self.enabled:
            return
        if self._loaded and self.vindex.exists():
            return
        if self.vindex.exists():
            self.vindex.load()
            self._loaded = True

    def build_or_update_index(self) -> Dict[str, int | bool]:
        if not self.enabled:
            return {"ok": True, "updated_docs": 0, "updated_chunks": 0, "rebuilt_index": False}

        t0 = time.perf_counter()
        stats = BuildStats()

        paths = list_doc_paths(
            patterns=self.cfg.DOCS_GLOBS,
            exts=None,
            follow_symlinks=False,
            max_file_size_mb=self.cfg.RAG.MAX_FILE_SIZE_MB,
        )
        stats.scanned = len(paths)
        live_paths = {str(p.resolve()) for p in paths}

        removed_docs: List[DocRecord] = []
        changed_docs: List[tuple[DocRecord | None, DocRecord, List[ChunkRecord]]] = []

        for existing_doc in self.store.list_docs():
            if existing_doc.path not in live_paths:
                removed_docs.append(existing_doc)
                stats.removed_docs += 1

        for p in paths:
            ap = str(p.resolve())
            mtime = float(p.stat().st_mtime)
            existing = self.store.get_doc_by_path(ap)

            if existing is not None and abs(existing.mtime - mtime) < 1e-6:
                continue

            sha1 = file_sha1(p)
            if existing is not None and existing.sha1 == sha1:
                self.store.upsert_doc(DocRecord(existing.doc_id, ap, mtime, sha1, existing.mime))
                continue

            try:
                sections, mime = parse_file_sections(p)
            except Exception:
                continue

            doc_id = existing.doc_id if existing is not None else _stable_doc_id(ap)
            next_doc = DocRecord(doc_id=doc_id, path=ap, mtime=mtime, sha1=sha1, mime=mime)

            chunks: List[ChunkRecord] = []
            chunk_idx = 0
            for section in sections:
                spans = chunk_text(
                    section.text,
                    chunk_size=self.cfg.RAG.CHUNK_SIZE,
                    overlap=self.cfg.RAG.CHUNK_OVERLAP,
                )
                for s, e, ctext in spans:
                    chunk_id = f"{doc_id}:{chunk_idx:06d}"
                    chunks.append(
                        ChunkRecord(
                            chunk_id=chunk_id,
                            doc_id=doc_id,
                            path=ap,
                            idx=chunk_idx,
                            start=s,
                            end=e,
                            text=ctext,
                            source_label=section.source_label,
                        )
                    )
                    chunk_idx += 1
            changed_docs.append((existing, next_doc, chunks))
            stats.updated_docs += 1
            stats.updated_chunks += len(chunks)

        any_change = bool(removed_docs or changed_docs)
        needs_full_rebuild = not self.vindex.exists()
        if self.vindex.exists():
            self._ensure_loaded()
            if any_change and not self.vindex.is_mutable():
                needs_full_rebuild = True

        if needs_full_rebuild:
            for existing_doc in removed_docs:
                self.store.delete_doc(existing_doc.doc_id)
            for existing, next_doc, chunks in changed_docs:
                if existing is not None:
                    self.store.delete_chunks_for_doc(existing.doc_id)
                self.store.upsert_doc(next_doc)
                self.store.insert_chunks(chunks)

            all_chunks = self.store.get_all_chunks()
            if all_chunks:
                vecs, ids = _embed_chunks(self.embedder, all_chunks)
                assert vecs is not None
                self.vindex.build(vecs, ids)
            else:
                self.vindex.build(np.empty((0, self.cfg.RAG.EMBED_DIM), dtype=np.float32), [])
            self.vindex.save()
            self._loaded = True
            stats.rebuilt_index = True
        elif any_change:
            for existing_doc in removed_docs:
                old_chunk_ids = self.store.list_chunk_ids_for_doc(existing_doc.doc_id)
                self.vindex.remove_ids(old_chunk_ids)
                self.store.delete_doc(existing_doc.doc_id)

            for existing, next_doc, chunks in changed_docs:
                old_chunk_ids = self.store.list_chunk_ids_for_doc(next_doc.doc_id) if existing is not None else []
                if old_chunk_ids:
                    self.vindex.remove_ids(old_chunk_ids)
                if existing is not None:
                    self.store.delete_chunks_for_doc(existing.doc_id)
                self.store.upsert_doc(next_doc)
                self.store.insert_chunks(chunks)
                vecs, ids = _embed_chunks(self.embedder, chunks)
                if vecs is not None:
                    self.vindex.add(vecs, ids)

            self.vindex.save()
            self._loaded = True

        stats.ms = int((time.perf_counter() - t0) * 1000)
        return {
            "ok": True,
            "scanned": stats.scanned,
            "updated_docs": stats.updated_docs,
            "removed_docs": stats.removed_docs,
            "updated_chunks": stats.updated_chunks,
            "rebuilt_index": bool(stats.rebuilt_index),
            "ms": stats.ms,
        }

    def _snippets_from_chunks(
            self,
            query: str,
            chunks: List[ChunkRecord],
            top_k: int,
            preferred_doc_ids: Optional[set[str]] = None,
    ) -> List[RagSnippet]:
        if not chunks:
            return []

        qv = self.embedder.embed([query])
        vecs, ids = _embed_chunks(self.embedder, chunks)
        if vecs is None:
            return []

        sims = (qv @ vecs.T)[0]
        by_doc_first_seen: set[str] = set()
        snips: List[RagSnippet] = []
        for idx, chunk in enumerate(chunks):
            score = float(sims[idx]) if idx < len(sims) else 0.0
            file_name = Path(chunk.path).name.lower()
            ql = query.lower()
            if ql and any(tok in file_name for tok in ql.split()):
                score += 0.12
            if chunk.idx == 0:
                score += 0.05
            if preferred_doc_ids and chunk.doc_id in preferred_doc_ids and chunk.doc_id not in by_doc_first_seen:
                score += 0.08
                by_doc_first_seen.add(chunk.doc_id)
            snips.append(
                RagSnippet(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    path=chunk.path,
                    score=score,
                    text=chunk.text,
                    source_label=chunk.source_label,
                    citation_id=None,
                )
            )

        snips = self.reranker.rerank(query, snips)
        return snips[:top_k]

    def retrieve(self, query: str, top_k: int | None = None, preferred_doc_ids: Optional[List[str]] = None) -> List[RagSnippet]:
        if not self.enabled:
            return []

        if not self.vindex.exists():
            self.build_or_update_index()
        self._ensure_loaded()

        top_k = int(top_k or self.cfg.RAG.TOP_K)
        cand_k = int(max(top_k, self.cfg.RAG.CANDIDATES_K))

        if preferred_doc_ids:
            preferred_chunks = self.store.get_chunks_for_doc_ids(preferred_doc_ids)
            preferred_snips = self._snippets_from_chunks(
                query,
                preferred_chunks,
                top_k=top_k,
                preferred_doc_ids=set(preferred_doc_ids),
            )
            if preferred_snips:
                return preferred_snips

        qv = self.embedder.embed([query])
        scores, id_lists = self.vindex.search(qv, k=cand_k)
        if not id_lists or not id_lists[0]:
            return []

        cand_ids = id_lists[0]
        cand_chunks = self.store.get_chunk_text_by_ids(cand_ids)

        by_id = {c.chunk_id: c for c in cand_chunks}
        snips: List[RagSnippet] = []
        for rank, cid in enumerate(cand_ids):
            c = by_id.get(cid)
            if c is None:
                continue
            score = float(scores[0][rank]) if scores.size else 0.0
            snips.append(
                RagSnippet(
                    chunk_id=cid,
                    doc_id=c.doc_id,
                    path=c.path,
                    score=score,
                    text=c.text,
                    source_label=c.source_label,
                    citation_id=None,
                )
            )

        snips = self.reranker.rerank(query, snips)
        return snips[:top_k]

