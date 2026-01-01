#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration management for the application.
src/config.py

@author: LIU Ziyi
@date: 2025-12-30
@license: Apache-2.0
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml

DEFAULT_CONFIG_PATH = Path("configs/mobile_rag.yaml")


@dataclass(frozen=True)
class ModelConfig:
    BACKEND: str = "ollama"
    MODEL_NAME: str = "qwen3:0.6b"
    TEMPERATURE: float = 0.2
    TOP_P: float = 0.9
    MAX_NEW_TOKENS: int = 512
    STREAM: bool = True
    THINK: bool = False

    @property
    def name(self) -> str:
        return self.MODEL_NAME

    @property
    def temperature(self) -> float:
        return self.TEMPERATURE

    @property
    def top_p(self) -> float:
        return self.TOP_P

    @property
    def max_new_tokens(self) -> int:
        return self.MAX_NEW_TOKENS

    @property
    def stream(self) -> bool:
        return self.STREAM

    @property
    def THINK_FLAG(self) -> bool:
        return self.THINK


@dataclass(frozen=True)
class RagConfig:
    ENABLED: bool = True

    # Storage
    INDEX_DIR: str = "data/rag"
    INDEX_FILE: str = "chunks.index.faiss"  # if faiss exists -> real faiss index; else -> npz stored with this name
    SQLITE_FILE: str = "rag_meta.db"

    # Scanning
    MAX_FILE_SIZE_MB: int = 30

    # Chunking
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 150

    # Retrieval
    TOP_K: int = 6
    CANDIDATES_K: int = 30  # pre-rerank candidates

    # Embedding (configurable)
    EMBEDDER_BACKEND: str = "hashing"  # "hashing" | "ollama"
    EMBED_DIM: int = 2048
    OLLAMA_URL: str = "http://localhost:11434"
    OLLAMA_EMBED_MODEL: str = "nomic-embed-text"

    # Rerank (configurable)
    RERANK_BACKEND: str = "hybrid"  # "hybrid"
    RERANK_ALPHA: float = 0.10

    # Prompt injection
    PROMPT_MAX_CHARS: int = 6000


@dataclass(frozen=True)
class AppConfig:
    LOG_LEVEL: str = "INFO"
    DEVICE: str = "auto"
    HISTORY: str = "data/history"
    DOCS_GLOBS: tuple[str, ...] = ("data/raw/*",)
    DOCS_EXTS: tuple[str, ...] = (".txt", ".md", ".pdf")
    MODEL: ModelConfig = ModelConfig()
    RAG: RagConfig = RagConfig()

    @property
    def model(self) -> ModelConfig:
        return self.MODEL

    @property
    def rag(self) -> RagConfig:
        return self.RAG


def _get(d: Dict[str, Any], key: str, default: Any) -> Any:
    v = d.get(key, default)
    return default if v is None else v


def load_config(path: str | Path | None = None) -> AppConfig:
    p = Path(path or DEFAULT_CONFIG_PATH)
    data: Dict[str, Any] = {}
    if p.exists():
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}

    # promote lowercase keys to uppercase for backward compatibility
    normalized = {str(k).upper(): v for k, v in data.items()}

    model_d_raw = normalized.get("MODEL", {}) or {}
    model_d = {str(k).upper(): v for k, v in model_d_raw.items()}

    rag_d_raw = normalized.get("RAG", {}) or {}
    rag_d = {str(k).upper(): v for k, v in rag_d_raw.items()}

    model = ModelConfig(
        BACKEND=str(_get(model_d, "BACKEND", ModelConfig.BACKEND)),
        MODEL_NAME=str(_get(model_d, "MODEL_NAME", ModelConfig.MODEL_NAME)),
        TEMPERATURE=float(_get(model_d, "TEMPERATURE", ModelConfig.TEMPERATURE)),
        TOP_P=float(_get(model_d, "TOP_P", ModelConfig.TOP_P)),
        MAX_NEW_TOKENS=int(_get(model_d, "MAX_NEW_TOKENS", ModelConfig.MAX_NEW_TOKENS)),
        STREAM=bool(_get(model_d, "STREAM", ModelConfig.STREAM)),
        THINK=bool(_get(model_d, "THINK", ModelConfig.THINK)),
    )

    rag = RagConfig(
        ENABLED=bool(_get(rag_d, "ENABLED", RagConfig.ENABLED)),
        INDEX_DIR=str(_get(rag_d, "INDEX_DIR", RagConfig.INDEX_DIR)),
        INDEX_FILE=str(_get(rag_d, "INDEX_FILE", RagConfig.INDEX_FILE)),
        SQLITE_FILE=str(_get(rag_d, "SQLITE_FILE", RagConfig.SQLITE_FILE)),
        MAX_FILE_SIZE_MB=int(_get(rag_d, "MAX_FILE_SIZE_MB", RagConfig.MAX_FILE_SIZE_MB)),
        CHUNK_SIZE=int(_get(rag_d, "CHUNK_SIZE", RagConfig.CHUNK_SIZE)),
        CHUNK_OVERLAP=int(_get(rag_d, "CHUNK_OVERLAP", RagConfig.CHUNK_OVERLAP)),
        TOP_K=int(_get(rag_d, "TOP_K", RagConfig.TOP_K)),
        CANDIDATES_K=int(_get(rag_d, "CANDIDATES_K", RagConfig.CANDIDATES_K)),
        EMBEDDER_BACKEND=str(_get(rag_d, "EMBEDDER_BACKEND", RagConfig.EMBEDDER_BACKEND)),
        EMBED_DIM=int(_get(rag_d, "EMBED_DIM", RagConfig.EMBED_DIM)),
        OLLAMA_URL=str(_get(rag_d, "OLLAMA_URL", RagConfig.OLLAMA_URL)),
        OLLAMA_EMBED_MODEL=str(_get(rag_d, "OLLAMA_EMBED_MODEL", RagConfig.OLLAMA_EMBED_MODEL)),
        RERANK_BACKEND=str(_get(rag_d, "RERANK_BACKEND", RagConfig.RERANK_BACKEND)),
        RERANK_ALPHA=float(_get(rag_d, "RERANK_ALPHA", RagConfig.RERANK_ALPHA)),
        PROMPT_MAX_CHARS=int(_get(rag_d, "PROMPT_MAX_CHARS", RagConfig.PROMPT_MAX_CHARS)),
    )

    return AppConfig(
        LOG_LEVEL=str(_get(normalized, "LOG_LEVEL", AppConfig.LOG_LEVEL)),
        DEVICE=str(_get(normalized, "DEVICE", AppConfig.DEVICE)),
        HISTORY=str(_get(normalized, "HISTORY", AppConfig.HISTORY)),
        DOCS_GLOBS=tuple(_get(normalized, "DOCS_GLOBS", list(AppConfig.DOCS_GLOBS))),
        DOCS_EXTS=tuple(_get(normalized, "DOCS_EXTS", list(AppConfig.DOCS_EXTS))),
        MODEL=model,
        RAG=rag,
    )
