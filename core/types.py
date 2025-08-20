# -*- coding: utf-8 -*-
"""
@file: core/types.py
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/13
@version: 0.3.0
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional


@dataclass
class IngestItem:
    path: str
    doc_id: str
    modality: Literal["text", "image"]
    meta: Dict[str, Any]


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    content: str
    lang: str
    meta: Dict[str, Any]
    # Optional fields for text/PDF
    page: Optional[int] = None
    bbox: Optional[tuple[int, int, int, int]] = None
    # Optional fields for vectors and metadata
    dense_vector: Optional[List[float]] = None
    sparse_vector: Optional[Dict[str, Any]] = None
    image_vector: Optional[List[float]] = None
    modality: Optional[str] = None
    caption: Optional[str] = None


@dataclass
class Message:
    role: str
    content: str
    token_count: int
