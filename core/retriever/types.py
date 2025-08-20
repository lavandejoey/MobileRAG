# -*- coding: utf-8 -*-
"""
@file: core/retriever/types.py
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/14
@version: 0.6.0
"""

from typing import Any, Dict, Optional, Tuple

from pydantic import BaseModel


class Evidence(BaseModel):
    file_path: str
    page: Optional[int] = None
    bbox: Optional[Tuple[int, int, int, int]] = None
    caption: Optional[str] = None
    title: Optional[str] = None


class Candidate(BaseModel):
    id: str
    score: float
    text: Optional[str] = None
    evidence: Evidence
    lang: str
    modality: str


class HybridQuery(BaseModel):
    text: Optional[str] = None
    image_path: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None
    topk_dense: int = 60
    topk_sparse: int = 60
