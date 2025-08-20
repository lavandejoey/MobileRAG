# -*- coding: utf-8 -*-
"""
@file: core/ingest/embed_sparse.py
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/13
@version: 0.4.0
"""

from typing import Any, Dict, List, Union

from core.sparse.fastembed import build_sparse_vectors
from core.types import Chunk


class SparseEmbedder:
    def embed_sparse(self, items: List[Union[Chunk, str]]) -> List[Dict[str, Any]]:
        if not items:
            return []
        if isinstance(items[0], Chunk):
            texts = [item.content for item in items]
        else:
            texts = items
        return build_sparse_vectors(texts)
