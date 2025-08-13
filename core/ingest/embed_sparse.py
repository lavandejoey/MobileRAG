# -*- coding: utf-8 -*-
"""
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/13
@version: 0.4.0
"""

from typing import Any, Dict, List

from core.sparse.fastembed import build_sparse_vectors
from core.types import Chunk


class SparseEmbedder:
    def embed_sparse(self, chunks: List[Chunk]) -> List[Dict[str, Any]]:
        texts = [chunk.content for chunk in chunks]
        return build_sparse_vectors(texts)
