# -*- coding: utf-8 -*-
"""
@file: core/sparse/fastembed.py
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/13
@version: 0.4.0
"""

from typing import Any, Dict, List

from fastembed import SparseTextEmbedding


def build_sparse_vectors(texts: List[str]) -> List[Dict[str, Any]]:
    """
    Builds sparse vectors using FastEmbed and serializes them to Qdrant sparse format.
    """
    model = SparseTextEmbedding(model_name="Qdrant/bm25")
    embeddings = model.embed(texts)

    qdrant_sparse_vectors: List[Dict[str, Any]] = []
    for emb in embeddings:
        # FastEmbed returns a dictionary with 'indices' and 'values'
        # This is already compatible with Qdrant's sparse vector format
        qdrant_sparse_vectors.append(
            {"indices": emb.indices.tolist(), "values": emb.values.tolist()}
        )
    return qdrant_sparse_vectors
