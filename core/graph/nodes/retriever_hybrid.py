# -*- coding: utf-8 -*-
"""
@file: core/graph/nodes/retriever_hybrid.py
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/14
@version: 0.11.0
"""

from core.retriever.types import HybridQuery


def retriever_hybrid_node(state, hybrid_retriever):
    """
    Retrieves documents using the hybrid retriever.
    """
    hybrid_query = HybridQuery(
        text=state["normalized_query"],
        image_path=None,  # Assuming no image query for now
        topk_dense=60,
        topk_sparse=60,
    )
    retrieved_docs = hybrid_retriever.search(hybrid_query)
    return {"retrieved_docs": retrieved_docs}
