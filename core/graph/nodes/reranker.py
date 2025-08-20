# -*- coding: utf-8 -*-
"""
@file: core/graph/nodes/reranker.py
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/14
@version: 0.11.0
"""


def reranker_node(state, reranker_instance):
    """
    Reranks the retrieved documents.
    """
    reranked_docs = reranker_instance.rank(state["normalized_query"], state["retrieved_docs"])
    return {"reranked_docs": reranked_docs}
