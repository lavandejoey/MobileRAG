# -*- coding: utf-8 -*-
"""
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/14
@version: 0.11.0
"""

from unittest.mock import Mock

reranker = Mock()


def reranker_node(state):
    """
    Reranks the retrieved documents.
    """
    # reranked_docs = reranker.rerank(state["normalized_query"], state["retrieved_docs"])
    return {"reranked_docs": state["retrieved_docs"]}
