# -*- coding: utf-8 -*-
"""
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/14
@version: 0.11.0
"""

from unittest.mock import Mock

retriever = Mock()


def retriever_hybrid_node(state):
    """
    Retrieves documents from the vector store.
    """
    # documents = retriever.search(state["normalized_query"])
    return {"retrieved_docs": []}
