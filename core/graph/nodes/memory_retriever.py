# -*- coding: utf-8 -*-
"""
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/14
@version: 0.11.0
"""

from unittest.mock import Mock

memory_store = Mock()


def memory_retriever_node(state):
    """
    Retrieves memories from the memory store.
    """
    # memories = memory_store.search(state["normalized_query"])
    return {"retrieved_memories": []}
