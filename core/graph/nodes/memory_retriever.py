# -*- coding: utf-8 -*-
"""
@file: core/graph/nodes/memory_retriever.py
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/14
@version: 0.11.0
"""


def memory_retriever_node(state, memory_store):
    """
    Retrieves memories from the memory store.
    """
    memories = memory_store.retrieve_memories(state["normalized_query"])
    return {"retrieved_memories": memories}
