# -*- coding: utf-8 -*-
"""
@file: core/graph/nodes/history_loader.py
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/14
@version: 0.11.0
"""

# from core.history.store import ChatHistoryStore as HistoryStore


def history_loader_node(state, chat_history_store):
    """
    Loads the chat history.
    """
    chat_history = chat_history_store.load_messages(state["session_id"])
    return {"chat_history": chat_history}
