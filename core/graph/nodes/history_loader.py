# -*- coding: utf-8 -*-
"""
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/14
@version: 0.11.0
"""

from core.history.store import ChatHistoryStore as HistoryStore

history_store = HistoryStore()


def history_loader_node(state):
    """
    Loads the chat history.
    """
    chat_history = history_store.load_messages(state["session_id"])
    return {"chat_history": chat_history}
