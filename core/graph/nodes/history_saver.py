# -*- coding: utf-8 -*-
"""
@file: core/graph/nodes/history_saver.py
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/15
@version: 0.13.0
"""

from typing import Any, Dict

from core.history.store import ChatHistoryStore
from core.utils.tokens import count_tokens


def history_saver_node(state: Dict[str, Any], chat_history_store: ChatHistoryStore):
    """
    Saves the chat history, including the user's query and the assistant's response with evidence.
    """
    session_id = state["session_id"]
    user_query = state["query"]
    assistant_answer = state["answer"]
    evidence = state.get("evidence", None)

    # Get the last turn ID and increment for the new turn
    last_turn_id = chat_history_store.get_last_turn_id(session_id)
    new_turn_id = (last_turn_id or 0) + 1

    # Save user's message
    chat_history_store.append_message(
        session_id=session_id,
        turn_id=new_turn_id,
        role="user",
        content=user_query,
        token_count=count_tokens(user_query),
        evidence=None,  # User messages don't have evidence
    )

    # Save assistant's message with evidence
    chat_history_store.append_message(
        session_id=session_id,
        turn_id=new_turn_id + 1,  # Increment turn_id for assistant's response
        role="assistant",
        content=assistant_answer,
        token_count=count_tokens(assistant_answer),
        evidence=evidence,
    )

    return {}
