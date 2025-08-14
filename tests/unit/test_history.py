# -*- coding: utf-8 -*-
"""
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/14
@version: 0.8.0
"""

import os

import pytest

from core.history.compactor import HistoryCompactor
from core.history.store import ChatHistoryStore
from core.types import Message
from core.utils.tokens import count_tokens


@pytest.fixture
def chat_history_store(tmp_path):
    db_path = tmp_path / "test_chat_history.db"
    store = ChatHistoryStore(str(db_path))
    yield store
    os.remove(str(db_path))


@pytest.fixture
def history_compactor():
    return HistoryCompactor(summary_token_limit=100, trigger_token_limit=200, trigger_turn_count=3)


def test_append_and_load_messages(chat_history_store):
    session_id = "test_session_1"
    chat_history_store.append_message(session_id, 1, "user", "Hello", count_tokens("Hello"))
    chat_history_store.append_message(
        session_id, 2, "assistant", "Hi there!", count_tokens("Hi there!")
    )

    messages = chat_history_store.load_messages(session_id)
    assert len(messages) == 2
    assert messages[0]["content"] == "Hello"
    assert messages[1]["role"] == "assistant"


def test_load_messages_limit(chat_history_store):
    session_id = "test_session_2"
    chat_history_store.append_message(session_id, 1, "user", "Msg 1", count_tokens("Msg 1"))
    chat_history_store.append_message(session_id, 2, "assistant", "Msg 2", count_tokens("Msg 2"))
    chat_history_store.append_message(session_id, 3, "user", "Msg 3", count_tokens("Msg 3"))

    messages = chat_history_store.load_messages(session_id, limit=2)
    assert len(messages) == 2
    assert messages[0]["content"] == "Msg 1"
    assert messages[1]["content"] == "Msg 2"


def test_get_last_turn_id(chat_history_store):
    session_id = "test_session_3"
    assert chat_history_store.get_last_turn_id(session_id) is None
    chat_history_store.append_message(session_id, 1, "user", "First", count_tokens("First"))
    assert chat_history_store.get_last_turn_id(session_id) == 1
    chat_history_store.append_message(session_id, 5, "assistant", "Fifth", count_tokens("Fifth"))
    assert chat_history_store.get_last_turn_id(session_id) == 5


def test_save_and_load_summary(chat_history_store):
    session_id = "test_session_4"
    summary_content = "This is a test summary."
    summary_token_count = count_tokens(summary_content)
    last_turn_id = 5

    chat_history_store.save_summary(session_id, summary_content, summary_token_count, last_turn_id)
    summary = chat_history_store.load_summary(session_id)

    assert summary is not None
    assert summary["summary"] == summary_content
    assert summary["token_count"] == summary_token_count
    assert summary["last_turn_id"] == last_turn_id

    # Test update
    new_summary_content = "This is an updated summary."
    new_summary_token_count = count_tokens(new_summary_content)
    new_last_turn_id = 10
    chat_history_store.save_summary(
        session_id, new_summary_content, new_summary_token_count, new_last_turn_id
    )
    updated_summary = chat_history_store.load_summary(session_id)
    assert updated_summary["summary"] == new_summary_content
    assert updated_summary["last_turn_id"] == new_last_turn_id


def test_delete_session_history(chat_history_store):
    session_id = "test_session_5"
    chat_history_store.append_message(session_id, 1, "user", "Msg 1", count_tokens("Msg 1"))
    chat_history_store.save_summary(session_id, "Summary", count_tokens("Summary"), 1)

    chat_history_store.delete_session_history(session_id)

    assert len(chat_history_store.load_messages(session_id)) == 0
    assert chat_history_store.load_summary(session_id) is None


# --- HistoryCompactor Tests ---


def test_compactor_no_compaction_needed(history_compactor):
    messages = [
        Message(role="user", content="short msg", token_count=count_tokens("short msg")),
        Message(role="assistant", content="reply", token_count=count_tokens("reply")),
    ]
    result = history_compactor.compact_history(messages)
    assert not result["compacted"]


def test_compactor_compaction_by_token_limit(history_compactor):
    long_text = "a " * 200  # ~200 tokens
    messages = [
        Message(role="user", content=long_text, token_count=count_tokens(long_text)),
        Message(role="assistant", content="reply", token_count=count_tokens("reply")),
    ]
    result = history_compactor.compact_history(messages)
    assert result["compacted"]
    assert "SUMMARY" in result["summary"]
    assert result["summary_token_count"] <= history_compactor.summary_token_limit


def test_compactor_compaction_by_turn_count(history_compactor):
    messages = [
        Message(role="user", content="1", token_count=count_tokens("1")),
        Message(role="assistant", content="2", token_count=count_tokens("2")),
        Message(role="user", content="3", token_count=count_tokens("3")),
        Message(role="assistant", content="4", token_count=count_tokens("4")),
    ]
    result = history_compactor.compact_history(messages)
    assert result["compacted"]
    assert "SUMMARY" in result["summary"]
    assert result["summary_token_count"] <= history_compactor.summary_token_limit


def test_compactor_summary_token_limit_adherence(history_compactor):
    very_long_text = "b " * 500  # Will exceed summary_token_limit
    messages = [
        Message(role="user", content=very_long_text, token_count=count_tokens(very_long_text)),
    ]
    result = history_compactor.compact_history(messages)
    assert result["compacted"]
    assert result["summary_token_count"] <= history_compactor.summary_token_limit
    assert "..." in result["summary"]  # Check if truncation happened
