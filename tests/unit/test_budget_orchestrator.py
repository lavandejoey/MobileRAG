# -*- coding: utf-8 -*-
"""
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/14
@version: 0.10.0
"""

from unittest.mock import Mock

import pytest

from core.generator.budget import BudgetOrchestrator
from core.memory.types import MemoryCard, QueryResult
from core.retriever.types import Candidate, Evidence
from core.utils.tokens import count_tokens


@pytest.fixture
def mock_history_store():
    mock_store = Mock()
    mock_store.load_summary.return_value = {
        "summary": "Summary content.",
        "token_count": count_tokens("Summary content."),
    }
    mock_store.load_messages.return_value = [
        {
            "role": "user",
            "content": "User message 1.",
            "token_count": count_tokens("User message 1."),
        },
        {
            "role": "assistant",
            "content": "Assistant message 1.",
            "token_count": count_tokens("Assistant message 1."),
        },
        {
            "role": "user",
            "content": "User message 2.",
            "token_count": count_tokens("User message 2."),
        },
    ]
    return mock_store


@pytest.fixture
def mock_history_compactor():
    return Mock()


@pytest.fixture
def mock_memory_store():
    return Mock()


@pytest.fixture
def budget_orchestrator(mock_history_store, mock_history_compactor, mock_memory_store):
    return BudgetOrchestrator(
        model_context_window=20,
        summary_token_limit=20,
        recent_message_limit=3,
        memory_token_limit=30,
        evidence_token_limit=30,
        history_store=mock_history_store,
        history_compactor=mock_history_compactor,
        memory_store=mock_memory_store,
    )


def test_orchestrate_budget_basic(budget_orchestrator):
    query = "Test query."
    retrieved_memories = [
        QueryResult(
            memory_card=MemoryCard(id="mem1", content="Memory content 1.", metadata={}), score=0.9
        ),
    ]
    retrieved_evidence = [
        Candidate(
            id="ev1",
            score=0.8,
            text="Evidence content 1.",
            evidence=Evidence(file_path="file1.txt"),
            lang="en",
            modality="text",
        ),
    ]

    budget = budget_orchestrator.orchestrate_budget(
        "session1", query, retrieved_memories, retrieved_evidence
    )

    assert budget["summary"] == "Summary content."
    assert len(budget["recent_messages"]) < 3  # Should trim some messages
    assert len(budget["memories"]) == 0
    assert len(budget["evidence"]) == 0
    assert budget["total_tokens"] > 0


def test_orchestrate_budget_exceed_recent_messages(budget_orchestrator, mock_history_store):
    # Make recent messages too long to fit
    mock_history_store.load_messages.return_value = [
        {
            "role": "user",
            "content": "A very long user message that will exceed the budget.",
            "token_count": count_tokens("A very long user message that will exceed the budget."),
        },
        {
            "role": "assistant",
            "content": "A very long assistant message that will exceed the budget.",
            "token_count": count_tokens(
                "A very long assistant message that will exceed the budget."
            ),
        },
    ]
    query = "Short query."
    retrieved_memories = []
    retrieved_evidence = []

    budget = budget_orchestrator.orchestrate_budget(
        "session1", query, retrieved_memories, retrieved_evidence
    )

    assert len(budget["recent_messages"]) < 2  # Should trim some messages
    assert budget["total_tokens"] <= budget_orchestrator.model_context_window


def test_orchestrate_budget_exceed_memory_limit(budget_orchestrator):
    query = "Short query."
    retrieved_memories = [
        QueryResult(
            memory_card=MemoryCard(
                id="mem1", content="Memory content 1. This is a long memory.", metadata={}
            ),
            score=0.9,
        ),
        QueryResult(
            memory_card=MemoryCard(
                id="mem2", content="Memory content 2. This is another long memory.", metadata={}
            ),
            score=0.8,
        ),
    ]
    retrieved_evidence = []

    budget = budget_orchestrator.orchestrate_budget(
        "session1", query, retrieved_memories, retrieved_evidence
    )

    assert len(budget["memories"]) < 2  # Should trim some memories
    assert budget["total_tokens"] <= budget_orchestrator.model_context_window


def test_orchestrate_budget_exceed_evidence_limit(budget_orchestrator):
    query = "Short query."
    retrieved_memories = []
    retrieved_evidence = [
        Candidate(
            id="ev1",
            score=0.8,
            text="Evidence content 1. This is long evidence.",
            evidence=Evidence(file_path="file1.txt"),
            lang="en",
            modality="text",
        ),
        Candidate(
            id="ev2",
            score=0.7,
            text="Evidence content 2. This is another long evidence.",
            evidence=Evidence(file_path="file2.txt"),
            lang="en",
            modality="text",
        ),
    ]

    budget = budget_orchestrator.orchestrate_budget(
        "session1", query, retrieved_memories, retrieved_evidence
    )

    assert len(budget["evidence"]) < 2  # Should trim some evidence
    assert budget["total_tokens"] <= budget_orchestrator.model_context_window
