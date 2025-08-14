# -*- coding: utf-8 -*-
"""
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/14
@version: 0.9.0
"""

import pytest

from core.memory.gate import MemoryGate
from core.memory.types import MemoryCard


@pytest.fixture
def memory_gate():
    return MemoryGate()


def test_is_stable_fact_positive(memory_gate):
    assert memory_gate._is_stable_fact("This is a stable fact about something important.")
    assert memory_gate._is_stable_fact("Another stable fact with enough words.")


def test_is_stable_fact_negative(memory_gate):
    assert not memory_gate._is_stable_fact("Too short.")
    assert not memory_gate._is_stable_fact("Is this a question?")
    assert not memory_gate._is_stable_fact("Short?")


def test_deduplicate_and_merge_no_duplicate(memory_gate):
    new_card = MemoryCard(id="1", content="Unique fact", metadata={})
    existing_cards = [MemoryCard(id="2", content="Another fact", metadata={})]
    result = memory_gate._deduplicate_and_merge(new_card, existing_cards)
    assert result == new_card


def test_deduplicate_and_merge_with_duplicate(memory_gate):
    existing_card = MemoryCard(id="1", content="Duplicate fact", metadata={})
    new_card = MemoryCard(id="2", content="Duplicate fact", metadata={})
    existing_cards = [existing_card]
    result = memory_gate._deduplicate_and_merge(new_card, existing_cards)
    assert result == existing_card


def test_process_new_memory_stable_and_unique(memory_gate):
    new_memory = MemoryCard(id="1", content="A new stable fact to store.", metadata={})
    existing_memories = []
    processed_memories = memory_gate.process_new_memory(new_memory, existing_memories)
    assert len(processed_memories) == 1
    assert processed_memories[0] == new_memory


def test_process_new_memory_unstable(memory_gate):
    new_memory = MemoryCard(id="1", content="Short?", metadata={})
    existing_memories = []
    processed_memories = memory_gate.process_new_memory(new_memory, existing_memories)
    assert len(processed_memories) == 0


def test_process_new_memory_duplicate(memory_gate):
    existing_memory = MemoryCard(id="1", content="This is an existing stable fact.", metadata={})
    new_memory = MemoryCard(id="2", content="This is an existing stable fact.", metadata={})
    existing_memories = [existing_memory]
    processed_memories = memory_gate.process_new_memory(new_memory, existing_memories)
    assert len(processed_memories) == 1
    assert processed_memories[0] == existing_memory  # Should return the existing one
