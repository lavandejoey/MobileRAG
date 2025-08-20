# -*- coding: utf-8 -*-
"""
@file: core/memory/gate.py
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/14
@version: 0.9.0
"""

from typing import List

from core.memory.types import MemoryCard


class MemoryGate:
    def __init__(self):
        pass

    def _is_stable_fact(self, content: str) -> bool:
        # Placeholder for rule-based or light LLM check
        # For now, a very simple rule: must contain at least 5 words and not be a question.
        words = content.split()
        if len(words) < 5 or content.endswith("?"):
            return False
        return True

    def _deduplicate_and_merge(
        self, new_card: MemoryCard, existing_cards: List[MemoryCard]
    ) -> MemoryCard:
        # Placeholder for deduplication and merging logic
        # For now, if content is exactly the same, consider it a duplicate and return existing.
        for card in existing_cards:
            if card.content == new_card.content:
                return card  # Return existing card if duplicate
        return new_card  # Return new card if no duplicate found

    def _apply_decay(self, memory_cards: List[MemoryCard]) -> List[MemoryCard]:
        # Placeholder for decay logic (e.g., based on recency, usage, etc.)
        # For now, no decay applied.
        return memory_cards

    def process_new_memory(
        self, new_memory: MemoryCard, existing_memories: List[MemoryCard]
    ) -> List[MemoryCard]:
        if not self._is_stable_fact(new_memory.content):
            return []  # Do not store unstable facts

        processed_memory = self._deduplicate_and_merge(new_memory, existing_memories)

        # If the processed memory is the new_memory, it means it's not a duplicate of existing ones
        if processed_memory == new_memory:
            existing_memories.append(processed_memory)

        return self._apply_decay(existing_memories)
