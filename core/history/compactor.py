# -*- coding: utf-8 -*-
"""
@file: core/history/compactor.py
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/14
@version: 0.8.0
"""

from typing import Any, Dict, List

from core.types import Message
from core.utils.tokens import count_tokens


class HistoryCompactor:
    def __init__(
        self,
        summary_token_limit: int = 600,
        trigger_token_limit: int = 1800,
        trigger_turn_count: int = 6,
    ):
        self.summary_token_limit = summary_token_limit
        self.trigger_token_limit = trigger_token_limit
        self.trigger_turn_count = trigger_turn_count

    def _summarize_messages(self, messages: List[Message]) -> str:
        # This is a placeholder for an actual LLM call for summarization.
        # In a real scenario, you would call an LLM here to generate a concise summary.
        # For now, we'll just concatenate the messages with a note.
        concatenated_content = " ".join([msg.content for msg in messages])
        return (
            f"[SUMMARY: This is a placeholder summary of the following conversation: "
            f"{concatenated_content[:200]}...]"
        )

    def compact_history(self, messages: List[Message]) -> Dict[str, Any]:
        current_token_count = sum(msg.token_count for msg in messages)
        current_turn_count = len(messages)

        if (
            current_token_count > self.trigger_token_limit
            or current_turn_count >= self.trigger_turn_count
        ):
            # Determine which messages to summarize. For simplicity, summarize all for now.
            # In a more advanced implementation, you might summarize older turns
            # while keeping recent ones verbatim.
            summary_content = self._summarize_messages(messages)
            summary_token_count = count_tokens(summary_content)

            # Ensure summary doesn't exceed its limit
            while summary_token_count > self.summary_token_limit:
                # This is a very naive way to shorten. A real LLM would handle this better.
                summary_content = (
                    summary_content[
                        : len(summary_content) * self.summary_token_limit // summary_token_count - 5
                    ]
                    + "..."
                )
                summary_token_count = count_tokens(summary_content)

            return {
                "summary": summary_content,
                "summary_token_count": summary_token_count,
                "compacted": True,
            }
        return {"compacted": False}
