# -*- coding: utf-8 -*-
"""
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/14
@version: 0.10.0
"""

from typing import Any, Dict, List

from core.history.compactor import HistoryCompactor
from core.history.store import ChatHistoryStore
from core.memory.store import MemoryStore
from core.memory.types import QueryResult
from core.retriever.types import Candidate
from core.utils.tokens import count_tokens


class BudgetOrchestrator:
    def __init__(
        self,
        model_context_window: int,
        summary_token_limit: int,
        recent_message_limit: int,
        memory_token_limit: int,
        evidence_token_limit: int,
        history_store: ChatHistoryStore,
        history_compactor: HistoryCompactor,
        memory_store: MemoryStore,
    ):
        self.model_context_window = model_context_window
        self.summary_token_limit = summary_token_limit
        self.recent_message_limit = recent_message_limit
        self.memory_token_limit = memory_token_limit
        self.evidence_token_limit = evidence_token_limit
        self.history_store = history_store
        self.history_compactor = history_compactor
        self.memory_store = memory_store

    def orchestrate_budget(
        self,
        session_id: str,
        query: str,
        retrieved_memories: List[QueryResult],
        retrieved_evidence: List[Candidate],
    ) -> Dict[str, Any]:
        budget_info = {
            "summary": "",
            "recent_messages": [],
            "memories": [],
            "evidence": [],
            "total_tokens": 0,
        }

        # 1. Add query tokens
        query_tokens = count_tokens(query)
        budget_info["total_tokens"] += query_tokens

        # 2. Add summary (if exists and fits)
        summary_data = self.history_store.load_summary(session_id)
        if summary_data and count_tokens(summary_data["summary"]) <= self.summary_token_limit:
            budget_info["summary"] = summary_data["summary"]
            budget_info["total_tokens"] += count_tokens(summary_data["summary"])

        # Calculate remaining budget after query and summary
        remaining_budget = self.model_context_window - budget_info["total_tokens"]

        # 3. Add recent messages
        recent_messages = self.history_store.load_messages(
            session_id, limit=self.recent_message_limit
        )
        current_recent_tokens = 0
        for msg in reversed(recent_messages):  # Add from most recent backwards
            msg_tokens = count_tokens(f"{msg['role']}: {msg['content']}")
            if current_recent_tokens + msg_tokens <= remaining_budget:
                budget_info["recent_messages"].insert(0, msg)  # Add to front to maintain order
                current_recent_tokens += msg_tokens
            else:
                break
        budget_info["total_tokens"] += current_recent_tokens
        remaining_budget = self.model_context_window - budget_info["total_tokens"]

        # 4. Add memories
        current_memory_tokens = 0
        for mem_result in retrieved_memories:
            mem_content = mem_result.memory_card.content
            mem_tokens = count_tokens(mem_content)
            if current_memory_tokens + mem_tokens <= min(remaining_budget, self.memory_token_limit):
                budget_info["memories"].append(mem_result.memory_card)
                current_memory_tokens += mem_tokens
            else:
                break
        budget_info["total_tokens"] += current_memory_tokens
        remaining_budget = self.model_context_window - budget_info["total_tokens"]

        # 5. Add evidence
        current_evidence_tokens = 0
        for evidence_candidate in retrieved_evidence:
            evidence_text = (
                evidence_candidate.text
            )  # Assuming text is the main content for evidence
            evidence_tokens = count_tokens(evidence_text)
            if current_evidence_tokens + evidence_tokens <= min(
                remaining_budget, self.evidence_token_limit
            ):
                budget_info["evidence"].append(evidence_candidate)
                current_evidence_tokens += evidence_tokens
            else:
                break
        budget_info["total_tokens"] += current_evidence_tokens

        return budget_info
