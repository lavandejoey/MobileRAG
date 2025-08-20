# -*- coding: utf-8 -*-
"""
@file: core/generator/budget.py
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/14
@version: 0.10.0
"""

from typing import Any, Dict, List, Optional

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
        chat_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Build a prompt budget within the model context window.

        Returns a dict with:
        - summary: str
        - recent_messages: List[str] (message contents only)
        - memories: List[str] (memory contents)
        - evidence: List[str] (evidence texts)
        - total_tokens: int
        """
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
        recent_messages = (
            chat_history
            if chat_history is not None
            else self.history_store.load_messages(session_id, limit=self.recent_message_limit)
        )
        current_recent_tokens = 0
        final_recent_contents: List[str] = []
        for msg in reversed(recent_messages):  # Add from most recent backwards
            # Prefer provided token_count if present
            msg_content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
            msg_tokens = (
                msg.get("token_count", count_tokens(msg_content))
                if isinstance(msg, dict)
                else count_tokens(msg_content)
            )
            if current_recent_tokens + msg_tokens <= remaining_budget:
                final_recent_contents.insert(0, msg_content)  # maintain order
                current_recent_tokens += msg_tokens
            else:
                break
        budget_info["recent_messages"] = final_recent_contents
        budget_info["total_tokens"] += current_recent_tokens
        remaining_budget = self.model_context_window - budget_info["total_tokens"]

        # 4. Add memories
        current_memory_tokens = 0
        memory_contents: List[str] = []
        for mem_result in retrieved_memories:
            mem_content = mem_result.memory_card.content
            mem_tokens = count_tokens(mem_content)
            if current_memory_tokens + mem_tokens <= min(remaining_budget, self.memory_token_limit):
                memory_contents.append(mem_content)
                current_memory_tokens += mem_tokens
            else:
                break
        budget_info["memories"] = memory_contents
        budget_info["total_tokens"] += current_memory_tokens
        remaining_budget = self.model_context_window - budget_info["total_tokens"]

        # 5. Add evidence (reranked docs)
        current_evidence_tokens = 0
        evidence_contents: List[str] = []
        for evidence_candidate in retrieved_evidence:
            evidence_text = evidence_candidate.text or ""
            evidence_tokens = count_tokens(evidence_text)
            if current_evidence_tokens + evidence_tokens <= min(
                remaining_budget, self.evidence_token_limit
            ):
                evidence_contents.append(evidence_text)
                current_evidence_tokens += evidence_tokens
            else:
                break

        budget_info["evidence"] = evidence_contents
        budget_info["total_tokens"] += current_evidence_tokens

        return budget_info
