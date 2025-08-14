# -*- coding: utf-8 -*-
"""
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/14
@version: 0.11.0
"""

from typing import Any, Dict, List
from unittest.mock import Mock

from core.history.compactor import HistoryCompactor
from core.history.store import ChatHistoryStore as HistoryStore
from core.memory.store import MemoryStore
from core.memory.types import QueryResult
from core.retriever.types import Candidate
from core.utils.tokens import count_tokens


class BudgetOrchestrator:
    """
    Manages the token budget for the language model prompt, ensuring the total number of tokens
    does not exceed the model's context window.

    It orchestrates the inclusion and trimming of different context components:
    - Chat summary
    - Recent chat messages
    - Retrieved long-term memories
    - Retrieved evidence (from documents)

    The orchestration follows a priority order, trimming the least important information first
    to stay within the allocated token budget.
    """

    def __init__(
        self,
        model_context_window: int,
        summary_token_limit: int,
        recent_message_limit: int,
        memory_token_limit: int,
        evidence_token_limit: int,
        history_store: HistoryStore,
        history_compactor: HistoryCompactor,
        memory_store: MemoryStore,
    ):
        """
        Initializes the BudgetOrchestrator.

        Args:
            model_context_window: The total token limit for the model's context.
            summary_token_limit: The maximum number of tokens allocated for the chat summary.
            recent_message_limit: The maximum number of recent messages to include.
            memory_token_limit: The maximum number of tokens allocated for retrieved memories.
            evidence_token_limit: The maximum number of tokens allocated for retrieved evidence.
            history_store: The store for accessing chat history (summaries and messages).
            history_compactor: The tool for compacting chat history.
            memory_store: The store for accessing long-term memories.
        """
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
        chat_history: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Orchestrates the token budget for the prompt.

        Args:
            session_id: The ID of the current chat session.
            query: The user's query.
            retrieved_memories: A list of retrieved memory cards.
            retrieved_evidence: A list of retrieved evidence candidates.

        Returns:
            A dictionary containing the orchestrated components of the prompt, including:
            - summary: The chat summary.
            - recent_messages: A list of recent messages.
            - memories: A list of memory contents.
            - evidence: A list of evidence contents.
            - total_tokens: The total number of tokens in the orchestrated prompt.
        """
        total_tokens = count_tokens(query)

        # 1. Load and budget for summary
        summary_content = ""
        summary = self.history_store.load_summary(session_id)
        if summary:
            summary_content = summary.get("summary", "")
            summary_tokens = count_tokens(summary_content)
            if summary_tokens > self.summary_token_limit:
                # This case should ideally be handled by the compactor, but as a fallback:
                summary_content = summary_content[: self.summary_token_limit * 4]  # Rough trim
                summary_tokens = count_tokens(summary_content)
            total_tokens += summary_tokens

        # 2. Load recent messages
        recent_messages = chat_history

        # 3. Budget for evidence
        evidence_content = []
        evidence_tokens = 0
        for ev in retrieved_evidence:
            ev_tokens = count_tokens(ev.text)
            if (
                total_tokens + evidence_tokens + ev_tokens <= self.model_context_window
                and evidence_tokens + ev_tokens <= self.evidence_token_limit
            ):
                evidence_content.append(ev.text)
                evidence_tokens += ev_tokens
        total_tokens += evidence_tokens

        # 4. Budget for memories
        memory_content = []
        memory_tokens = 0
        for mem in retrieved_memories:
            mem_tokens = count_tokens(mem.memory_card.content)
            if (
                total_tokens + memory_tokens + mem_tokens <= self.model_context_window
                and memory_tokens + mem_tokens <= self.memory_token_limit
            ):
                memory_content.append(mem.memory_card.content)
                memory_tokens += mem_tokens
        total_tokens += memory_tokens

        # 5. Budget for recent messages (trimming from oldest if necessary)
        final_recent_messages = []
        for msg in reversed(recent_messages):
            msg_tokens = msg.get("token_count", count_tokens(msg.get("content", "")))
            if total_tokens + msg_tokens <= self.model_context_window:
                final_recent_messages.insert(0, msg)
                total_tokens += msg_tokens
            else:
                break  # Stop adding older messages if budget is exceeded

        return {
            "summary": summary_content,
            "recent_messages": [m["content"] for m in final_recent_messages],
            "memories": memory_content,
            "evidence": evidence_content,
            "total_tokens": total_tokens,
        }


history_store = HistoryStore()
history_compactor = HistoryCompactor()
memory_store = Mock()

budget_orchestrator = BudgetOrchestrator(
    model_context_window=8192,
    summary_token_limit=2048,
    recent_message_limit=10,
    memory_token_limit=1024,
    evidence_token_limit=4096,
    history_store=history_store,
    history_compactor=history_compactor,
    memory_store=memory_store,
)


def budget_orchestrator_node(state):
    """
    Orchestrates the token budget for the prompt.
    """
    budget = budget_orchestrator.orchestrate_budget(
        state["session_id"],
        state["normalized_query"],
        state["retrieved_memories"],
        state["reranked_docs"],
        state["chat_history"],
    )
    return {"budget": budget}
