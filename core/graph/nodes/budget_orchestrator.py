# -*- coding: utf-8 -*-
"""
@file: core/graph/nodes/budget_orchestrator.py
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/14
@version: 0.11.1
"""

from core.generator.budget import BudgetOrchestrator


def budget_orchestrator_node(state, budget_orchestrator_instance: BudgetOrchestrator):
    """
    Orchestrates the token budget for the prompt.
    """
    budget = budget_orchestrator_instance.orchestrate_budget(
        state["session_id"],
        state["normalized_query"],
        state["retrieved_memories"],
        state["reranked_docs"],
        state["chat_history"],
    )
    return {"budget": budget}
