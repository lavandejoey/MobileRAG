# -*- coding: utf-8 -*-
"""
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/14
@version: 0.11.0
"""

from typing import List, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph

from core.generator.formatter import AnswerFormatter
from core.graph.nodes.budget_orchestrator import budget_orchestrator_node
from core.graph.nodes.device_resolver import device_resolver_node
from core.graph.nodes.generator import generator_node
from core.graph.nodes.history_loader import history_loader_node
from core.graph.nodes.memory_retriever import memory_retriever_node
from core.graph.nodes.query_normaliser import query_normaliser_node
from core.graph.nodes.reranker import reranker_node
from core.graph.nodes.retriever_hybrid import retriever_hybrid_node
from core.memory.types import MemoryCard
from core.retriever.types import Candidate


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        query: The user's query.
        session_id: The ID of the current chat session.
        chat_history: The chat history.
        devices: The devices to use for the different models.
        normalized_query: The normalized query.
        retrieved_docs: The documents retrieved from the vector store.
        reranked_docs: The documents reranked by the reranker.
        retrieved_memories: The memories retrieved from the memory store.
        budget: The token budget for the prompt.
        generation: The generated response.
        answer: The final answer.
    """

    query: str
    session_id: str
    chat_history: List[BaseMessage]
    devices: dict
    normalized_query: str
    retrieved_docs: List[Candidate]
    reranked_docs: List[Candidate]
    retrieved_memories: List[MemoryCard]
    budget: dict
    generation: str
    answer: str


def create_graph():
    """
    Creates the LangGraph topology.
    """
    workflow = StateGraph(GraphState)

    # Add the nodes
    workflow.add_node("device_resolver", device_resolver_node)
    workflow.add_node("query_normaliser", query_normaliser_node)
    workflow.add_node("history_loader", history_loader_node)
    workflow.add_node("memory_retriever", memory_retriever_node)
    workflow.add_node("retriever_hybrid", retriever_hybrid_node)
    workflow.add_node("reranker", reranker_node)
    workflow.add_node("budget_orchestrator", budget_orchestrator_node)
    workflow.add_node("generator", generator_node)
    workflow.add_node("answer", answer_node)

    # Set the entrypoint
    workflow.set_entry_point("device_resolver")

    # Add the edges
    workflow.add_edge("device_resolver", "query_normaliser")
    workflow.add_edge("query_normaliser", "history_loader")
    workflow.add_edge("history_loader", "memory_retriever")
    workflow.add_edge("memory_retriever", "retriever_hybrid")
    workflow.add_edge("retriever_hybrid", "reranker")
    workflow.add_edge("reranker", "budget_orchestrator")
    workflow.add_edge("budget_orchestrator", "generator")
    workflow.add_edge("generator", "answer")
    workflow.add_edge("answer", END)

    # Compile the graph
    graph = workflow.compile()
    return graph


answer_formatter = AnswerFormatter()


def answer_node(state):
    """
    Formats the answer with citations.
    """
    answer = answer_formatter.format_answer_with_citations(
        state["generation"],
        state["reranked_docs"],
    )
    return {"answer": answer}
