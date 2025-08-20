# -*- coding: utf-8 -*-
"""
@file: core/graph/topology.py
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/14
@version: 0.11.0
"""

from typing import List, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph

from core.config.devices import resolve_devices
from core.config.settings import Settings
from core.generator.budget import BudgetOrchestrator
from core.generator.formatter import AnswerFormatter
from core.generator.llm import LLMGenerator
from core.graph.nodes.budget_orchestrator import budget_orchestrator_node
from core.graph.nodes.device_resolver import device_resolver_node
from core.graph.nodes.generator import generator_node
from core.graph.nodes.history_loader import history_loader_node
from core.graph.nodes.history_saver import history_saver_node  # New import
from core.graph.nodes.memory_retriever import memory_retriever_node
from core.graph.nodes.query_normaliser import query_normaliser_node
from core.graph.nodes.reranker import reranker_node
from core.graph.nodes.retriever_hybrid import retriever_hybrid_node
from core.history.compactor import HistoryCompactor
from core.history.store import ChatHistoryStore
from core.ingest.caption import ImageCaptioner
from core.ingest.embed_dense import DenseEmbedder
from core.ingest.embed_image import ImageEmbedder
from core.ingest.embed_sparse import SparseEmbedder
from core.memory.store import MemoryStore
from core.memory.types import MemoryCard
from core.reranker.reranker import Reranker
from core.retriever.hybrid import HybridRetriever
from core.retriever.types import Candidate
from core.vecdb.client import VecDB

# Persistent, file-backed history store (reused across graphs/requests)
chat_history_store = ChatHistoryStore(db_path="data/chat_history.db")


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
    settings = Settings()
    resolved_devices = resolve_devices()

    # Initialize core components
    vecdb = VecDB(settings)  # Use in-memory for graph testing
    vecdb.create_collections()

    dense_embedder = DenseEmbedder(str(resolved_devices["embed"]))
    image_embedder = ImageEmbedder(str(resolved_devices["embed"]))
    image_captioner = ImageCaptioner(str(resolved_devices["embed"]))
    sparse_embedder = SparseEmbedder()
    reranker_instance = Reranker(str(resolved_devices["reranker"]))

    hybrid_retriever = HybridRetriever(
        settings,
        vecdb,
        dense_embedder,
        image_embedder,
        image_captioner,
        sparse_embedder,
    )

    global chat_history_store
    history_compactor = HistoryCompactor()
    memory_store = MemoryStore(
        settings,
        vecdb,
        dense_embedder,
        sparse_embedder,
        image_embedder,
    )

    budget_orchestrator_instance = BudgetOrchestrator(
        model_context_window=8192,  # Example value, should come from settings
        summary_token_limit=2048,
        recent_message_limit=10,
        memory_token_limit=1024,
        evidence_token_limit=4096,
        history_store=chat_history_store,
        history_compactor=history_compactor,
        memory_store=memory_store,
    )

    llm_generator = LLMGenerator(settings)

    workflow = StateGraph(GraphState)

    # Add the nodes, passing dependencies
    workflow.add_node(
        "device_resolver", lambda state: device_resolver_node(state, resolved_devices)
    )
    workflow.add_node("query_normaliser", query_normaliser_node)
    workflow.add_node(
        "history_loader", lambda state: history_loader_node(state, chat_history_store)
    )
    workflow.add_node("memory_retriever", lambda state: memory_retriever_node(state, memory_store))
    workflow.add_node(
        "retriever_hybrid", lambda state: retriever_hybrid_node(state, hybrid_retriever)
    )
    workflow.add_node("reranker", lambda state: reranker_node(state, reranker_instance))
    workflow.add_node(
        "budget_orchestrator",
        lambda state: budget_orchestrator_node(state, budget_orchestrator_instance),
    )
    workflow.add_node("generator", lambda state: generator_node(state, llm_generator))
    workflow.add_node("answer", answer_node)
    workflow.add_node("history_saver", lambda state: history_saver_node(state, chat_history_store))

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
    workflow.add_edge("answer", "history_saver")  # New edge
    workflow.add_edge("history_saver", END)  # New edge

    # Compile the graph
    graph = workflow.compile()
    return graph


answer_formatter = AnswerFormatter()


def answer_node(state):
    """
    Formats the answer with citations and includes evidence.
    """
    answer = answer_formatter.format_answer_with_citations(
        state["generation"],
        state["reranked_docs"],
    )
    # Extract relevant evidence information for the UI
    evidence_for_ui = [
        {
            "file_path": doc.evidence.file_path,
            "page": doc.evidence.page,
            "caption": doc.evidence.caption,
            "text": doc.text,  # Include the text of the evidence for display
        }
        for doc in state["reranked_docs"]
    ]
    return {"answer": answer, "evidence": evidence_for_ui}
