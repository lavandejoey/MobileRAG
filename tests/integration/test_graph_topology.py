# -*- coding: utf-8 -*-
"""
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/14
@version: 0.11.0
"""

from unittest.mock import patch

from core.graph.topology import create_graph


@patch("core.graph.nodes.device_resolver.resolve_devices")
@patch("core.graph.nodes.query_normaliser.query_normaliser_node")
@patch("core.graph.nodes.history_loader.history_loader_node")
@patch("core.graph.nodes.memory_retriever.memory_retriever_node")
@patch("core.graph.nodes.retriever_hybrid.retriever_hybrid_node")
@patch("core.graph.nodes.reranker.reranker_node")
@patch("core.graph.nodes.budget_orchestrator.budget_orchestrator_node")
@patch("core.graph.nodes.generator.generator_node")
@patch("core.graph.topology.answer_node")
def test_create_graph(
    mock_answer_node,
    mock_generator_node,
    mock_budget_orchestrator_node,
    mock_reranker_node,
    mock_retriever_hybrid_node,
    mock_memory_retriever_node,
    mock_history_loader_node,
    mock_query_normaliser_node,
    mock_resolve_devices,
):
    """
    Tests that the graph can be created and compiled.
    """
    # Mock the return values of the nodes
    mock_resolve_devices.return_value = {"llm": "cpu"}
    mock_query_normaliser_node.return_value = {"normalized_query": "normalized query"}
    mock_history_loader_node.return_value = {"chat_history": []}
    mock_memory_retriever_node.return_value = {"retrieved_memories": []}
    mock_retriever_hybrid_node.return_value = {"retrieved_docs": []}
    mock_reranker_node.return_value = {"reranked_docs": []}
    mock_budget_orchestrator_node.return_value = {"budget": {}}
    mock_generator_node.return_value = {"generation": "generation"}
    mock_answer_node.return_value = {"answer": "answer"}

    # Create the graph
    graph = create_graph()

    # Run the graph
    inputs = {"query": "test query", "session_id": "test_session"}
    result = graph.invoke(inputs)

    # Assert that the final result is as expected
    assert result["answer"] == "answer"
