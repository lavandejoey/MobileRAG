# -*- coding: utf-8 -*-
"""
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/14
@version: 0.9.0
"""

import uuid
from unittest.mock import Mock

import pytest

from core.config.settings import NamedVectorConfig, Settings, VectorStoreConfig
from core.ingest.embed_dense import DenseEmbedder
from core.ingest.embed_image import ImageEmbedder
from core.ingest.embed_sparse import SparseEmbedder
from core.memory.store import MemoryStore
from core.memory.types import MemoryCard, QueryResult
from core.vecdb.client import VecDB


@pytest.fixture
def mock_settings():
    return Settings(
        vectorstore=VectorStoreConfig(
            path="./mock_qdrant_db_memory",  # Use a different path for memory store
            collection="agent_memory",
            named_vectors={
                "text_dense": NamedVectorConfig(size=1024, distance="cosine", name="text_dense"),
                "text_sparse": NamedVectorConfig(sparse=True, name="text_sparse"),
            },
        ),
        collection_mem="agent_memory",  # Ensure this matches the collection name
    )


@pytest.fixture
def mock_vecdb_client():
    mock_client = Mock()
    # Mock the search method to return predefined results
    mock_client.search.return_value = [
        Mock(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, "fact1")),
            score=0.9,
            payload={
                "id": "fact1",
                "content": "The sky is blue",
                "metadata": {"source": "observation"},
            },
        ),
        Mock(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, "fact2")),
            score=0.8,
            payload={
                "id": "fact2",
                "content": "Water is wet",
                "metadata": {"source": "observation"},
            },
        ),
    ]
    mock_client.upsert.return_value = None
    mock_client.delete.return_value = None
    mock_client.retrieve.return_value = [
        Mock(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, "fact1")),
            payload={
                "id": "fact1",
                "content": "The sky is blue",
                "metadata": {"source": "observation"},
            },
        )
    ]
    return mock_client


@pytest.fixture
def mock_vecdb(mock_settings, mock_vecdb_client):
    vecdb = VecDB(mock_settings)
    vecdb.client = mock_vecdb_client  # Replace real client with mock
    vecdb.create_collections = Mock()  # Mock create_collections
    vecdb.close = Mock()  # Mock close method
    return vecdb


@pytest.fixture
def mock_dense_embedder():
    mock_embedder = Mock(spec=DenseEmbedder)
    mock_embedder.embed_text_query.return_value = [0.1] * 1024  # Dummy dense embedding
    return mock_embedder


@pytest.fixture
def mock_sparse_embedder():
    mock_embedder = Mock(spec=SparseEmbedder)
    mock_embedder.embed_sparse.return_value = [
        {"indices": [1, 2], "values": [0.1, 0.2]}
    ]  # Dummy sparse embedding
    return mock_embedder


@pytest.fixture
def mock_image_embedder():
    mock_embedder = Mock(spec=ImageEmbedder)
    mock_embedder.embed_image_query.return_value = [0.2] * 512  # Dummy image embedding
    return mock_embedder


@pytest.fixture
def memory_store(
    mock_settings, mock_vecdb, mock_dense_embedder, mock_sparse_embedder, mock_image_embedder
):
    return MemoryStore(
        mock_settings, mock_vecdb, mock_dense_embedder, mock_sparse_embedder, mock_image_embedder
    )


def test_upsert_memory(memory_store, mock_vecdb_client):
    memory_card = MemoryCard(
        id="test_id", content="This is a test memory.", metadata={"type": "fact"}
    )
    memory_store.upsert_memory(memory_card)
    mock_vecdb_client.upsert.assert_called_once()


def test_retrieve_memories(memory_store, mock_vecdb_client):
    query_text = "What is the color of the sky?"
    results = memory_store.retrieve_memories(query_text)
    mock_vecdb_client.search.assert_called_once()
    assert len(results) == 2
    assert isinstance(results[0], QueryResult)
    assert results[0].memory_card.id == "fact1"
    assert results[0].memory_card.content == "The sky is blue"
    assert results[0].score == 0.9


def test_delete_memory(memory_store, mock_vecdb_client):
    memory_id = "test_id_to_delete"
    memory_store.delete_memory(memory_id)
    mock_vecdb_client.delete.assert_called_once()


def test_get_memory(memory_store, mock_vecdb_client):
    memory_id = "fact1"
    retrieved_card = memory_store.get_memory(memory_id)
    mock_vecdb_client.retrieve.assert_called_once()
    assert retrieved_card is not None
    assert retrieved_card.id == "fact1"
    assert retrieved_card.content == "The sky is blue"
