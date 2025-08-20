# -*- coding: utf-8 -*-
"""
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/14
@version: 0.6.0
"""

from unittest.mock import Mock

import pytest

from core.config.settings import NamedVectorConfig, Settings, VectorStoreConfig
from core.ingest.caption import ImageCaptioner
from core.ingest.embed_dense import DenseEmbedder
from core.ingest.embed_image import ImageEmbedder
from core.ingest.embed_sparse import SparseEmbedder
from core.retriever.hybrid import HybridRetriever
from core.retriever.types import HybridQuery
from core.vecdb.client import VecDB


@pytest.fixture
def mock_settings():
    return Settings(
        vectorstore=VectorStoreConfig(
            local_path="./mock_qdrant_db",
            collection="rag_multimodal",
            named_vectors={
                "text_dense": NamedVectorConfig(size=1024, distance="cosine", name="text_dense"),
                "image": NamedVectorConfig(size=512, distance="cosine", name="image"),
                "text_sparse": NamedVectorConfig(sparse=True, name="text_sparse"),
            },
        )
    )


@pytest.fixture
def mock_vecdb_client():
    mock_client = Mock()
    # Mock the query_points method to return predefined results
    mock_client.query_points.return_value = [
        Mock(
            id="doc1#chunk0#100",
            score=0.9,
            payload={
                "file_path": "/path/to/doc1.txt",
                "page": 1,
                "lang": "en",
                "modality": "text",
                "content": "This is a test document.",
            },
        ),
        Mock(
            id="img1#chunk0#200",
            score=0.8,
            payload={
                "file_path": "/path/to/img1.png",
                "caption": "A black image.",
                "lang": "en",
                "modality": "image",
            },
        ),
    ]
    return mock_client


@pytest.fixture
def mock_vecdb(mock_settings, mock_vecdb_client):
    vecdb = VecDB(mock_settings)
    vecdb.client = mock_vecdb_client  # Replace real client with mock
    return vecdb


@pytest.fixture
def mock_dense_embedder():
    mock_embedder = Mock(spec=DenseEmbedder)
    mock_embedder.embed_text_query.return_value = [0.1] * 1024  # Dummy dense embedding
    return mock_embedder


@pytest.fixture
def mock_image_embedder():
    mock_embedder = Mock(spec=ImageEmbedder)
    mock_embedder.embed_image_query.return_value = [0.2] * 512  # Dummy image embedding
    return mock_embedder


@pytest.fixture
def mock_image_captioner():
    mock_captioner = Mock(spec=ImageCaptioner)
    mock_captioner.caption_images.return_value = ["A dummy caption."]  # Dummy caption
    return mock_captioner


@pytest.fixture
def mock_sparse_embedder():
    mock_embedder = Mock(spec=SparseEmbedder)
    mock_embedder.embed_sparse.return_value = [
        {"indices": [1, 2], "values": [0.1, 0.2]}
    ]  # Dummy sparse embedding
    return mock_embedder


@pytest.fixture
def hybrid_retriever(
    mock_settings,
    mock_vecdb,
    mock_dense_embedder,
    mock_image_embedder,
    mock_image_captioner,
    mock_sparse_embedder,
):
    return HybridRetriever(
        mock_settings,
        mock_vecdb,
        mock_dense_embedder,
        mock_image_embedder,
        mock_image_captioner,
        mock_sparse_embedder,
    )


def test_hybrid_retriever_text_query(hybrid_retriever, mock_vecdb_client):
    query = HybridQuery(text="test query")
    candidates = hybrid_retriever.search(query)

    # Assertions for search results
    mock_vecdb_client.query_points.assert_called_once()  # Ensure query_points was called
    assert len(candidates) == 2  # Expecting 2 candidates from mock

    # Check first candidate (text document)
    assert candidates[0].id == "doc1#chunk0#100"
    assert candidates[0].score == 0.9
    assert candidates[0].text == "This is a test document."
    assert candidates[0].evidence.file_path == "/path/to/doc1.txt"
    assert candidates[0].evidence.page == 1
    assert candidates[0].lang == "en"
    assert candidates[0].modality == "text"


def test_hybrid_retriever_image_query(hybrid_retriever, mock_vecdb_client):
    query = HybridQuery(image_path="/path/to/query_image.png")
    candidates = hybrid_retriever.search(query)

    # Assertions for search results
    mock_vecdb_client.query_points.assert_called_once()  # Ensure query_points was called
    assert len(candidates) == 2  # Expecting 2 candidates from mock

    # Check first candidate (text document)
    assert candidates[0].id == "doc1#chunk0#100"
    assert candidates[0].score == 0.9
    assert candidates[0].text == "This is a test document."
    assert candidates[0].evidence.file_path == "/path/to/doc1.txt"
    assert candidates[0].evidence.page == 1
    assert candidates[0].lang == "en"
    assert candidates[0].modality == "text"
