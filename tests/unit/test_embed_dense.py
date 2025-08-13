# -*- coding: utf-8 -*-
"""
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/13
@version: 0.4.0
"""

import pytest

from core.config.settings import Settings
from core.ingest import Ingestor
from core.types import Chunk


@pytest.fixture
def settings():
    # Use CPU for testing to avoid GPU dependency
    return Settings(device="cpu")


@pytest.fixture
def ingestor(settings):
    # Use CPU for testing to avoid GPU dependency
    return Ingestor(settings)


def test_dense_embedder_deterministic(ingestor):
    chunks = [
        Chunk(doc_id="doc1", chunk_id="c1", content="hello world", lang="en", meta={}),
        Chunk(doc_id="doc2", chunk_id="c2", content="test sentence", lang="en", meta={}),
    ]

    # Embed twice to check for determinism
    embeddings1 = ingestor.embed_dense(chunks)
    embeddings2 = ingestor.embed_dense(chunks)

    # Check if embeddings are deterministic (within a small tolerance for float precision)
    assert len(embeddings1) == len(chunks)
    assert len(embeddings2) == len(chunks)
    for i in range(len(chunks)):
        for j in range(len(embeddings1[i])):
            assert abs(embeddings1[i][j] - embeddings2[i][j]) < 1e-6


def test_dense_embedder_output_dimension(ingestor):
    chunks = [Chunk(doc_id="doc1", chunk_id="c1", content="single sentence", lang="en", meta={})]
    embeddings = ingestor.embed_dense(chunks)

    # Qwen3-Embedding-0.6B outputs 1024-dim
    assert len(embeddings) == 1
    assert len(embeddings[0]) == 1024


def test_dense_embedder_batch_inference(ingestor):
    chunks = [
        Chunk(doc_id="doc1", chunk_id="c1", content="sentence one", lang="en", meta={}),
        Chunk(doc_id="doc2", chunk_id="c2", content="sentence two", lang="en", meta={}),
        Chunk(doc_id="doc3", chunk_id="c3", content="sentence three", lang="en", meta={}),
    ]
    embeddings = ingestor.embed_dense(chunks)

    assert len(embeddings) == len(chunks)
    assert all(len(e) == 1024 for e in embeddings)
