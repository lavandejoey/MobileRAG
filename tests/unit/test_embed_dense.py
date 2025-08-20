# -*- coding: utf-8 -*-
"""
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/13
@version: 0.4.0
"""

from unittest.mock import Mock

import pytest

from core.config.settings import Settings
from core.ingest.caption import ImageCaptioner
from core.ingest.embed_dense import DenseEmbedder
from core.ingest.embed_image import ImageEmbedder
from core.ingest.embed_sparse import SparseEmbedder
from core.ingest.pipeline import IngestPipeline
from core.types import Chunk
from core.vecdb.client import VecDB


@pytest.fixture
def settings():
    # Use CPU for testing to avoid GPU dependency
    return Settings(device="cpu")


@pytest.fixture
def ingest_pipeline(settings):
    # Use CPU for testing to avoid GPU dependency
    mock_vecdb = Mock(spec=VecDB)
    mock_vecdb.create_collections.return_value = None

    dense_embedder = DenseEmbedder(settings.device)
    image_embedder = ImageEmbedder(settings.device)
    image_captioner = ImageCaptioner(settings.device)
    sparse_embedder = SparseEmbedder()

    return IngestPipeline(
        vecdb=mock_vecdb,
        dense_embedder=dense_embedder,
        sparse_embedder=sparse_embedder,
        image_embedder=image_embedder,
        image_captioner=image_captioner,
    )


def test_dense_embedder_deterministic(ingest_pipeline):
    chunks = [
        Chunk(doc_id="doc1", chunk_id="c1", content="hello world", lang="en", meta={}),
        Chunk(doc_id="doc2", chunk_id="c2", content="test sentence", lang="en", meta={}),
    ]

    # Embed twice to check for determinism
    texts = [c.content for c in chunks]
    embeddings1 = ingest_pipeline.dense_embedder.embed_dense(texts)
    embeddings2 = ingest_pipeline.dense_embedder.embed_dense(texts)

    # Check if embeddings are deterministic (within a small tolerance for float precision)
    assert len(embeddings1) == len(chunks)
    assert len(embeddings2) == len(chunks)
    for i in range(len(chunks)):
        for j in range(len(embeddings1[i])):
            assert abs(embeddings1[i][j] - embeddings2[i][j]) < 1e-6


def test_dense_embedder_output_dimension(ingest_pipeline):
    chunks = [Chunk(doc_id="doc1", chunk_id="c1", content="single sentence", lang="en", meta={})]
    embeddings = ingest_pipeline.dense_embedder.embed_dense(chunks)

    # Qwen3-Embedding-0.6B outputs 1024-dim
    assert len(embeddings) == 1
    assert len(embeddings[0]) == 1024


def test_dense_embedder_batch_inference(ingest_pipeline):
    chunks = [
        Chunk(doc_id="doc1", chunk_id="c1", content="sentence one", lang="en", meta={}),
        Chunk(doc_id="doc2", chunk_id="c2", content="sentence two", lang="en", meta={}),
        Chunk(doc_id="doc3", chunk_id="c3", content="sentence three", lang="en", meta={}),
    ]
    embeddings = ingest_pipeline.dense_embedder.embed_dense(chunks)

    assert len(embeddings) == len(chunks)
    assert all(len(e) == 1024 for e in embeddings)
