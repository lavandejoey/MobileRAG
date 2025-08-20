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
    return Settings(device="cpu")


@pytest.fixture
def ingest_pipeline(settings):
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


def test_embed_sparse_output_format(ingest_pipeline):
    chunks = [
        Chunk(doc_id="doc1", chunk_id="c1", content="hello world", lang="en", meta={}),
        Chunk(
            doc_id="doc2",
            chunk_id="c2",
            content="test sentence for sparse embedding",
            lang="en",
            meta={},
        ),
    ]

    sparse_embeddings = ingest_pipeline.sparse_embedder.embed_sparse(chunks)

    assert len(sparse_embeddings) == len(chunks)

    for emb in sparse_embeddings:
        assert isinstance(emb, dict)
        assert "indices" in emb
        assert "values" in emb

        assert isinstance(emb["indices"], list)
        assert isinstance(emb["values"], list)

        assert len(emb["indices"]) == len(emb["values"])
        assert all(isinstance(i, int) for i in emb["indices"])
        assert all(isinstance(v, float) for v in emb["values"])

        # Basic sanity check: ensure some indices and values are present
        assert len(emb["indices"]) > 0
        assert len(emb["values"]) > 0


def test_embed_sparse_determinism(ingest_pipeline):
    chunks = [
        Chunk(doc_id="doc1", chunk_id="c1", content="hello world", lang="en", meta={}),
        Chunk(
            doc_id="doc2",
            chunk_id="c2",
            content="test sentence for sparse embedding",
            lang="en",
            meta={},
        ),
    ]

    embeddings1 = ingest_pipeline.sparse_embedder.embed_sparse(chunks)
    embeddings2 = ingest_pipeline.sparse_embedder.embed_sparse(chunks)

    assert embeddings1 == embeddings2
