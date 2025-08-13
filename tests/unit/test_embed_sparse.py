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
    return Settings(device="cpu")


@pytest.fixture
def ingestor(settings):
    return Ingestor(settings)


def test_embed_sparse_output_format(ingestor):
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

    sparse_embeddings = ingestor.embed_sparse(chunks)

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


def test_embed_sparse_determinism(ingestor):
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

    embeddings1 = ingestor.embed_sparse(chunks)
    embeddings2 = ingestor.embed_sparse(chunks)

    assert embeddings1 == embeddings2
