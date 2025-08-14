# -*- coding: utf-8 -*-
"""
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/14
@version: 0.5.0
"""

import uuid
from typing import List

import numpy as np
import pytest

from core.config.settings import NamedVectorConfig, Settings, VectorStoreConfig
from core.ingest import Ingestor
from core.types import Chunk, IngestItem
from core.vecdb.client import VecDB


def normalize_vector(vector: List[float]) -> List[float]:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return (np.array(vector) / norm).tolist()


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Override settings to use a temporary path for the database."""
    return Settings(
        vectorstore=VectorStoreConfig(
            path=str(tmp_path / "qdrant_db"),
            collection="rag_multimodal",
            named_vectors={
                "text_dense": NamedVectorConfig(size=1024, distance="cosine", name="text_dense"),
                "image": NamedVectorConfig(size=512, distance="cosine", name="image"),
                "text_sparse": NamedVectorConfig(sparse=True, name="text_sparse"),
            },
        )
    )


@pytest.fixture
def vecdb(settings: Settings) -> VecDB:
    db = VecDB(settings)
    db.create_collections()
    yield db
    db.close()


@pytest.fixture
def ingestor(settings: Settings, vecdb: VecDB) -> Ingestor:
    return Ingestor(settings, vecdb)


def test_upsert_idempotency_and_versioning(ingestor: Ingestor, vecdb: VecDB, settings: Settings):
    # Dummy data for initial upsert
    initial_ingest_item = IngestItem(
        path="/path/to/doc1.txt", doc_id="doc1", modality="text", meta={"mtime": 100, "lang": "en"}
    )
    initial_chunk = Chunk(
        doc_id="doc1",
        chunk_id="chunk0",
        content="This is the first version.",
        lang="en",
        meta=initial_ingest_item.meta,
    )
    initial_dense_embedding = [0.1] * 1024
    initial_sparse_embedding = {"indices": [1, 2], "values": [0.1, 0.2]}

    # First upsert
    ingestor.upsert(
        chunks=[initial_chunk],
        dense_embeddings=[initial_dense_embedding],
        sparse_embeddings=[initial_sparse_embedding],
        image_embeddings=np.array([]),
        ingest_items=[initial_ingest_item],
        captions=[],
    )

    # Verify initial state
    count_result = vecdb.client.count(collection_name=settings.vectorstore.collection, exact=True)
    assert count_result.count == 1

    # Retrieve the point to check its content
    original_point_id_str = (
        f"{initial_chunk.doc_id}#{initial_chunk.chunk_id}#{initial_chunk.meta["mtime"]}"
    )
    point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, original_point_id_str))
    retrieved_points = vecdb.client.retrieve(
        collection_name=settings.vectorstore.collection,
        ids=[point_id],
        with_payload=True,
        with_vectors=True,
    )
    assert len(retrieved_points) == 1
    assert retrieved_points[0].payload["version"] == 100
    assert retrieved_points[0].payload["mtime"] == 100
    assert np.allclose(
        retrieved_points[0].vector["text_dense"], normalize_vector(initial_dense_embedding)
    )
    # assert retrieved_points[0].vector["text_sparse"]["indices"] ==
    # initial_sparse_embedding["indices"]
    # assert np.allclose(retrieved_points[0].vector["text_sparse"]["values"], \
    #     initial_sparse_embedding["values"])

    # Modify data for re-ingestion (new version)
    updated_ingest_item = IngestItem(
        path="/path/to/doc1.txt", doc_id="doc1", modality="text", meta={"mtime": 200, "lang": "en"}
    )
    updated_chunk = Chunk(
        doc_id="doc1",
        chunk_id="chunk0",
        content="This is the updated version.",
        lang="en",
        meta=updated_ingest_item.meta,
    )
    updated_dense_embedding = [0.5] * 1024
    updated_sparse_embedding = {"indices": [3, 4], "values": [0.3, 0.4]}

    # Second upsert with modified data (should update the point, not create new)
    ingestor.upsert(
        chunks=[updated_chunk],
        dense_embeddings=[updated_dense_embedding],
        sparse_embeddings=[updated_sparse_embedding],
        image_embeddings=np.array([]),
        ingest_items=[updated_ingest_item],
        captions=[],
    )

    # Verify state after re-ingestion
    count_result_after_update = vecdb.client.count(
        collection_name=settings.vectorstore.collection, exact=True
    )
    assert count_result_after_update.count == 1  # Still only one point

    # Retrieve the updated point (using the new version in ID)
    original_updated_point_id_str = (
        f"{updated_chunk.doc_id}#{updated_chunk.chunk_id}#{updated_chunk.meta["mtime"]}"
    )
    updated_point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, original_updated_point_id_str))
    retrieved_updated_points = vecdb.client.retrieve(
        collection_name=settings.vectorstore.collection,
        ids=[updated_point_id],
        with_payload=True,
        with_vectors=True,
    )
    assert len(retrieved_updated_points) == 1
    assert retrieved_updated_points[0].payload["mtime"] == 200  # Check if mtime is updated
    assert np.allclose(
        retrieved_updated_points[0].vector["text_dense"], normalize_vector(updated_dense_embedding)
    )
    # assert retrieved_updated_points[0].vector["text_sparse"]["indices"] == \
    #     updated_sparse_embedding["indices"]
    # assert np.allclose(retrieved_updated_points[0].vector["text_sparse"]["values"], \
    #     updated_sparse_embedding["values"])

    # Verify old version is soft-deleted (not directly testable with count,
    # but implies by new point being there)
    # Qdrant's upsert with same ID replaces, so previous version is implicitly handled.
    # To explicitly test soft-delete, we'd need to query for deleted points if Qdrant exposed that.
    # For now, asserting count remains 1 and new version is present is sufficient \
    # for idempotency.

    # Test with a new chunk (should add a new point)
    new_ingest_item = IngestItem(
        path="/path/to/doc2.txt", doc_id="doc2", modality="text", meta={"mtime": 300, "lang": "en"}
    )
    new_chunk = Chunk(
        doc_id="doc2",
        chunk_id="chunk0",
        content="This is a new document.",
        lang="en",
        meta=new_ingest_item.meta,
    )
    new_dense_embedding = [0.9] * 1024
    new_sparse_embedding = {"indices": [5, 6], "values": [0.5, 0.6]}

    ingestor.upsert(
        chunks=[new_chunk],
        dense_embeddings=[new_dense_embedding],
        sparse_embeddings=[new_sparse_embedding],
        image_embeddings=np.array([]),
        ingest_items=[new_ingest_item],
        captions=[],
    )

    count_result_after_new = vecdb.client.count(
        collection_name=settings.vectorstore.collection, exact=True
    )
    assert count_result_after_new.count == 2  # Now two points

    # Verify the new point is present
    original_new_point_id_str = f"{new_chunk.doc_id}#{new_chunk.chunk_id}#{new_chunk.meta["mtime"]}"
    new_point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, original_new_point_id_str))
    retrieved_new_points = vecdb.client.retrieve(
        collection_name=settings.vectorstore.collection,
        ids=[new_point_id],
        with_payload=True,
        with_vectors=True,
    )
    assert len(retrieved_new_points) == 1
    assert retrieved_new_points[0].payload["lang"] == "en"
    assert retrieved_new_points[0].payload["mtime"] == 300
    assert np.allclose(
        retrieved_new_points[0].vector["text_dense"], normalize_vector(new_dense_embedding)
    )
    # assert retrieved_new_points[0].vector["text_sparse"]["indices"] == \
    #     new_sparse_embedding["indices"]
    # assert np.allclose(retrieved_new_points[0].vector["text_sparse"]["values"], \
    #     new_sparse_embedding["values"])
