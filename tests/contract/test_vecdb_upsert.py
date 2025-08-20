# -*- coding: utf-8 -*-
"""
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/14
@version: 0.5.0
"""

import uuid
from typing import List
from unittest.mock import Mock

import numpy as np
import pytest

from core.config.devices import resolve_devices
from core.config.settings import NamedVectorConfig, Settings, VectorStoreConfig
from core.ingest.caption import ImageCaptioner
from core.ingest.embed_dense import DenseEmbedder
from core.ingest.embed_image import ImageEmbedder
from core.ingest.embed_sparse import SparseEmbedder
from core.ingest.pipeline import IngestPipeline
from core.ingest.upsert import upsert_factory
from core.types import Chunk
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
        device="cpu",  # Explicitly set device for testing
        vectorstore=VectorStoreConfig(
            local_path=str(tmp_path / "qdrant_db"),
            collection="rag_multimodal",
            named_vectors={
                "text_dense": NamedVectorConfig(size=1024, distance="cosine", name="text_dense"),
                "image": NamedVectorConfig(size=512, distance="cosine", name="image"),
                "text_sparse": NamedVectorConfig(sparse=True, name="text_sparse"),
            },
        ),
    )


@pytest.fixture
def vecdb(settings: Settings) -> VecDB:
    db = VecDB(settings)
    db.create_collections()
    yield db
    db.close()


@pytest.fixture
def ingest_components(settings: Settings, vecdb: VecDB):
    resolved_devices = resolve_devices()
    dense_embedder = DenseEmbedder(str(resolved_devices["embed"]))
    image_embedder = ImageEmbedder(str(resolved_devices["embed"]))
    image_captioner = ImageCaptioner(str(resolved_devices["embed"]))
    sparse_embedder = SparseEmbedder()

    ingest_pipeline = IngestPipeline(
        vecdb=vecdb,
        dense_embedder=dense_embedder,
        sparse_embedder=sparse_embedder,
        image_embedder=image_embedder,
        image_captioner=image_captioner,
    )
    upsert_func = upsert_factory(vecdb_client=vecdb, ingest_pipeline=ingest_pipeline)

    return ingest_pipeline, upsert_func


def test_upsert_idempotency_and_versioning(ingest_components, vecdb: VecDB, settings: Settings):
    ingest_pipeline, upsert_func = ingest_components

    # Dummy data for initial upsert
    initial_file_path = "/path/to/doc1.txt"
    # The ingest_pipeline.run() will handle scanning, chunking, embedding, and captioning
    # We need to ensure the test setup provides a file that can be processed by the pipeline
    # For this contract test, we will mock the pipeline.run() to return predefined chunks

    # Mock the ingest_pipeline.run method to return specific chunks for testing upsert logic
    # This is crucial because the actual pipeline.run() requires real files and models
    # For a contract test of upsert, we want to control the input chunks precisely.
    mock_chunks = [
        Chunk(
            doc_id="doc1",
            chunk_id="chunk0",
            content="This is the first version.",
            lang="en",
            meta={"mtime": 100, "file_path": initial_file_path},
            dense_vector=[0.1] * 1024,
            sparse_vector={"indices": [1, 2], "values": [0.5, 0.5]},
            image_vector=None,
            modality="text",
            page=1,
            caption=None,
        )
    ]
    ingest_pipeline.run = Mock(return_value=mock_chunks)

    # First upsert
    # The upsert_func now takes user_id and collection_name,
    # and internally calls ingest_pipeline.run()
    upsert_func(
        user_id="test_user",
        collection_name=settings.vectorstore.collection,
        file_path=initial_file_path,
    )

    # Verify initial state
    count_result = vecdb.client.count(collection_name=settings.vectorstore.collection, exact=True)
    assert count_result.count == 1

    # Retrieve the point to check its content
    original_point_id_str = f"{mock_chunks[0].doc_id}#{mock_chunks[0].chunk_id}"
    point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, original_point_id_str))
    retrieved_points = vecdb.client.retrieve(
        collection_name=settings.vectorstore.collection,
        ids=[point_id],
        with_payload=True,
        with_vectors=True,
    )
    assert len(retrieved_points) == 1
    assert retrieved_points[0].payload["mtime"] == 100
    # Assert dense vector (assuming it's normalized by the pipeline)
    assert np.allclose(
        retrieved_points[0].vector["text_dense"],
        normalize_vector([0.1] * 1024),  # Dummy embedding for assertion
    )

    # Modify data for re-ingestion (new version)
    updated_file_path = "/path/to/doc1.txt"
    updated_mock_chunks = [
        Chunk(
            doc_id="doc1",
            chunk_id="chunk0",
            content="This is the updated version.",
            lang="en",
            meta={"mtime": 200, "file_path": updated_file_path},
            dense_vector=[0.5] * 1024,
            sparse_vector={"indices": [3, 4], "values": [0.6, 0.4]},
            image_vector=None,
            modality="text",
            page=1,
            caption=None,
        )
    ]
    ingest_pipeline.run = Mock(return_value=updated_mock_chunks)

    # Second upsert with modified data (should update the point, not create new)
    upsert_func(
        user_id="test_user",
        collection_name=settings.vectorstore.collection,
        file_path=updated_file_path,
    )

    # Verify state after re-ingestion
    count_result_after_update = vecdb.client.count(
        collection_name=settings.vectorstore.collection, exact=True
    )
    assert count_result_after_update.count == 1  # Still only one point

    # Retrieve the updated point (ID should be the same)
    updated_point_id_str = f"{updated_mock_chunks[0].doc_id}#{updated_mock_chunks[0].chunk_id}"
    updated_point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, updated_point_id_str))
    assert point_id == updated_point_id
    retrieved_updated_points = vecdb.client.retrieve(
        collection_name=settings.vectorstore.collection,
        ids=[updated_point_id],
        with_payload=True,
        with_vectors=True,
    )
    assert len(retrieved_updated_points) == 1
    assert retrieved_updated_points[0].payload["mtime"] == 200  # Check if mtime is updated
    assert np.allclose(
        retrieved_updated_points[0].vector["text_dense"],
        normalize_vector([0.5] * 1024),  # Dummy embedding for assertion
    )

    # Test with a new chunk (should add a new point)
    new_file_path = "/path/to/doc2.txt"
    new_mock_chunks = [
        Chunk(
            doc_id="doc2",
            chunk_id="chunk0",
            content="This is a new document.",
            lang="en",
            meta={"mtime": 300, "file_path": new_file_path},
            dense_vector=[0.9] * 1024,
            sparse_vector={"indices": [5, 6], "values": [0.7, 0.3]},
            image_vector=None,
            modality="text",
            page=1,
            caption=None,
        )
    ]
    ingest_pipeline.run = Mock(return_value=new_mock_chunks)

    upsert_func(
        user_id="test_user",
        collection_name=settings.vectorstore.collection,
        file_path=new_file_path,
    )
