# -*- coding: utf-8 -*-
"""
@file: core/vecdb/client.py
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/13
@version: 0.2.0
"""

from typing import List

from qdrant_client import QdrantClient, models

try:
    # Present in modern qdrant-client; use for robust local detection
    from qdrant_client.local.qdrant_local import QdrantLocal  # type: ignore
except Exception:  # pragma: no cover
    QdrantLocal = None  # fallback if not available

from core.config.settings import Settings
from core.vecdb.schema import Point, SparseVector


class VecDB:
    """A client wrapper for Qdrant vector database."""

    def __init__(self, settings: Settings, in_memory: bool = False):
        """
        Initialises the VecDB client.

        Args:
            settings: The application settings.
            in_memory: If True, use an in-memory Qdrant client.
        """
        if in_memory:
            self.client = QdrantClient(location=":memory:")
        else:
            self.client = QdrantClient(path=settings.vectorstore.local_path)
        self.settings = settings

    def create_collections(self):
        """Creates the collections in the vector database if they don't exist."""
        # Main collection
        if not self.client.collection_exists(collection_name=self.settings.vectorstore.collection):
            # Split configs into dense and sparse
            dense_vectors_config: dict[str, models.VectorParams] = {}
            sparse_vectors_config: dict[str, models.SparseVectorParams] = {}
            for name, config in self.settings.vectorstore.named_vectors.items():
                if config.sparse:
                    # Register sparse vector under sparse_vectors_config
                    sparse_vectors_config[name] = models.SparseVectorParams()
                else:
                    dense_vectors_config[name] = models.VectorParams(
                        size=config.size,
                        distance=models.Distance[config.distance.upper()],
                    )

            self.client.create_collection(
                collection_name=self.settings.vectorstore.collection,
                vectors_config=dense_vectors_config,
                sparse_vectors_config=sparse_vectors_config if sparse_vectors_config else None,
            )
            # Payload indices for main collection (skip when running in-memory/local)
            is_local = getattr(self.client, "location", None) == ":memory:"
            if not is_local:
                self.client.create_payload_index(
                    collection_name=self.settings.vectorstore.collection,
                    field_name="lang",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
                self.client.create_payload_index(
                    collection_name=self.settings.vectorstore.collection,
                    field_name="modality",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
                self.client.create_payload_index(
                    collection_name=self.settings.vectorstore.collection,
                    field_name="mtime",  # Using mtime for versioning/time-based filtering
                    field_schema=models.PayloadSchemaType.INTEGER,
                )

        # Memory collection
        if not self.client.collection_exists(collection_name=self.settings.collection_mem):
            # Memory collection has dense text and sparse text vectors
            dense_cfg = {
                self.settings.vectorstore.named_vectors["text_dense"].name: models.VectorParams(
                    size=self.settings.vectorstore.named_vectors["text_dense"].size,
                    distance=models.Distance[
                        self.settings.vectorstore.named_vectors["text_dense"].distance.upper()
                    ],
                )
            }
            sparse_cfg = {
                self.settings.vectorstore.named_vectors[
                    "text_sparse"
                ].name: models.SparseVectorParams()
            }
            self.client.create_collection(
                collection_name=self.settings.collection_mem,
                vectors_config=dense_cfg,
                sparse_vectors_config=sparse_cfg,
            )

    def upsert(self, points: List[Point], collection_name: str):
        """Upserts points to the specified collection."""
        qdrant_points = []
        for point in points:
            qdrant_vectors = {}
            for vec_name, vec_data in point.vectors.items():
                if isinstance(vec_data, SparseVector):
                    qdrant_vectors[vec_name] = {
                        "indices": vec_data.indices,
                        "values": vec_data.values,
                    }
                else:
                    qdrant_vectors[vec_name] = vec_data

            qdrant_points.append(
                models.PointStruct(
                    id=point.id,
                    payload=point.payload,
                    vector=qdrant_vectors,  # This should correctly map named vectors
                )
            )
        if qdrant_points:
            self.client.upsert(collection_name=collection_name, points=qdrant_points)

    def close(self):
        """Closes the Qdrant client."""
        self.client.close()
