# -*- coding: utf-8 -*-
"""
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/13
@version: 0.2.0
"""

from typing import List

from qdrant_client import QdrantClient, models

from core.config.settings import Settings
from core.vecdb.schema import Point, SparseVector


class VecDB:
    """A client wrapper for Qdrant vector database."""

    def __init__(self, settings: Settings):
        """
        Initialises the VecDB client.

        Args:
            settings: The application settings.
        """
        self.client = QdrantClient(path=settings.vectorstore.path)
        self.settings = settings

    def create_collections(self):
        """Creates the collections in the vector database if they don't exist."""
        # Main collection
        if not self.client.collection_exists(collection_name=self.settings.vectorstore.collection):
            vectors_config = {}
            for name, config in self.settings.vectorstore.named_vectors.items():
                if config.sparse:
                    vectors_config[name] = models.VectorParams(
                        size=0,  # Qdrant handles sparse vector size
                        distance=models.Distance.DOT,
                    )
                else:
                    vectors_config[name] = models.VectorParams(
                        size=config.size,
                        distance=models.Distance[config.distance.upper()],
                    )

            self.client.create_collection(
                collection_name=self.settings.vectorstore.collection,
                vectors_config=vectors_config,
            )
            # Payload indices for main collection
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
            # Assuming memory collection only has text_dense and text_sparse
            vectors_config = {
                self.settings.vectorstore.named_vectors["text_dense"].name: models.VectorParams(
                    size=self.settings.vectorstore.named_vectors["text_dense"].size,
                    distance=models.Distance[
                        self.settings.vectorstore.named_vectors["text_dense"].distance.upper()
                    ],
                ),
                # self.settings.vectorstore.named_vectors["text_sparse"].name: models.VectorParams(
                #     size=0,
                #     distance=models.Distance.DOT,
                # ),
            }
            self.client.create_collection(
                collection_name=self.settings.collection_mem,
                vectors_config=vectors_config,
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
