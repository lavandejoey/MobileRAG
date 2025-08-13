# -*- coding: utf-8 -*-
"""
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/13
@version: 0.2.0
"""

from qdrant_client import QdrantClient, models

from core.config.settings import Settings


class VecDB:
    """A client wrapper for Qdrant vector database."""

    def __init__(self, settings: Settings):
        """
        Initialises the VecDB client.

        Args:
            settings: The application settings.
        """
        self.client = QdrantClient(path=settings.qdrant_path)
        self.settings = settings

    def create_collections(self):
        """Creates the collections in the vector database if they don't exist."""
        if not self.client.collection_exists(
            collection_name=self.settings.collection_main
        ):
            self.client.create_collection(
                collection_name=self.settings.collection_main,
                vectors_config={
                    "text_dense": models.VectorParams(
                        size=self.settings.dense_dim_text,
                        distance=models.Distance.COSINE,
                    ),
                    "image": models.VectorParams(
                        size=self.settings.dense_dim_image,
                        distance=models.Distance.COSINE,
                    ),
                    "text_sparse": models.VectorParams(
                        size=0,  # Placeholder, Qdrant handles sparse vector size
                        distance=models.Distance.DOT,
                    ),
                },
            )
            self.client.create_payload_index(
                collection_name=self.settings.collection_main,
                field_name="lang",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            self.client.create_payload_index(
                collection_name=self.settings.collection_main,
                field_name="modality",
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
            self.client.create_payload_index(
                collection_name=self.settings.collection_main,
                field_name="time",
                field_schema=models.PayloadSchemaType.INTEGER,
            )

        if not self.client.collection_exists(
            collection_name=self.settings.collection_mem
        ):
            self.client.create_collection(
                collection_name=self.settings.collection_mem,
                vectors_config={
                    "text_dense": models.VectorParams(
                        size=self.settings.dense_dim_text,
                        distance=models.Distance.COSINE,
                    ),
                    "text_sparse": models.VectorParams(
                        size=0,  # Placeholder
                        distance=models.Distance.DOT,
                    ),
                },
            )

    def close(self):
        """Closes the Qdrant client."""
        self.client.close()
