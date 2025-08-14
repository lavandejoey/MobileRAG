# -*- coding: utf-8 -*-
"""
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/14
@version: 0.9.0
"""

import uuid
from typing import List, Optional

from qdrant_client import models

from core.config.settings import Settings
from core.ingest.embed_dense import DenseEmbedder
from core.ingest.embed_image import ImageEmbedder  # For completeness, though memory is text-focused
from core.ingest.embed_sparse import SparseEmbedder
from core.memory.types import MemoryCard, QueryResult
from core.vecdb.client import VecDB


class MemoryStore:
    def __init__(
        self,
        settings: Settings,
        vecdb: VecDB,
        dense_embedder: DenseEmbedder,
        sparse_embedder: SparseEmbedder,
        image_embedder: ImageEmbedder,
    ):
        self.settings = settings
        self.vecdb = vecdb
        self.dense_embedder = dense_embedder
        self.sparse_embedder = sparse_embedder
        self.image_embedder = image_embedder

    def upsert_memory(self, memory_card: MemoryCard):
        # Embed the memory card content
        dense_vector = self.dense_embedder.embed_text_query(memory_card.content)
        sparse_vector_data = self.sparse_embedder.embed_sparse([memory_card.content])[
            0
        ]  # Pass content as a list

        # Create Qdrant PointStruct
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, memory_card.id))
        point = models.PointStruct(
            id=point_id,
            payload=memory_card.to_payload(),
            vector={
                self.settings.vectorstore.named_vectors["text_dense"].name: dense_vector,
                self.settings.vectorstore.named_vectors["text_sparse"].name: models.SparseVector(
                    indices=sparse_vector_data["indices"],
                    values=sparse_vector_data["values"],
                ),
            },
        )
        self.vecdb.client.upsert(collection_name=self.settings.collection_mem, points=[point])

    def retrieve_memories(self, query_text: str, top_k: int = 5) -> List[QueryResult]:
        # Embed the query text
        query_dense_vector = self.dense_embedder.embed_text_query(query_text)
        query_sparse_data = self.sparse_embedder.embed_sparse([query_text])[0]

        # Perform hybrid search on the agent_memory collection
        search_results = self.vecdb.client.search(
            collection_name=self.settings.collection_mem,
            query_vector=models.NamedVector(
                name=self.settings.vectorstore.named_vectors["text_dense"].name,
                vector=query_dense_vector,
            ),
            query_sparse=models.SparseVector(
                indices=query_sparse_data["indices"],
                values=query_sparse_data["values"],
            ),
            limit=top_k,
            with_payload=True,
            with_vectors=False,  # No need to retrieve vectors for results
        )

        results = []
        for hit in search_results:
            results.append(
                QueryResult(
                    memory_card=MemoryCard.from_payload(hit.payload),
                    score=hit.score,
                )
            )
        return results

    def delete_memory(self, memory_id: str):
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, memory_id))
        self.vecdb.client.delete(
            collection_name=self.settings.collection_mem, points_selector=[point_id]
        )

    def get_memory(self, memory_id: str) -> Optional[MemoryCard]:
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, memory_id))
        retrieved = self.vecdb.client.retrieve(
            collection_name=self.settings.collection_mem, ids=[point_id], with_payload=True
        )
        if retrieved:
            return MemoryCard.from_payload(retrieved[0].payload)
        return None
