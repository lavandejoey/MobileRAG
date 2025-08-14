# -*- coding: utf-8 -*-
"""
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/14
@version: 0.6.0
"""

from typing import List

from qdrant_client import models

from core.config.settings import Settings
from core.ingest.caption import ImageCaptioner
from core.ingest.embed_dense import DenseEmbedder
from core.ingest.embed_image import ImageEmbedder
from core.ingest.embed_sparse import SparseEmbedder  # Import SparseEmbedder
from core.retriever.types import Candidate, Evidence, HybridQuery
from core.vecdb.client import VecDB


class HybridRetriever:
    def __init__(
        self,
        settings: Settings,
        vecdb: VecDB,
        dense_embedder: DenseEmbedder,
        image_embedder: ImageEmbedder,
        image_captioner: ImageCaptioner,
        sparse_embedder: SparseEmbedder,
    ):
        self.settings = settings
        self.vecdb = vecdb
        self.dense_embedder = dense_embedder
        self.image_embedder = image_embedder
        self.image_captioner = image_captioner
        self.sparse_embedder = sparse_embedder  # Initialize SparseEmbedder

    def search(self, query: HybridQuery) -> List[Candidate]:
        query_vector = None
        query_image_embedding = None

        # Handle text query
        if query.text:
            query_vector = self.dense_embedder.embed_text_query(query.text)
            # query_sparse = self.sparse_embedder.embed_sparse([Chunk(doc_id="query",
            # chunk_id="q0", content=query.text, lang="en", meta={})])[0] # Embed as a dummy chunk

        # Handle image query
        if query.image_path:
            query_image_embedding = self.image_embedder.embed_image_query(query.image_path)

        # Determine the primary query vector
        final_query_vector = None
        if query_image_embedding is not None:
            final_query_vector = models.NamedVector(
                name=self.settings.vectorstore.named_vectors["image"].name,
                vector=query_image_embedding,
            )
        elif query_vector is not None:
            final_query_vector = models.NamedVector(
                name=self.settings.vectorstore.named_vectors["text_dense"].name, vector=query_vector
            )

        if final_query_vector is None:
            return []  # No query vector to search with

        # Prepare sparse query for Qdrant
        # qdrant_sparse_query = None
        # if query_sparse:
        #     qdrant_sparse_query = models.SparseVector(
        #         indices=query_sparse["indices"], values=query_sparse["values"]
        #     )

        # Perform hybrid search
        search_results = self.vecdb.client.search(
            collection_name=self.settings.vectorstore.collection,
            query_vector=final_query_vector,
            query_filter=(
                models.Filter(
                    must=[
                        models.FieldCondition(key=k, match=models.MatchValue(value=v))
                        for k, v in query.filters.items()
                    ]
                )
                if query.filters
                else None
            ),
            limit=query.topk_dense,  # Use topk_dense for limit for now
            with_payload=True,
            with_vectors=True,
            # query_sparse=qdrant_sparse_query if qdrant_sparse_query else None # Pass sparse query
        )

        candidates: List[Candidate] = []
        for hit in search_results:
            evidence_payload = hit.payload.copy()
            evidence = Evidence(
                file_path=evidence_payload.get("file_path"),
                page=evidence_payload.get("page"),
                bbox=tuple(evidence_payload["bbox"]) if "bbox" in evidence_payload else None,
                caption=evidence_payload.get("caption"),
                title=evidence_payload.get("title"),  # Assuming title might be in payload
            )
            candidates.append(
                Candidate(
                    id=hit.id,
                    score=hit.score,
                    text=hit.payload.get("content"),  # Assuming chunk content is stored in payload
                    evidence=evidence,
                    lang=hit.payload.get("lang"),
                    modality=hit.payload.get("modality"),
                )
            )
        return candidates
