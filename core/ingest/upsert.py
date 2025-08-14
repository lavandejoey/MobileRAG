# -*- coding: utf-8 -*-
"""
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/14
@version: 0.5.0
"""

import uuid
from typing import Any, Dict, List

import numpy as np

from core.config.settings import Settings
from core.types import Chunk, IngestItem
from core.vecdb.client import VecDB
from core.vecdb.schema import Point


class Upserter:
    def __init__(self, vecdb: VecDB, settings: Settings):
        self.vecdb = vecdb
        self.settings = settings

    def upsert(
        self,
        chunks: List[Chunk],
        dense_embeddings: List[List[float]],
        sparse_embeddings: List[Dict[str, Any]],
        image_embeddings: np.ndarray,
        ingest_items: List[IngestItem],
        captions: List[str],
    ) -> None:
        points: List[Point] = []
        # Assuming a 1:1 mapping between chunks/ingest_items and their embeddings/captions \
        # for simplicity
        # In a real scenario, you might need more complex mapping or filtering

        # Create a map for image embeddings and captions by doc_id/path for easier lookup
        image_data_map = {}
        for i, item in enumerate(ingest_items):
            if item.modality == "image":
                image_data_map[item.doc_id] = {
                    "embedding": image_embeddings[i] if image_embeddings.size > 0 else None,
                    "caption": captions[i] if captions else None,
                }

        for i, chunk in enumerate(chunks):
            vectors = {}
            payload = chunk.meta.copy()

            # Add dense text embedding
            vectors[self.settings.vectorstore.named_vectors["text_dense"].name] = dense_embeddings[
                i
            ]

            # Add sparse text embedding
            # FastEmbed already returns in Qdrant sparse format
            # sparse_data = sparse_embeddings[i]
            # vectors[self.settings.vectorstore.named_vectors["text_sparse"].name] = sparse_data

            # Handle image-related data if the chunk corresponds to an image
            if payload.get("modality") == "image" and chunk.doc_id in image_data_map:
                image_data = image_data_map[chunk.doc_id]
                if image_data["embedding"] is not None:
                    vectors[self.settings.vectorstore.named_vectors["image"].name] = image_data[
                        "embedding"
                    ].tolist()
                if image_data["caption"] is not None:
                    payload["caption"] = image_data["caption"]

            # Construct doc_id#chunk_id#version
            # For now, version can be a timestamp or a simple counter.
            # Let's use mtime from meta as version.
            version = payload.get("mtime", 0)  # Use mtime as a simple version for now
            point_id_base = f"{chunk.doc_id}#{chunk.chunk_id}"
            point_id = str(
                uuid.uuid5(uuid.NAMESPACE_URL, point_id_base)
            )  # Convert to UUID for Qdrant ID
            payload["version"] = version  # Store version in payload

            points.append(Point(id=point_id, vectors=vectors, payload=payload))

        if points:
            self.vecdb.upsert(points, collection_name=self.settings.vectorstore.collection)
