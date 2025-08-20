# -*- coding: utf-8 -*-
"""
@file: core/ingest/upsert.py
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/14
@version: 0.14.0
"""

import uuid
from typing import Protocol

import numpy as np
from qdrant_client import models

from core.ingest.pipeline import IngestPipeline
from core.vecdb.client import VecDB
from core.vecdb.schema import Point


class Upsert(Protocol):
    def __call__(self, user_id: str, collection_name: str, file_path: str) -> None: ...


def upsert_factory(vecdb_client: VecDB, ingest_pipeline: IngestPipeline) -> Upsert:
    def _upsert(user_id: str, collection_name: str, file_path: str) -> None:
        chunks = ingest_pipeline.run(file_path=file_path)

        points = []
        for chunk in chunks:
            # Construct the unique ID for the point
            point_id_str = f"{chunk.doc_id}#{chunk.chunk_id}"
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, point_id_str))

            # Prepare vectors
            qdrant_vectors = {}
            if chunk.dense_vector:
                vec = np.array(chunk.dense_vector, dtype=np.float32)
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = (vec / norm).astype(np.float32)
                qdrant_vectors[
                    vecdb_client.settings.vectorstore.named_vectors["text_dense"].name
                ] = vec.tolist()
            if chunk.sparse_vector:
                qdrant_vectors[
                    vecdb_client.settings.vectorstore.named_vectors["text_sparse"].name
                ] = models.SparseVector(
                    indices=chunk.sparse_vector["indices"],
                    values=chunk.sparse_vector["values"],
                )
            if chunk.image_vector:
                img_vec = np.array(chunk.image_vector, dtype=np.float32)
                norm = np.linalg.norm(img_vec)
                if norm > 0:
                    img_vec = (img_vec / norm).astype(np.float32)
                qdrant_vectors[vecdb_client.settings.vectorstore.named_vectors["image"].name] = (
                    img_vec.tolist()
                )

            # Prepare payload
            payload = {
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
                "lang": chunk.lang,
                "modality": chunk.modality,
                "mtime": chunk.meta.get("mtime", 0),
                "file_path": chunk.meta.get("file_path"),
                "page": chunk.page,
                "caption": chunk.caption,
            }

            points.append(
                Point(
                    id=point_id,
                    vectors=qdrant_vectors,
                    payload=payload,
                )
            )

        if points:
            vecdb_client.upsert(points=points, collection_name=collection_name)

    return _upsert
