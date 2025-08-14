# -*- coding: utf-8 -*-
"""
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/14
@version: 0.5.0
"""

from typing import Any, Dict, List

import numpy as np

from core.config.devices import resolve_devices
from core.config.settings import Settings
from core.types import Chunk, IngestItem
from core.vecdb.client import VecDB

from .caption import ImageCaptioner
from .chunk import chunk as _chunk
from .embed_dense import DenseEmbedder
from .embed_image import ImageEmbedder
from .embed_sparse import SparseEmbedder
from .scan import scan as _scan
from .upsert import Upserter


class Ingestor:
    def __init__(self, settings: Settings, vecdb: VecDB):
        self.settings = settings
        self.vecdb = vecdb

        # Resolve devices for models
        resolved_devices = resolve_devices()  # Assuming auto-detection for now
        embed_device = resolved_devices["embed"]

        self.dense_embedder = DenseEmbedder(embed_device)
        self.sparse_embedder = SparseEmbedder()
        self.image_embedder = ImageEmbedder(embed_device)
        self.image_captioner = ImageCaptioner(embed_device)
        self.upserter = Upserter(vecdb, settings)

    def scan(self, root_dir: str) -> List[IngestItem]:
        return _scan(root_dir)

    def chunk(self, ingest_items: List[IngestItem]) -> List[Chunk]:
        return _chunk(ingest_items)

    def embed_dense(self, chunks: List[Chunk]) -> List[List[float]]:
        return self.dense_embedder.embed_dense(chunks)

    def embed_sparse(self, chunks: List[Chunk]) -> List[Dict[str, Any]]:
        return self.sparse_embedder.embed_sparse(chunks)

    def embed_image(self, ingest_items: List[IngestItem]) -> np.ndarray:
        return self.image_embedder.embed_image(ingest_items)

    def caption_images(self, ingest_items: List[IngestItem]) -> List[str]:
        return self.image_captioner.caption_images(ingest_items)

    def upsert(
        self,
        chunks: List[Chunk],
        dense_embeddings: List[List[float]],  # Keep argument for now
        sparse_embeddings: List[Dict[str, Any]],
        image_embeddings: np.ndarray,
        ingest_items: List[IngestItem],
        captions: List[str],
    ) -> None:
        # Temporarily pass dummy dense_embeddings if dense_embedder is disabled
        # In a real scenario, you'd handle this more gracefully or ensure embeddings are generated
        # For testing upsert logic, we can pass the provided dense_embeddings directly
        self.upserter.upsert(
            chunks, dense_embeddings, sparse_embeddings, image_embeddings, ingest_items, captions
        )
