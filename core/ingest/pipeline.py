# -*- coding: utf-8 -*-
"""
@file: core/ingest/pipeline.py
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/15
@version: 0.14.0
"""

from typing import List

import torch

from core.ingest.caption import ImageCaptioner
from core.ingest.chunk import chunk
from core.ingest.embed_dense import DenseEmbedder
from core.ingest.embed_image import ImageEmbedder
from core.ingest.embed_sparse import SparseEmbedder
from core.ingest.scan import scan
from core.types import Chunk, IngestItem
from core.vecdb.client import VecDB


class IngestPipeline:
    def __init__(
        self,
        vecdb: VecDB,
        dense_embedder: DenseEmbedder,
        sparse_embedder: SparseEmbedder,
        image_embedder: ImageEmbedder,
        image_captioner: ImageCaptioner,
    ):
        self.vecdb = vecdb
        self.dense_embedder = dense_embedder
        self.sparse_embedder = sparse_embedder
        self.image_embedder = image_embedder
        self.image_captioner = image_captioner

    def run(self, file_path: str) -> List[Chunk]:
        ingest_items: List[IngestItem] = scan(file_path)
        if not ingest_items:
            return []

        # Split by modality
        text_items = [it for it in ingest_items if it.modality == "text"]
        image_items = [it for it in ingest_items if it.modality == "image"]

        # 1) Chunk text/PDFs
        text_chunks: List[Chunk] = []
        if text_items:
            text_chunks = chunk(text_items)

        # 2) Embed text chunks (dense + sparse)
        if text_chunks:
            self.dense_embedder.to("cuda" if torch.cuda.is_available() else "cpu")
            dense_vecs = self.dense_embedder.embed_dense(text_chunks)
            sparse_vecs = self.sparse_embedder.embed_sparse(text_chunks)
            for i, c in enumerate(text_chunks):
                c.dense_vector = dense_vecs[i]
                c.sparse_vector = sparse_vecs[i]
            self.dense_embedder.to("cpu")

        # 3) Convert images to chunks, then embed + caption
        image_chunks: List[Chunk] = []
        if image_items:
            self.image_embedder.to("cuda" if torch.cuda.is_available() else "cpu")
            self.image_captioner.to("cuda" if torch.cuda.is_available() else "cpu")
            # Embed and caption in batch
            img_vecs = self.image_embedder.embed_image(image_items)  # np.ndarray [N, D] or empty
            captions = self.image_captioner.caption_images(image_items)
            self.image_embedder.to("cpu")
            self.image_captioner.to("cpu")
            for idx, it in enumerate(image_items):
                # Minimal Chunk for images; content left empty, info lives in payload/meta
                img_chunk = Chunk(
                    doc_id=it.doc_id,
                    chunk_id=f"{it.doc_id}#img_{idx}",
                    content="",
                    lang="",
                    meta=it.meta | {"modality": "image"},
                    page=None,
                    bbox=None,
                )
                # Attach vectors/caption (guard shapes)
                if hasattr(img_vecs, "shape") and img_vecs.size > 0:
                    img_chunk.image_vector = img_vecs[idx].tolist()
                if captions:
                    img_chunk.caption = captions[idx]
                image_chunks.append(img_chunk)

        return text_chunks + image_chunks
