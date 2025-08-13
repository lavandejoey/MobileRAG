from typing import Any, Dict, List

import numpy as np

from core.config.settings import Settings
from core.types import Chunk, IngestItem

from .caption import ImageCaptioner
from .chunk import chunk as _chunk
from .embed_dense import DenseEmbedder
from .embed_image import ImageEmbedder
from .embed_sparse import SparseEmbedder
from .scan import scan as _scan


class Ingestor:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.dense_embedder = DenseEmbedder(settings)
        self.sparse_embedder = SparseEmbedder()
        self.image_embedder = ImageEmbedder(settings)
        self.image_captioner = ImageCaptioner(settings)

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
