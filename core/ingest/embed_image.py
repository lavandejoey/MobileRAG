# -*- coding: utf-8 -*-
"""
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/13
@version: 0.4.0
"""

from typing import List

import numpy as np

from core.clip.openclip import OpenCLIPEmbedder
from core.types import IngestItem


class ImageEmbedder:
    def __init__(self, device: str):
        self.embedder = OpenCLIPEmbedder(device=device)

    def embed_image(self, ingest_items: List[IngestItem]) -> np.ndarray:
        image_paths = [item.path for item in ingest_items if item.modality == "image"]
        if not image_paths:
            return np.array([])
        return self.embedder.embed_image(image_paths)

    def embed_image_query(self, image_path: str) -> List[float]:
        return self.embedder.embed_image([image_path]).tolist()[0]
