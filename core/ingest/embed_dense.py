# -*- coding: utf-8 -*-
"""
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/13
@version: 0.4.0
"""

from typing import List

import torch
from sentence_transformers import SentenceTransformer

from core.config.settings import Settings
from core.types import Chunk


class DenseEmbedder:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_name = "Qwen/Qwen3-Embedding-0.6B"
        self.model = SentenceTransformer(self.model_name, device=settings.device)
        self.model.eval()  # Set model to evaluation mode

    @torch.no_grad()
    def embed_dense(self, chunks: List[Chunk]) -> List[List[float]]:
        texts = [chunk.content for chunk in chunks]
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings.tolist()
