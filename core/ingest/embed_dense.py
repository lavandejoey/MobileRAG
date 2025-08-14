# -*- coding: utf-8 -*-
"""
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/13
@version: 0.4.0
"""

from typing import List

import torch
from transformers import AutoModel, AutoTokenizer

from core.types import Chunk


class DenseEmbedder:
    def __init__(self, device: str = "cpu"):
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True, use_fast=False
        )
        self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True).to(device)
        self.model.eval()  # Set model to evaluation mode
        self.device = device

    @torch.no_grad()
    def embed_dense(self, chunks: List[Chunk]) -> List[List[float]]:
        texts = [chunk.content for chunk in chunks]
        # Tokenize the texts
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(
            self.device
        )
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Mean pooling to get sentence embeddings
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().tolist()
        return embeddings

    @torch.no_grad()
    def embed_text_query(self, text: str) -> List[float]:
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).cpu().tolist()[0]
        return embedding
