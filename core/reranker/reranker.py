# -*- coding: utf-8 -*-
"""
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/14
@version: 0.7.0
"""

from typing import List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from core.config.settings import Settings
from core.retriever.types import Candidate


class Reranker:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(
            settings.device
        )
        self.model.eval()

    @torch.no_grad()
    def rank(self, query: str, candidates: List[Candidate], topr: int = 10) -> List[Candidate]:
        if not candidates:
            return []

        # Prepare input for the re-ranker model
        # The model expects pairs of (query, candidate_text)
        texts_to_rank = []
        for candidate in candidates:
            # Use candidate.text if available, otherwise use a placeholder or skip
            candidate_text = candidate.text if candidate.text else ""
            texts_to_rank.append([query, candidate_text])

        # Perform batched inference
        inputs = self.tokenizer(texts_to_rank, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.settings.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        scores = outputs.logits.squeeze().cpu().tolist()

        # Pair candidates with their scores
        scored_candidates = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

        # Return top topr candidates
        reranked_candidates = [cand for cand, score in scored_candidates[:topr]]
        return reranked_candidates
