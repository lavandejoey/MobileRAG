# -*- coding: utf-8 -*-
"""
@file: core/ingest/embed_dense.py
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/13
@version: 0.13.0
"""

import os
from typing import List, Union

import torch
from transformers import AutoModel, AutoTokenizer

from core.types import Chunk


class DenseEmbedder:
    def __init__(self, device: str):
        self.model_name = "Qwen/Qwen3-Embedding-0.6B"
        # Resolve device string to torch.device, support "auto"
        if device == "auto":
            resolved_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            resolved_device = torch.device(device)
        self.device = resolved_device

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True, use_fast=False, padding_side="left"
        )
        # Ensure pad token is defined and consistent
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Force model to be aware of pad token id
        self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)

        # Configure model kwargs based on device
        model_kwargs = {}
        if self.device.type == "cuda":
            # Optional speedups that are safe on most GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            # Decide if FlashAttention is usable: Ampere (SM>=80) and package present
            disable_flash_env = os.getenv("DISABLE_FLASH_ATTN", "0") == "1"
            can_flash = False
            if not disable_flash_env:
                try:
                    import flash_attn  # noqa: F401

                    major, minor = torch.cuda.get_device_capability(self.device)
                    can_flash = major >= 8  # Ampere or newer
                except Exception:
                    can_flash = False
            if can_flash:
                model_kwargs["attn_implementation"] = "flash_attention_2"
            # Use fp16 on CUDA for embeddings
            model_kwargs["torch_dtype"] = torch.float16
        else:
            # CPU path
            model_kwargs["torch_dtype"] = torch.float32

        self.model = AutoModel.from_pretrained(self.model_name, **model_kwargs).to(self.device)
        self.model.eval()  # Set model to evaluation mode

    def to(self, device: str):
        self.model.to(device)
        self.device = device

    @torch.no_grad()
    def embed_dense(self, items: List[Union[Chunk, str]]) -> List[List[float]]:
        if not items:
            return []
        if isinstance(items[0], Chunk):
            texts = [item.content for item in items]
        else:
            texts = items
        # Tokenize the texts
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        # Always include attention_mask
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Mean pooling to get sentence embeddings
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().tolist()
        return embeddings

    @torch.no_grad()
    def embed_text_query(self, text: str) -> List[float]:
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
        embedding = outputs.last_hidden_state.mean(dim=1).cpu().tolist()[0]
        return embedding
