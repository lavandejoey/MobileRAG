#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model registry and factory functions.
src/models/registry.py

@author: LIU Ziyi
@date: 2025-12-30
@license: Apache-2.0
"""
from __future__ import annotations

from src.config import ModelConfig
from .base import ChatModel
from .ollama import OllamaChatModel


def create_chat_model(cfg: ModelConfig) -> ChatModel:
    backend = cfg.BACKEND.lower().strip()

    if backend == "ollama":
        return OllamaChatModel(model=cfg.MODEL_NAME, think=cfg.THINK)

    if backend == "hf":
        raise NotImplementedError(
            "HF backend placeholder: implement transformers/vLLM later."
        )

    if backend == "onnx":
        raise NotImplementedError(
            "ONNX backend placeholder: implement onnxruntime-genai later."
        )

    if backend == "gguf":
        raise NotImplementedError(
            "GGUF backend placeholder: implement llama-cpp-python later."
        )

    raise ValueError(f"Unknown backend: {cfg.BACKEND}")
