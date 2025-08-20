# -*- coding: utf-8 -*-
"""
@file: core/config/devices.py
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/13
@version: 0.1.0
"""

import os
from typing import Dict

import torch


def resolve_devices(vram_gb: float = -1.0) -> Dict[str, torch.device]:
    """
    Resolves the devices for the different models based on available VRAM.

    Args:
        vram_gb: The amount of VRAM in GB. If -1.0, it will be detected automatically.

    Returns:
        A dictionary mapping model types to torch devices.
    """
    force = os.getenv("FORCE_DEVICE", "").strip().lower()

    def pick(name: str, fallback: str) -> torch.device:
        override = os.getenv(name, "").strip()
        if override:
            return torch.device(override)
        if force in {"cuda", "gpu"} and torch.cuda.is_available():
            return torch.device("cuda:0")
        if force == "cpu":
            return torch.device("cpu")
        return torch.device(fallback)

    # Default fallback: prefer CUDA if available
    default = "cuda:0" if torch.cuda.is_available() else "cpu"
    return {
        "llm": pick("LLM_DEVICE", default),
        "clip": pick("CLIP_DEVICE", default),
        "embed": pick("EMBED_DEVICE", default),
        "reranker": pick("RERANKER_DEVICE", default),
    }
