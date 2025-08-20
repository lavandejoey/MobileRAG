# -*- coding: utf-8 -*-
"""
@file: core/clip/openclip.py
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/13
@version: 0.4.0
"""

from typing import List

import numpy as np
import open_clip
import torch
from PIL import Image


class OpenCLIPEmbedder:
    def __init__(self, device: str = "cpu"):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device
        )
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def embed_image(self, image_paths: List[str]) -> np.ndarray:
        images = []
        for path in image_paths:
            image = Image.open(path).convert("RGB")
            images.append(self.preprocess(image))

        if not images:
            return np.array([])

        image_input = torch.tensor(np.stack(images)).to(self.device)
        image_features = self.model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy()
