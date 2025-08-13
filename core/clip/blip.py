# -*- coding: utf-8 -*-
"""
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/13
@version: 0.4.0
"""

from typing import List

import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor


class BLIPCaptioner:
    def __init__(self, device: str = "cpu"):
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(device)
        self.device = device

    @torch.no_grad()
    def caption_image(self, image_paths: List[str]) -> List[str]:
        captions: List[str] = []
        for path in image_paths:
            image = Image.open(path).convert("RGB")
            # unconditional image captioning
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            out = self.model.generate(**inputs)
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            captions.append(caption)
        return captions
