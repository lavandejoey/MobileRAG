# -*- coding: utf-8 -*-
"""
@file: core/ingest/caption.py
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/13
@version: 0.4.0
"""

from typing import List

from core.clip.blip import BLIPCaptioner
from core.types import IngestItem


class ImageCaptioner:
    def __init__(self, device: str):
        self.captioner = BLIPCaptioner(device=device)

    def caption_images(self, ingest_items: List[IngestItem]) -> List[str]:
        image_paths = [item.path for item in ingest_items if item.modality == "image"]
        if not image_paths:
            return []
        return self.captioner.caption_image(image_paths)

    def to(self, device: str):
        self.captioner.model.to(device)
        self.captioner.device = device
