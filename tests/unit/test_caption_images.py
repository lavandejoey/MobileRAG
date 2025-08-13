# -*- coding: utf-8 -*-
"""
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/13
@version: 0.4.0
"""

import pytest
from PIL import Image

from core.config.settings import Settings
from core.ingest import Ingestor
from core.types import IngestItem


@pytest.fixture
def settings():
    return Settings(device="cpu")


@pytest.fixture
def ingestor(settings):
    return Ingestor(settings)


@pytest.fixture
def dummy_image_paths(tmp_path):
    # Create dummy image files
    img1_path = tmp_path / "test_image1.png"
    img2_path = tmp_path / "test_image2.png"

    # Create a simple 10x10 black image
    img = Image.new("RGB", (10, 10), color="black")
    img.save(img1_path)
    img.save(img2_path)

    return [str(img1_path), str(img2_path)]


def test_caption_images_non_empty(ingestor, dummy_image_paths):
    ingest_items = [IngestItem(path=dummy_image_paths[0], doc_id="img1", modality="image", meta={})]
    captions = ingestor.caption_images(ingest_items)

    assert len(captions) == 1
    assert isinstance(captions[0], str)
    assert len(captions[0]) > 0  # Caption should not be empty


def test_caption_images_batch_inference(ingestor, dummy_image_paths):
    ingest_items = [
        IngestItem(path=dummy_image_paths[0], doc_id="img1", modality="image", meta={}),
        IngestItem(path=dummy_image_paths[1], doc_id="img2", modality="image", meta={}),
    ]

    captions = ingestor.caption_images(ingest_items)

    assert len(captions) == 2
    assert all(isinstance(c, str) and len(c) > 0 for c in captions)
