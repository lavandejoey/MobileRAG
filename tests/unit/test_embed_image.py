# -*- coding: utf-8 -*-
"""
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/13
@version: 0.4.0
"""

import numpy as np
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


def test_embed_image_output_dimension(ingestor, dummy_image_paths):
    ingest_items = [IngestItem(path=dummy_image_paths[0], doc_id="img1", modality="image", meta={})]
    embeddings = ingestor.embed_image(ingest_items)

    assert embeddings.shape[0] == 1
    assert embeddings.shape[1] == 512  # OpenCLIP ViT-B/32 outputs 512-dim


def test_embed_image_determinism(ingestor, dummy_image_paths):
    ingest_items = [IngestItem(path=dummy_image_paths[0], doc_id="img1", modality="image", meta={})]

    embeddings1 = ingestor.embed_image(ingest_items)
    embeddings2 = ingestor.embed_image(ingest_items)

    np.testing.assert_allclose(embeddings1, embeddings2, rtol=1e-5, atol=1e-5)


def test_embed_image_batch_inference(ingestor, dummy_image_paths):
    ingest_items = [
        IngestItem(path=dummy_image_paths[0], doc_id="img1", modality="image", meta={}),
        IngestItem(path=dummy_image_paths[1], doc_id="img2", modality="image", meta={}),
    ]

    embeddings = ingestor.embed_image(ingest_items)

    assert embeddings.shape[0] == 2
    assert embeddings.shape[1] == 512
