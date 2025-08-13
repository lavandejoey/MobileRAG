# -*- coding: utf-8 -*-
"""
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/13
@version: 0.2.0
"""

import uuid

import pytest
from qdrant_client import models

from core.config.settings import Settings
from core.vecdb.client import VecDB


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Override settings to use a temporary path for the database."""
    return Settings(qdrant_path=str(tmp_path / "qdrant_db"))


def test_vecdb_roundtrip(settings: Settings):
    """Tests a full round-trip (create, upsert, retrieve, filter, delete) with the VecDB client."""
    vecdb = VecDB(settings)

    # 1. Create collections (idempotent)
    vecdb.create_collections()
    vecdb.create_collections()  # Should not fail - test idempotency

    # 2. Upsert a point
    doc_id = uuid.uuid4()
    vecdb.client.upsert(
        collection_name=settings.collection_main,
        points=[
            models.PointStruct(
                id=str(doc_id),
                vector={
                    "text_dense": [0.1] * settings.dense_dim_text,
                    "image": [0.2] * settings.dense_dim_image,
                },
                payload={"lang": "en", "modality": "text", "time": 12345},
            )
        ],
        wait=True,
    )

    # 3. Retrieve the point
    retrieved = vecdb.client.retrieve(collection_name=settings.collection_main, ids=[str(doc_id)])
    assert len(retrieved) == 1
    assert retrieved[0].payload["lang"] == "en"

    # 4. Filter points
    filtered, _ = vecdb.client.scroll(
        collection_name=settings.collection_main,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key="lang", match=models.MatchValue(value="en"))]
        ),
        limit=10,
    )
    assert len(filtered) == 1
    assert filtered[0].id == str(doc_id)

    # 5. Delete the point
    vecdb.client.delete(collection_name=settings.collection_main, points_selector=[str(doc_id)])

    # 6. Verify deletion
    retrieved_after_delete = vecdb.client.retrieve(
        collection_name=settings.collection_main, ids=[str(doc_id)]
    )
    assert len(retrieved_after_delete) == 0

    vecdb.close()
