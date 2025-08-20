# -*- coding: utf-8 -*-
"""
@file: apps/chat_api/routes/ingest.py
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/15
@version: 0.13.0
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from core.config.devices import resolve_devices
from core.config.settings import Settings
from core.ingest.caption import ImageCaptioner
from core.ingest.embed_dense import DenseEmbedder
from core.ingest.embed_image import ImageEmbedder
from core.ingest.embed_sparse import SparseEmbedder
from core.ingest.pipeline import IngestPipeline
from core.ingest.upsert import upsert_factory
from core.vecdb.client import VecDB

router = APIRouter()


class IngestRequest(BaseModel):
    file_path: str


@router.post("/ingest")
async def ingest(request: IngestRequest):
    """
    Handles the ingest endpoint.
    """
    try:
        settings = Settings()
        resolved_devices = resolve_devices()

        vecdb = VecDB(settings)  # Use persistent Qdrant for API
        vecdb.create_collections()

        dense_embedder = DenseEmbedder(str(resolved_devices["embed"]))
        image_embedder = ImageEmbedder(str(resolved_devices["embed"]))
        image_captioner = ImageCaptioner(str(resolved_devices["embed"]))
        sparse_embedder = SparseEmbedder()

        ingest_pipeline = IngestPipeline(
            vecdb,
            dense_embedder,
            sparse_embedder,
            image_embedder,
            image_captioner,
        )

        # Run the ingestion pipeline to get processed chunks
        chunks = ingest_pipeline.run(request.file_path)

        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No chunks generated from the provided file.",
            )

        # Get the upsert function and perform upsert
        upsert_func = upsert_factory(vecdb_client=vecdb, ingest_pipeline=ingest_pipeline)
        # The upsert_factory returns a function that takes user_id and collection_name
        # For now, let's use a dummy user_id and the main collection
        upsert_func(
            user_id="default_user",
            collection_name=settings.vectorstore.collection,
            file_path=request.file_path,
        )

        return {
            "message": f"Successfully ingested and processed: {request.file_path}",
            "chunks_processed": len(chunks),
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}",
        )
