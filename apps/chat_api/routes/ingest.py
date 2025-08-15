# -*- coding: utf-8 -*-
"""
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/15
@version: 0.12.0
"""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class IngestRequest(BaseModel):
    file_path: str


@router.post("/ingest")
async def ingest(request: IngestRequest):
    """
    Handles the ingest endpoint.
    """
    # TODO: Implement actual ingestion logic
    return {"message": f"Ingested file: {request.file_path}"}
