# -*- coding: utf-8 -*-
"""
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/15
@version: 0.12.0
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/evidence/{turn_id}")
async def get_evidence(turn_id: str):
    """
    Returns the evidence for a given turn.
    """
    # TODO: Implement actual evidence retrieval logic
    return {"turn_id": turn_id, "evidence": []}
