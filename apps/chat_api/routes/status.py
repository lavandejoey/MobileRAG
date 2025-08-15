# -*- coding: utf-8 -*-
"""
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/15
@version: 0.12.1
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/status")
async def get_status():
    """
    Returns the status of the API.
    """
    return {"status": "online"}
