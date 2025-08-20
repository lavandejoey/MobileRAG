# -*- coding: utf-8 -*-
"""
@file: apps/chat_api/routes/evidence.py
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/15
@version: 0.13.0
"""

from fastapi import APIRouter, HTTPException, status

from core.history.store import ChatHistoryStore

router = APIRouter()


@router.get("/evidence/{turn_id}")
async def get_evidence(turn_id: int):
    """
    Returns the evidence for a given turn.
    """
    try:
        history_store = ChatHistoryStore()
        # Assuming session_id is passed as a query parameter or derived from authentication
        # For simplicity, let's assume a default session_id for now or require it as a query param
        session_id = "default_session"  # This should be dynamic in a real app

        message = history_store.get_message_by_turn_id(session_id, turn_id)

        if not message:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Turn {turn_id} not found for session {session_id}",
            )

        evidence = message.get("evidence")

        if not evidence:
            return {
                "turn_id": turn_id,
                "evidence": [],
                "message": "No evidence found for this turn.",
            }

        return {"turn_id": turn_id, "evidence": evidence}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}",
        )
