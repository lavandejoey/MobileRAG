# -*- coding: utf-8 -*-
"""
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/15
@version: 0.12.0
"""

import asyncio
import json

from fastapi import APIRouter
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from core.graph.topology import create_graph

router = APIRouter()


class ChatRequest(BaseModel):
    query: str
    session_id: str


@router.post("/chat")
async def chat(request: ChatRequest):
    """
    Handles the chat endpoint.
    """
    graph = create_graph()

    async def event_generator():
        try:
            # Use the graph to stream responses
            async for chunk in graph.astream(request.dict()):
                if "answer" in chunk:
                    answer_payload = chunk["answer"]
                    # The graph might be nesting the answer in another dict, extract it
                    if isinstance(answer_payload, dict) and "answer" in answer_payload:
                        final_answer = answer_payload["answer"]
                    else:
                        final_answer = answer_payload
                    # Yield the answer chunk as a server-sent event
                    yield f'data: {json.dumps({"answer": final_answer})}\n\n'
                elif "evidence" in chunk:
                    # Yield the evidence chunk as a server-sent event
                    yield f'data: {json.dumps({"evidence": chunk["evidence"]})}\n\n'
                # Add a small delay to allow the client to process the event
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            # Handle client disconnection gracefully
            print("Client disconnected")

    # Return a streaming response
    return StreamingResponse(event_generator(), media_type="text/event-stream")
