# -*- coding: utf-8 -*-
"""
@file: apps/chat_api/routes/chat.py
"""

import asyncio
import json

from fastapi import APIRouter
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from core.graph.topology import create_graph

router = APIRouter()
graph = create_graph()


class ChatRequest(BaseModel):
    query: str
    session_id: str


class ThinkFilter:
    """Stateful filter to remove <think>...</think> from streamed text."""

    def __init__(self):
        self.residual = ""
        self.inside_think = False

    def __call__(self, delta: str) -> str:
        buf = self.residual + delta
        out = []
        i = 0
        while i < len(buf):
            if not self.inside_think and buf.startswith("<think>", i):
                self.inside_think = True
                i += len("<think>")
                continue
            if self.inside_think:
                end = buf.find("</think>", i)
                if end == -1:
                    self.residual = buf[i:]
                    return "".join(out)
                i = end + len("</think>")
                self.inside_think = False
                continue
            out.append(buf[i])
            i += 1
        self.residual = ""
        return "".join(out)


async def stream_answer(chunk, filter_fn, sent_reasoning_flag):
    """Process answer chunk and yield events."""
    answer_payload = chunk["answer"]
    if isinstance(answer_payload, dict) and "answer" in answer_payload:
        final_answer = answer_payload["answer"]
    else:
        final_answer = answer_payload
    visible = filter_fn(str(final_answer))
    events = []
    if visible:
        events.append(f'data: {json.dumps({"answer": visible})}\n\n')
    if not sent_reasoning_flag and ("<think>" in str(final_answer) or filter_fn.inside_think):
        events.append(f'data: {json.dumps({"reasoning_available": True})}\n\n')
        sent_reasoning_flag = True
    return events, sent_reasoning_flag


async def stream_evidence(chunk):
    """Process evidence chunk and yield event."""
    return [f'data: {json.dumps({"evidence": chunk["evidence"]})}\n\n']


async def event_generator(request: ChatRequest):
    filter_fn = ThinkFilter()
    sent_reasoning_flag = False

    try:
        async for chunk in graph.astream(request.dict()):
            if "answer" in chunk:
                events, sent_reasoning_flag = await stream_answer(
                    chunk, filter_fn, sent_reasoning_flag
                )
                for e in events:
                    yield e
            elif "evidence" in chunk:
                for e in await stream_evidence(chunk):
                    yield e
            await asyncio.sleep(0.01)
    except asyncio.CancelledError:
        print("Client disconnected")


@router.post("/chat")
async def chat(request: ChatRequest):
    """Handles the chat endpoint."""
    return StreamingResponse(event_generator(request), media_type="text/event-stream")
