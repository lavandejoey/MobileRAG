#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API server for MobileRAG application.
src/api/server.py

@author: LIU Ziyi
@date: 2025-12-30
@license: Apache-2.0
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
import asyncio
import json
import logging
import threading
from typing import Optional
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from src.chat.build_messages import build_llm_messages
from src.chat.think_split import split_think_stream
from src.storage.persist import persist_turn

from src.config import AppConfig, load_config
from src.models.base import GenerationParams
from src.models.registry import create_chat_model
from src.storage.history_db import HistoryDB
from src.rag.pipeline import RagPipeline

CFG: AppConfig = load_config()
HISTORY_DIR = Path(CFG.HISTORY).expanduser()
DB = HistoryDB(db_path=str(HISTORY_DIR / "history.db"))
MODEL = create_chat_model(CFG.MODEL)
RAG = RagPipeline(CFG)

logging.basicConfig(level=CFG.LOG_LEVEL, format="%(levelname)s:\t\t%(message)s")
logger = logging.getLogger("api_server")

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/v1/chats")
def list_chats(limit: int = 200):
    return [chat.to_dict() for chat in DB.list_chats(limit=limit)]


@app.get("/v1/chats/{chat_id}/messages")
def get_messages(chat_id: str, limit: int = 2000):
    return [msg.to_dict() for msg in DB.get_messages(chat_id=chat_id, limit=limit)]


@app.delete("/v1/chats/{chat_id}")
def delete_chat(chat_id: str):
    DB.delete_chat(chat_id=chat_id)
    return {"ok": True}


@app.websocket("/v1/chat/ws")
async def chat_ws(websocket: WebSocket):
    await websocket.accept()
    logger.info("WS connect from %s", websocket.client)

    try:
        init_text = await websocket.receive_text()
        init = json.loads(init_text)
    except Exception:
        await websocket.send_text(json.dumps({"event": "error", "error": "bad_init"}))
        await websocket.close()
        return

    session_id = str(init.get("session_id") or "default")
    message = str(init.get("message") or "").strip()
    chat_id = init.get("chat_id")

    print(f"WS init: session_id={session_id}, chat_id={chat_id}, message={message[:100]}")

    if not message:
        await websocket.send_text(json.dumps({"event": "error", "error": "empty_message"}))
        await websocket.close()
        return

    created_new = False
    if not chat_id:
        chat_id = DB.create_chat(title=message[:80])
        created_new = True
        await websocket.send_text(json.dumps({"event": "chat_created", "chat_id": chat_id}))

    DB.add_message(chat_id=chat_id, role="user", content=message)

    rag_context = ""
    rag_docs_payload = []

    params = GenerationParams(
        temperature=CFG.MODEL.TEMPERATURE,
        top_p=CFG.MODEL.TOP_P,
        max_new_tokens=min(CFG.MODEL.MAX_NEW_TOKENS, 8192),
    )

    async def token_stream():
        loop = asyncio.get_running_loop()
        q: asyncio.Queue[Optional[str]] = asyncio.Queue(maxsize=256)

        def _producer():
            try:
                for chunk in MODEL.stream_chat(messages, params):
                    asyncio.run_coroutine_threadsafe(q.put(chunk), loop).result()
            except Exception as e:
                # Send an error marker; consumer will raise.
                asyncio.run_coroutine_threadsafe(q.put(f"__STREAM_ERROR__:{e}"), loop).result()
            finally:
                asyncio.run_coroutine_threadsafe(q.put(None), loop).result()

        t = threading.Thread(target=_producer, daemon=True)
        t.start()

        while True:
            item = await q.get()
            if item is None:
                break
            if isinstance(item, str) and item.startswith("__STREAM_ERROR__:"):
                raise RuntimeError(item.split(":", 1)[1])
            yield item

    think_state = {"mode": "answer", "buf": ""}
    think_started = False
    think_text = []
    answer_text = []

    def _flush_split_buffer() -> None:
        """
        split_think_stream() keeps a short tail in state["buf"] to detect tag boundaries.
        If the stream ends, that tail must be flushed, otherwise the last few chars
        can be silently dropped (most visible as truncated answers).
        """
        tail = (think_state.get("buf") or "")
        if not tail:
            return
        if think_state.get("mode") == "think":
            think_text.append(tail)
        else:
            answer_text.append(tail)
        think_state["buf"] = ""

    t0 = time.perf_counter()
    think_t0: Optional[float] = None
    think_ms: Optional[int] = None

    async def send(obj: dict):
        await websocket.send_text(json.dumps(obj, ensure_ascii=False))

    try:
        await send({"event": "stage", "stage": "retrieval"})
        if CFG.RAG.ENABLED:
            snips = RAG.retrieve(message, top_k=CFG.RAG.TOP_K)
            rag_context = RAG.format_for_prompt(snips, max_chars=CFG.RAG.PROMPT_MAX_CHARS)
            rag_docs_payload = [
                {"path": s.path, "score": s.score, "chunk_id": s.chunk_id, "text": s.text[:800]}
                for s in snips
            ]
        await send({"event": "rag", "docs": rag_docs_payload})

        messages = build_llm_messages(DB, chat_id=chat_id, user_message=message, rag_context=rag_context)
        await send({"event": "stage", "stage": "generation"})

        # Stream tokens from backend
        async for tok in token_stream():
            # tok is a string chunk
            think_part, answer_part = split_think_stream(tok, think_state)

            if think_part:
                if not think_started:
                    think_started = True
                    think_t0 = time.perf_counter()
                    await send({"event": "think_start"})
                think_text.append(think_part)
                await send({"event": "think_token", "token": think_part})

            if answer_part:
                # If we were thinking and now output answer, close thinking once
                if think_started and think_ms is None and think_t0 is not None:
                    think_ms = int((time.perf_counter() - think_t0) * 1000)
                    await send({"event": "think_end", "think_ms": think_ms})
                answer_text.append(answer_part)
                await send({"event": "answer_token", "token": answer_part})

        # IMPORTANT: flush any remaining buffered tail from split_think_stream()
        _flush_split_buffer()

        total_ms = int((time.perf_counter() - t0) * 1000)

        # If model never produced answer but did think, still close it
        if think_started and think_ms is None and think_t0 is not None:
            think_ms = int((time.perf_counter() - think_t0) * 1000)
            await send({"event": "think_end", "think_ms": think_ms})

        # Persist assistant outputs
        full_think = "".join(think_text).strip()
        full_answer = "".join(answer_text).strip()

        if not full_answer:
            raise RuntimeError("Model returned empty answer")

        meta = {
            "think_ms": think_ms or 0,
            "total_ms": total_ms,
            "created_new": created_new,
            "session_id": session_id,
        }
        persist_turn(DB, chat_id=chat_id, assistant_answer=full_answer, assistant_think=full_think, meta=meta)

        await send({"event": "done", "chat_id": chat_id, "think_ms": meta["think_ms"], "total_ms": total_ms})

    except WebSocketDisconnect:
        return
    except asyncio.CancelledError:
        # This is normal when client disconnects while recv() is waiting.
        return
    except Exception as e:
        logger.exception("WS error: %s", e)
        await send({"event": "error", "error": str(e)})
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
        logger.info("WS close for chat_id=%s", chat_id)
