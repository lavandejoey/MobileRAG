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

import asyncio
import json
import logging
import os
from pathlib import Path
import re
import threading
import time
from contextlib import asynccontextmanager
from typing import Optional, cast

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.chat.build_messages import build_llm_messages
from src.chat.build_messages import format_rag_context
from src.chat.think_split import split_think_stream
from src.config import AppConfig, load_config
from src.models.base import GenerationParams
from src.models.base import ChatModel
from src.models.registry import create_chat_model
from src.rag.pipeline import RagPipeline
from src.storage.history_db import HistoryDB
from src.storage.persist import persist_turn

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"

logger = logging.getLogger("api_server")
RECALL_HISTORY_PATTERNS = (
    re.compile(r"(前面|之前|刚才|上面).{0,8}(问|说|提到).{0,8}(什么)"),
    re.compile(r"(我).{0,4}(前面|之前|刚才|上面).{0,8}(问|说|提到).{0,8}(什么)"),
    re.compile(r"(你记得|帮我回顾|回顾一下).{0,12}(前面|之前|刚才|上面)"),
)


def _state_cfg(app: FastAPI) -> AppConfig:
    return cast(AppConfig, app.state.cfg)


def _state_db(app: FastAPI) -> HistoryDB:
    return cast(HistoryDB, app.state.db)


def _state_model(app: FastAPI) -> ChatModel:
    return cast(ChatModel, app.state.model)


def _state_rag(app: FastAPI) -> RagPipeline:
    return cast(RagPipeline, app.state.rag)


def _health_payload(app: FastAPI) -> dict:
    cfg = _state_cfg(app)
    rag = _state_rag(app)
    history_dir = Path(cfg.HISTORY).expanduser()
    index_dir = Path(cfg.RAG.INDEX_DIR).expanduser()
    return {
        "ok": True,
        "service": "MobileRAG",
        "model_backend": cfg.MODEL.BACKEND,
        "model_name": cfg.MODEL.MODEL_NAME,
        "rag_enabled": cfg.RAG.ENABLED,
        "history_db": str(history_dir / "history.db"),
        "rag_index_dir": str(index_dir),
        "rag_index_ready": rag.vindex.exists(),
        "model_ready": bool(getattr(_state_model(app), "_model_ready", False)),
    }


def _looks_like_history_recall(message: str) -> bool:
    text = " ".join((message or "").strip().split())
    if not text:
        return False
    return any(p.search(text) for p in RECALL_HISTORY_PATTERNS)


def _build_history_recall_answer(db: HistoryDB, chat_id: str, current_message: str) -> str | None:
    if not _looks_like_history_recall(current_message):
        return None

    past = db.get_messages(chat_id=chat_id, limit=2000)
    user_msgs = [m.content.strip() for m in past if m.role == "user" and m.content.strip()]
    if not user_msgs:
        return "这段对话里还没有可回顾的历史提问。"

    # Exclude the current recall question itself.
    earlier_user_msgs = user_msgs[:-1]
    if not earlier_user_msgs:
        return "在这条问题之前，你还没有提出更早的用户问题。"

    recent = earlier_user_msgs[-8:]
    lines = [f"{idx}. {text}" for idx, text in enumerate(recent, start=1)]
    return (
        "你前面问过我的内容如下：\n\n"
        + "\n".join(lines)
        + "\n\n如果你要，我可以继续把这些问题对应的答案也一起回顾出来。"
    )


def _assign_citation_ids(snips: list) -> tuple[list, list[dict]]:
    doc_to_citation: dict[str, str] = {}
    docs: list[dict] = []
    next_idx = 1
    assigned = []
    for snip in snips:
        citation_id = doc_to_citation.get(snip.doc_id)
        if citation_id is None:
            citation_id = f"F{next_idx}"
            next_idx += 1
            doc_to_citation[snip.doc_id] = citation_id
            docs.append(
                {
                    "citation_id": citation_id,
                    "doc_id": snip.doc_id,
                    "path": snip.path,
                    "name": Path(snip.path).name,
                    "open_url": f"/v1/files/{snip.doc_id}",
                }
            )
        assigned.append(
            type(snip)(
                chunk_id=snip.chunk_id,
                doc_id=snip.doc_id,
                path=snip.path,
                score=snip.score,
                text=snip.text,
                source_label=snip.source_label,
                citation_id=citation_id,
            )
        )
    return assigned, docs


def create_app(config_path: str | None = None) -> FastAPI:
    resolved_config_path = config_path or os.environ.get("MOBILERAG_CONFIG", "configs/mobile_rag.yaml")
    cfg = load_config(resolved_config_path)
    logging.basicConfig(level=cfg.LOG_LEVEL, format="%(levelname)s:\t\t%(message)s")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        history_dir = Path(cfg.HISTORY).expanduser()
        app.state.cfg = cfg
        app.state.db = HistoryDB(db_path=str(history_dir / "history.db"))
        app.state.model = create_chat_model(cfg.MODEL)
        app.state.rag = RagPipeline(cfg)
        try:
            await asyncio.to_thread(app.state.model.prepare)
        except Exception:
            logger.exception("Model warmup failed")
        try:
            warmup_result = await asyncio.to_thread(app.state.rag.warmup, True)
            logger.info("RAG warmup: %s", warmup_result)
        except Exception:
            logger.exception("RAG warmup failed")
        yield

    app = FastAPI(lifespan=lifespan)

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

    @app.exception_handler(404)
    async def spa_fallback(request: Request, exc: StarletteHTTPException) -> Response:
        path = request.url.path or "/"
        if request.method != "GET":
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        if path.startswith("/v1/") or path.startswith("/static/") or path in {"/openapi.json", "/docs", "/redoc", "/favicon.ico", "/robots.txt"}:
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        return FileResponse(STATIC_DIR / "index.html")


    @app.get("/healthz")
    def healthz():
        return JSONResponse(_health_payload(app))


    @app.post("/v1/index/build")
    def build_index():
        return _state_rag(app).build_or_update_index()


    @app.get("/v1/files/{doc_id}")
    def open_file(doc_id: str):
        doc = _state_rag(app).store.get_doc_by_id(doc_id)
        if doc is None:
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        path = Path(doc.path)
        if not path.exists():
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        return FileResponse(
            path,
            media_type=doc.mime,
            filename=path.name,
            headers={"Content-Disposition": f'inline; filename="{path.name}"'},
        )


    @app.get("/v1/chats")
    def list_chats(limit: int = 200):
        return [chat.to_dict() for chat in _state_db(app).list_chats(limit=limit)]


    @app.get("/v1/chats/{chat_id}/messages")
    def get_messages(chat_id: str, limit: int = 2000):
        return [msg.to_dict() for msg in _state_db(app).get_messages(chat_id=chat_id, limit=limit)]


    @app.delete("/v1/chats/{chat_id}")
    def delete_chat(chat_id: str):
        _state_db(app).delete_chat(chat_id=chat_id)
        return {"ok": True}


    @app.websocket("/v1/chat/ws")
    async def chat_ws(websocket: WebSocket):
        db = _state_db(app)
        model = _state_model(app)
        rag = _state_rag(app)
        cfg = _state_cfg(app)

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

        logger.info("WS init session_id=%s chat_id=%s", session_id, chat_id)

        if not message:
            await websocket.send_text(json.dumps({"event": "error", "error": "empty_message"}))
            await websocket.close()
            return

        created_new = False
        if not chat_id:
            chat_id = db.create_chat(first_user_text=message)
            created_new = True
            await websocket.send_text(json.dumps({"event": "chat_created", "chat_id": chat_id}))

        db.add_message(chat_id=chat_id, role="user", content=message)

        snips = []
        citation_docs = []
        rag_context = ""
        rag_docs_payload = []

        params = GenerationParams(
            temperature=cfg.MODEL.TEMPERATURE,
            top_p=cfg.MODEL.TOP_P,
            max_new_tokens=min(cfg.MODEL.MAX_NEW_TOKENS, 8192),
        )

        async def token_stream(messages: list[dict[str, str]]):
            loop = asyncio.get_running_loop()
            q: asyncio.Queue[Optional[str]] = asyncio.Queue(maxsize=256)

            def _producer():
                try:
                    for chunk in model.stream_chat(messages, params):
                        asyncio.run_coroutine_threadsafe(q.put(chunk), loop).result()
                except Exception as e:
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
        think_text: list[str] = []
        answer_text: list[str] = []

        def _flush_split_buffer() -> None:
            tail = think_state.get("buf") or ""
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
            recall_answer = _build_history_recall_answer(db, chat_id=chat_id, current_message=message)
            if recall_answer:
                total_ms = int((time.perf_counter() - t0) * 1000)
                meta = {
                    "think_ms": 0,
                    "total_ms": total_ms,
                    "created_new": created_new,
                    "session_id": session_id,
                    "mode": "history_recall",
                }
                await send({"event": "stage", "stage": "generation"})
                await send({"event": "answer_token", "token": recall_answer})
                persist_turn(
                    db,
                    chat_id=chat_id,
                    assistant_answer=recall_answer,
                    assistant_think="",
                    meta=meta,
                )
                await send({"event": "done", "chat_id": chat_id, "think_ms": 0, "total_ms": total_ms})
                return

            await send({"event": "stage", "stage": "preparing"})
            await send({"event": "stage", "stage": "retrieval"})
            if cfg.RAG.ENABLED:
                snips = rag.retrieve(message, top_k=cfg.RAG.TOP_K)
                snips, citation_docs = _assign_citation_ids(snips)
                rag_context = format_rag_context(snips, max_chars=cfg.RAG.PROMPT_MAX_CHARS)
                rag_docs_payload = [
                    {
                        "citation_id": s.citation_id,
                        "doc_id": s.doc_id,
                        "path": s.path,
                        "score": s.score,
                        "chunk_id": s.chunk_id,
                        "source_label": s.source_label,
                        "text": s.text[:800],
                    }
                    for s in snips
                ]
            await send({"event": "rag", "docs": rag_docs_payload, "citations": citation_docs})

            messages = build_llm_messages(db, chat_id=chat_id, rag_context=rag_context)
            await send({"event": "stage", "stage": "generation"})

            async for tok in token_stream(messages):
                think_part, answer_part = split_think_stream(tok, think_state)

                if think_part:
                    if not think_started:
                        think_started = True
                        think_t0 = time.perf_counter()
                        await send({"event": "think_start"})
                    think_text.append(think_part)
                    await send({"event": "think_token", "token": think_part})

                if answer_part:
                    if think_started and think_ms is None and think_t0 is not None:
                        think_ms = int((time.perf_counter() - think_t0) * 1000)
                        await send({"event": "think_end", "think_ms": think_ms})
                    answer_text.append(answer_part)
                    await send({"event": "answer_token", "token": answer_part})

            _flush_split_buffer()
            total_ms = int((time.perf_counter() - t0) * 1000)

            if think_started and think_ms is None and think_t0 is not None:
                think_ms = int((time.perf_counter() - think_t0) * 1000)
                await send({"event": "think_end", "think_ms": think_ms})

            full_think = "".join(think_text).strip()
            full_answer = "".join(answer_text).strip()

            if not full_answer:
                raise RuntimeError("Model returned empty answer")

            meta = {
                "think_ms": think_ms or 0,
                "total_ms": total_ms,
                "created_new": created_new,
                "session_id": session_id,
                "citations": citation_docs,
            }
            persist_turn(db, chat_id=chat_id, assistant_answer=full_answer, assistant_think=full_think, meta=meta)

            await send({"event": "done", "chat_id": chat_id, "think_ms": meta["think_ms"], "total_ms": total_ms})

        except WebSocketDisconnect:
            return
        except asyncio.CancelledError:
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

    return app


app = create_app()
