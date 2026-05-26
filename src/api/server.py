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
import contextlib
import json
import logging
import os
from pathlib import Path
import re
import shutil
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Optional, cast

from fastapi import FastAPI, File, Request, UploadFile, WebSocket, WebSocketDisconnect
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
from src.storage.history_db import HistoryDB, UploadedFileRow
from src.storage.persist import persist_turn

APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"

logger = logging.getLogger("api_server")
RECALL_HISTORY_PATTERNS = (
    re.compile(r"(前面|之前|刚才|上面).{0,8}(问|说|提到).{0,8}(什么)"),
    re.compile(r"(我).{0,4}(前面|之前|刚才|上面).{0,8}(问|说|提到).{0,8}(什么)"),
    re.compile(r"(你记得|帮我回顾|回顾一下).{0,12}(前面|之前|刚才|上面)"),
)
COMPLEX_QUERY_PATTERNS = (
    re.compile(r"\b(compare|comparison|analy[sz]e|analysis|evaluate|evaluation|design|architecture)\b", re.I),
    re.compile(r"\b(why|how|pros|cons|trade[- ]?off|plan|strategy|roadmap)\b", re.I),
    re.compile(r"(比较|分析|评估|设计|架构|为什么|怎么|如何|方案|计划|路线图)"),
)


@dataclass
class ActiveTurn:
    chat_id: str
    backlog: list[dict[str, Any]] = field(default_factory=list)
    subscribers: set[WebSocket] = field(default_factory=set)
    task: asyncio.Task | None = None


def _state_cfg(app: FastAPI) -> AppConfig:
    return cast(AppConfig, app.state.cfg)


def _state_db(app: FastAPI) -> HistoryDB:
    return cast(HistoryDB, app.state.db)


def _state_model(app: FastAPI) -> ChatModel:
    return cast(ChatModel, app.state.model)


def _state_rag(app: FastAPI) -> RagPipeline:
    return cast(RagPipeline, app.state.rag)


def _state_turns(app: FastAPI) -> dict[str, ActiveTurn]:
    return cast(dict[str, ActiveTurn], app.state.active_turns)


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


def _chat_upload_root(cfg: AppConfig, chat_id: str) -> Path:
    return Path(cfg.RAG.UPLOAD_DIR).expanduser() / chat_id


def _safe_upload_name(original_name: str) -> tuple[str, str]:
    path = Path(original_name or "")
    suffix = path.suffix.lower()
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("._-") or "upload"
    return stem, suffix


def _safe_display_path(path: Path) -> str:
    resolved = path.resolve()
    cwd = Path.cwd().resolve()
    try:
        return str(resolved.relative_to(cwd))
    except ValueError:
        return str(resolved)


def _uploaded_file_to_dict(row: UploadedFileRow) -> dict:
    return row.to_dict()


def _resolve_doc_ids_for_uploads(app: FastAPI, uploads: list[dict]) -> list[str]:
    if not uploads:
        return []
    store = _state_rag(app).store
    resolved: list[str] = []
    for item in uploads:
        raw_path = str(item.get("rel_path") or "").strip()
        if not raw_path:
            continue
        try:
            ap = str(Path(raw_path).expanduser().resolve())
        except Exception:
            continue
        doc = store.get_doc_by_path(ap)
        if doc is not None:
            resolved.append(doc.doc_id)
    return resolved


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


def _classify_response_mode(message: str, has_rag_context: bool = False) -> str:
    text = " ".join((message or "").strip().split())
    if not text:
        return "simple"
    if len(text) > 120:
        return "detailed"
    if any(p.search(text) for p in COMPLEX_QUERY_PATTERNS):
        return "detailed"
    if text.count("\n") >= 2:
        return "detailed"
    if has_rag_context and len(text) <= 80:
        return "simple"
    if len(text) <= 60:
        return "simple"
    return "default"


def _compact_citation_id(path: str, used: set[str]) -> str:
    stem = Path(path).stem.upper()
    base = "".join(ch for ch in stem if ch.isalnum())[:8] or "FILE"
    candidate = base
    suffix = 2
    while candidate in used:
        tail = str(suffix)
        candidate = f"{base[:max(1, 8 - len(tail))]}{tail}"
        suffix += 1
    used.add(candidate)
    return candidate


def _assign_citation_ids(snips: list) -> tuple[list, list[dict]]:
    doc_to_citation: dict[str, str] = {}
    used_ids: set[str] = set()
    docs: list[dict] = []
    assigned = []
    for snip in snips:
        citation_id = doc_to_citation.get(snip.doc_id)
        if citation_id is None:
            citation_id = _compact_citation_id(snip.path, used_ids)
            doc_to_citation[snip.doc_id] = citation_id
            docs.append(
                {
                    "citation_id": citation_id,
                    "doc_id": snip.doc_id,
                    "path": snip.path,
                    "name": Path(snip.path).name,
                    "source_label": snip.source_label,
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
        app.state.active_turns = {}
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

    async def _broadcast(chat_id: str, obj: dict[str, Any]) -> None:
        turn = _state_turns(app).get(chat_id)
        if turn is None:
            return
        turn.backlog.append(obj)
        dead: list[WebSocket] = []
        payload = json.dumps(obj, ensure_ascii=False)
        for ws in list(turn.subscribers):
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            turn.subscribers.discard(ws)

    async def _replay_turn(turn: ActiveTurn, websocket: WebSocket) -> None:
        for obj in turn.backlog:
            await websocket.send_text(json.dumps(obj, ensure_ascii=False))

    async def _run_chat_turn(
            chat_id: str,
            session_id: str,
            message: str,
            created_new: bool,
            pending_uploads: int,
            user_msg_id: int | None,
            attached_uploads: list[dict],
            attached_doc_ids: list[str],
    ) -> None:
        db = _state_db(app)
        model = _state_model(app)
        rag = _state_rag(app)
        cfg = _state_cfg(app)

        snips = []
        citation_docs = []
        rag_context = ""
        rag_docs_payload = []
        response_mode = "default"
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

        try:
            if created_new:
                await _broadcast(chat_id, {"event": "chat_created", "chat_id": chat_id})

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
                await _broadcast(chat_id, {"event": "stage", "stage": "generation"})
                await _broadcast(chat_id, {"event": "answer_token", "token": recall_answer})
                persist_turn(db, chat_id=chat_id, assistant_answer=recall_answer, assistant_think="", meta=meta)
                await _broadcast(chat_id, {"event": "done", "chat_id": chat_id, "think_ms": 0, "total_ms": total_ms})
                return

            await _broadcast(chat_id, {"event": "stage", "stage": "preparing"})
            if pending_uploads > 0:
                await _broadcast(chat_id, {"event": "stage", "stage": "parsing"})
                build_result = await asyncio.to_thread(rag.build_or_update_index)
                if user_msg_id is not None:
                    db.mark_uploaded_files_processed(chat_id, user_msg_id)
                attached_doc_ids = _resolve_doc_ids_for_uploads(app, attached_uploads)
                await _broadcast(chat_id, {"event": "uploads_processed", "count": pending_uploads, "index": build_result})
                if not message:
                    total_ms = int((time.perf_counter() - t0) * 1000)
                    ack = f"Processed {pending_uploads} uploaded file{'s' if pending_uploads != 1 else ''}. You can ask about them now."
                    meta = {
                        "think_ms": 0,
                        "total_ms": total_ms,
                        "created_new": created_new,
                        "session_id": session_id,
                        "mode": "uploads_processed",
                        "uploads": attached_uploads,
                    }
                    await _broadcast(chat_id, {"event": "stage", "stage": "generation"})
                    await _broadcast(chat_id, {"event": "answer_token", "token": ack})
                    persist_turn(db, chat_id=chat_id, assistant_answer=ack, assistant_think="", meta=meta)
                    await _broadcast(chat_id, {"event": "done", "chat_id": chat_id, "think_ms": 0, "total_ms": total_ms})
                    return

            await _broadcast(chat_id, {"event": "stage", "stage": "retrieval"})
            if cfg.RAG.ENABLED:
                if not attached_doc_ids and attached_uploads:
                    attached_doc_ids = _resolve_doc_ids_for_uploads(app, attached_uploads)
                snips = rag.retrieve(message, top_k=cfg.RAG.TOP_K, preferred_doc_ids=attached_doc_ids or None)
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
            await _broadcast(chat_id, {"event": "rag", "docs": rag_docs_payload, "citations": citation_docs})

            response_mode = _classify_response_mode(message, has_rag_context=bool(rag_context.strip()))
            if response_mode == "simple":
                params = GenerationParams(
                    temperature=min(cfg.MODEL.TEMPERATURE, 0.2),
                    top_p=cfg.MODEL.TOP_P,
                    max_new_tokens=min(cfg.MODEL.MAX_NEW_TOKENS, 160),
                )
            elif response_mode == "detailed":
                params = GenerationParams(
                    temperature=cfg.MODEL.TEMPERATURE,
                    top_p=cfg.MODEL.TOP_P,
                    max_new_tokens=min(cfg.MODEL.MAX_NEW_TOKENS, 8192),
                )

            messages = build_llm_messages(db, chat_id=chat_id, rag_context=rag_context, response_mode=response_mode)
            await _broadcast(chat_id, {"event": "stage", "stage": "generation"})

            expose_thinking = response_mode != "simple"

            async for tok in token_stream(messages):
                think_part, answer_part = split_think_stream(tok, think_state)

                if think_part:
                    think_text.append(think_part)
                    if expose_thinking:
                        if not think_started:
                            think_started = True
                            think_t0 = time.perf_counter()
                            await _broadcast(chat_id, {"event": "think_start"})
                        await _broadcast(chat_id, {"event": "think_token", "token": think_part})

                if answer_part:
                    if expose_thinking and think_started and think_ms is None and think_t0 is not None:
                        think_ms = int((time.perf_counter() - think_t0) * 1000)
                        await _broadcast(chat_id, {"event": "think_end", "think_ms": think_ms})
                    answer_text.append(answer_part)
                    await _broadcast(chat_id, {"event": "answer_token", "token": answer_part})

            _flush_split_buffer()
            total_ms = int((time.perf_counter() - t0) * 1000)

            if expose_thinking and think_started and think_ms is None and think_t0 is not None:
                think_ms = int((time.perf_counter() - think_t0) * 1000)
                await _broadcast(chat_id, {"event": "think_end", "think_ms": think_ms})

            full_think = "".join(think_text).strip()
            full_answer = "".join(answer_text).strip()
            if not full_answer:
                fallback_answer = full_think or str(think_state.get("buf") or "").strip()
                if fallback_answer:
                    full_answer = fallback_answer
                else:
                    raise RuntimeError("Model returned empty answer")

            meta = {
                "think_ms": think_ms or 0,
                "total_ms": total_ms,
                "created_new": created_new,
                "session_id": session_id,
                "response_mode": response_mode,
                "uploads": attached_uploads,
                "citations": citation_docs,
            }
            persist_turn(
                db,
                chat_id=chat_id,
                assistant_answer=full_answer,
                assistant_think=full_think if expose_thinking else "",
                meta=meta,
            )
            await _broadcast(chat_id, {"event": "done", "chat_id": chat_id, "think_ms": meta["think_ms"], "total_ms": total_ms})
        except Exception as e:
            logger.exception("WS error: %s", e)
            await _broadcast(chat_id, {"event": "error", "error": str(e)})
        finally:
            _state_turns(app).pop(chat_id, None)


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


    @app.post("/v1/chats")
    def create_chat():
        chat_id = _state_db(app).create_empty_chat()
        return {"chat_id": chat_id}


    @app.get("/v1/chats/{chat_id}/uploads")
    def list_chat_uploads(chat_id: str):
        return [_uploaded_file_to_dict(row) for row in _state_db(app).list_uploaded_files(chat_id)]


    @app.post("/v1/chats/{chat_id}/uploads")
    async def upload_file(chat_id: str, file: UploadFile = File(...)):
        cfg = _state_cfg(app)
        db = _state_db(app)
        if db.get_chat(chat_id) is None:
            return JSONResponse({"detail": "chat not found"}, status_code=404)
        original_name = Path(file.filename or "").name
        suffix = Path(original_name).suffix.lower()
        if not original_name:
            return JSONResponse({"detail": "missing filename"}, status_code=400)

        safe_stem, _ = _safe_upload_name(original_name)
        upload_dir = _chat_upload_root(cfg, chat_id)
        upload_dir.mkdir(parents=True, exist_ok=True)

        target = upload_dir / f"{safe_stem}{suffix}"
        counter = 1
        while target.exists():
            target = upload_dir / f"{safe_stem}_{counter}{suffix}"
            counter += 1

        file_size = 0
        try:
            with target.open("wb") as out:
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    file_size += len(chunk)
                    if file_size > cfg.RAG.MAX_FILE_SIZE_MB * 1024 * 1024:
                        out.close()
                        with contextlib.suppress(FileNotFoundError):
                            target.unlink()
                        return JSONResponse({"detail": "file too large"}, status_code=400)
                    out.write(chunk)

            try:
                from src.rag.parsers import parse_file_sections
                parse_file_sections(target)
            except Exception as exc:
                with contextlib.suppress(FileNotFoundError):
                    target.unlink()
                return JSONResponse({"detail": f"unsupported or unreadable file: {exc}"}, status_code=400)

            rel_path = _safe_display_path(target)
            upload_id = db.add_uploaded_file(
                chat_id=chat_id,
                original_name=original_name,
                stored_name=target.name,
                rel_path=rel_path,
            )
        except Exception as exc:
            with contextlib.suppress(FileNotFoundError):
                target.unlink()
            logger.exception("Upload failed for chat_id=%s file=%s", chat_id, original_name)
            return JSONResponse({"detail": f"upload failed: {exc}"}, status_code=500)
        finally:
            await file.close()

        if not target.exists():
            return JSONResponse({"detail": "upload failed"}, status_code=500)

        return {
            "ok": True,
            "chat_id": chat_id,
            "upload_id": upload_id,
            "original_name": original_name,
            "filename": target.name,
            "path": str(target.resolve()),
            "processed": False,
        }


    @app.delete("/v1/chats/{chat_id}/uploads/{upload_id}")
    def delete_upload(chat_id: str, upload_id: int):
        cfg = _state_cfg(app)
        row = _state_db(app).delete_uploaded_file(chat_id=chat_id, upload_id=upload_id)
        if row is None:
            return JSONResponse({"detail": "upload not found"}, status_code=404)
        file_path = _chat_upload_root(cfg, chat_id) / row.stored_name
        with contextlib.suppress(FileNotFoundError):
            file_path.unlink()
        if bool(row.processed):
            _state_rag(app).build_or_update_index()
        return {"ok": True, "upload_id": upload_id}


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
        db = _state_db(app)
        msgs = [msg.to_dict() for msg in db.get_messages(chat_id=chat_id, limit=limit)]
        uploads_by_msg: dict[int, list[dict]] = {}
        for row in db.list_uploaded_files(chat_id):
            if row.attached_msg_id is None:
                continue
            uploads_by_msg.setdefault(int(row.attached_msg_id), []).append(_uploaded_file_to_dict(row))
        for msg in msgs:
            if msg["role"] == "user":
                msg["uploads"] = uploads_by_msg.get(int(msg["msg_id"]), [])
        return msgs


    @app.delete("/v1/chats/{chat_id}")
    def delete_chat(chat_id: str):
        cfg = _state_cfg(app)
        upload_root = _chat_upload_root(cfg, chat_id)
        if upload_root.exists():
            shutil.rmtree(upload_root, ignore_errors=True)
        _state_db(app).delete_chat(chat_id=chat_id)
        _state_rag(app).build_or_update_index()
        return {"ok": True}


    @app.websocket("/v1/chat/ws")
    async def chat_ws(websocket: WebSocket):
        db = _state_db(app)
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

        if chat_id:
            existing_turn = _state_turns(app).get(str(chat_id))
            if existing_turn is not None:
                if message:
                    await websocket.send_text(json.dumps({"event": "error", "error": "chat_busy"}))
                    await websocket.close()
                    return
                existing_turn.subscribers.add(websocket)
                await _replay_turn(existing_turn, websocket)
                try:
                    while True:
                        await websocket.receive_text()
                except WebSocketDisconnect:
                    existing_turn.subscribers.discard(websocket)
                    return

        pending_uploads = len(db.list_pending_uploaded_files(chat_id)) if chat_id else 0

        if not message and pending_uploads == 0:
            await websocket.send_text(json.dumps({"event": "error", "error": "no_active_turn"}))
            await websocket.close()
            return

        created_new = False
        if not chat_id:
            chat_id = db.create_chat(first_user_text=message)
            created_new = True

        pending_uploads = len(db.list_pending_uploaded_files(chat_id))
        attached_uploads: list[dict] = []
        attached_doc_ids: list[str] = []

        if message:
            db.maybe_update_title_from_first_user_text(chat_id=chat_id, first_user_text=message)
            user_msg_id = db.add_message(chat_id=chat_id, role="user", content=message)
            attached_rows = db.attach_pending_uploads_to_message(chat_id=chat_id, msg_id=user_msg_id)
            attached_uploads = [_uploaded_file_to_dict(row) for row in attached_rows]
        elif pending_uploads > 0:
            user_msg_id = db.add_message(chat_id=chat_id, role="user", content="")
            attached_rows = db.attach_pending_uploads_to_message(chat_id=chat_id, msg_id=user_msg_id)
            attached_uploads = [_uploaded_file_to_dict(row) for row in attached_rows]
        else:
            user_msg_id = None

        turn = ActiveTurn(chat_id=str(chat_id))
        turn.subscribers.add(websocket)
        _state_turns(app)[str(chat_id)] = turn
        turn.task = asyncio.create_task(
            _run_chat_turn(
                chat_id=str(chat_id),
                session_id=session_id,
                message=message,
                created_new=created_new,
                pending_uploads=pending_uploads,
                user_msg_id=user_msg_id,
                attached_uploads=attached_uploads,
                attached_doc_ids=attached_doc_ids,
            )
        )
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            turn.subscribers.discard(websocket)
            return
        finally:
            with contextlib.suppress(Exception):
                await websocket.close()
            logger.info("WS close for chat_id=%s", chat_id)

    return app


app = create_app()
