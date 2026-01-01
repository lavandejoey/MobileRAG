#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chat command-line interface
src/chat/cli.py

@author: LIU Ziyi
@date: 2025-12-31
@license: Apache-2.0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests

sys.path.append(Path(__file__).resolve().parent.parent.as_posix())

from src.config import load_config


def _http_base(server: str) -> str:
    s = (server or "").strip().rstrip("/")
    if not s:
        return "http://127.0.0.1:8000"
    if s.startswith("http://") or s.startswith("https://"):
        return s
    return "http://" + s


def _ws_url_from_http_base(http_base: str) -> str:
    if http_base.startswith("https://"):
        return "wss://" + http_base[len("https://"):] + "/v1/chat/ws"
    if http_base.startswith("http://"):
        return "ws://" + http_base[len("http://"):] + "/v1/chat/ws"
    # fallback
    return "ws://" + http_base + "/v1/chat/ws"


def _fmt_ms(ms: int) -> str:
    s = ms / 1000.0
    return f"{s:.2f}s"


def _print_chat_list(http_base: str, limit: int = 50) -> None:
    r = requests.get(f"{http_base}/v1/chats", params={"limit": limit}, timeout=30)
    r.raise_for_status()
    chats = r.json() or []
    if not chats:
        print("(no chats)")
        return
    for c in chats:
        cid = str(c.get("chat_id") or "")
        title = str(c.get("title") or "")
        updated_at = c.get("updated_at", None)
        # server stores epoch seconds (float); keep it raw to avoid locale issues in CLI
        updated_str = f"{updated_at:.0f}" if isinstance(updated_at, (int, float)) else str(updated_at or "")
        print(f"- {cid}  |  {updated_str}  |  {title}")


def _print_chat_messages(http_base: str, chat_id: str, limit: int = 2000, show_think: bool = False) -> None:
    r = requests.get(f"{http_base}/v1/chats/{chat_id}/messages", params={"limit": limit}, timeout=30)
    r.raise_for_status()
    msgs = r.json() or []

    pending_think: Optional[str] = None
    pending_meta: Optional[Dict[str, Any]] = None

    for m in msgs:
        role = str(m.get("role") or "")
        content = str(m.get("content") or "")

        if role == "user":
            print(f"\nYou> {content}")
            continue

        if role == "assistant_think":
            pending_think = content
            continue

        if role == "meta":
            try:
                pending_meta = json.loads(content or "{}")
            except Exception:
                pending_meta = None
            continue

        if role == "assistant":
            # print assistant, optionally include think summary
            if pending_meta and isinstance(pending_meta, dict):
                think_ms = int(pending_meta.get("think_ms") or 0)
                if think_ms > 0:
                    print(f"\nAssistant> [Thought for {_fmt_ms(think_ms)}]")
            print(f"\nAssistant> {content}")

            if show_think and pending_think:
                print("\n[think]")
                print(pending_think)

            pending_think = None
            pending_meta = None
            continue

        # ignore other roles by default (e.g., future extensions)


def _delete_chat(http_base: str, chat_id: str) -> None:
    r = requests.delete(f"{http_base}/v1/chats/{chat_id}", timeout=30)
    r.raise_for_status()
    print("Deleted:", chat_id)


async def _ws_chat_once(
        ws_url: str,
        session_id: str,
        chat_id: Optional[str],
        message: str,
        debug_thinking: bool = False,
) -> Tuple[Optional[str], int, int]:
    """
    Returns: (new_chat_id_if_created_or_existing, think_ms, total_ms)
    """
    try:
        import websockets  # type: ignore
    except Exception:
        raise RuntimeError(
            "Missing dependency: websockets. Install with `pip install websockets`."
        )

    # state
    selected_chat_id = chat_id
    think_ms = 0
    total_ms = 0

    # printing state
    saw_assistant_prefix = False
    saw_think = False

    async with websockets.connect(ws_url, ping_interval=20, ping_timeout=20) as ws:
        init: Dict[str, Any] = {"session_id": session_id, "message": message}
        if selected_chat_id:
            init["chat_id"] = selected_chat_id
        await ws.send(json.dumps(init, ensure_ascii=False))

        async for raw in ws:
            try:
                obj = json.loads(raw)
            except Exception:
                continue

            ev = str(obj.get("event") or "")

            if ev == "chat_created":
                selected_chat_id = str(obj.get("chat_id") or "") or selected_chat_id
                # No extra print; keep CLI clean

            elif ev == "stage":
                # optional: keep silent; can enable for debugging
                pass

            elif ev == "rag":
                # RAG placeholder: keep silent
                pass

            elif ev == "think_start":
                saw_think = True
                if debug_thinking:
                    print("\n[think] ", end="", flush=True)

            elif ev == "think_token":
                if debug_thinking:
                    t = str(obj.get("token") or "")
                    if t:
                        sys.stdout.write(t)
                        sys.stdout.flush()

            elif ev == "think_end":
                think_ms = int(obj.get("think_ms") or 0)
                if debug_thinking:
                    sys.stdout.write("\n")
                    sys.stdout.flush()

            elif ev == "answer_token":
                t = str(obj.get("token") or "")
                if not saw_assistant_prefix:
                    # Print a single assistant prefix right before first token
                    if saw_think and think_ms > 0:
                        print(f"\nAssistant> [Thought for {_fmt_ms(think_ms)}]")
                    else:
                        print("\nAssistant> ", end="")
                    saw_assistant_prefix = True

                if t:
                    sys.stdout.write(t)
                    sys.stdout.flush()

            elif ev == "done":
                total_ms = int(obj.get("total_ms") or 0)
                # Ensure newline
                sys.stdout.write("\n")
                sys.stdout.flush()
                break

            elif ev == "error":
                err = str(obj.get("error") or "unknown")
                raise RuntimeError(err)

    return selected_chat_id, think_ms, total_ms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mobile_rag.yaml")
    ap.add_argument("--server", default="http://127.0.0.1:8000", help="HTTP base URL, e.g. http://127.0.0.1:8000")
    ap.add_argument("--session", default="default")
    ap.add_argument("--chat-id", default="", help="Use an existing chat id (optional)")
    ap.add_argument("--debug-thinking", action="store_true", help="Print think tokens (from server events)")
    ap.add_argument("--list", action="store_true", help="List chats and exit")
    ap.add_argument("--load", default="", help="Load chat messages by chat_id and exit")
    ap.add_argument("--show-think", action="store_true", help="When loading history, show full assistant_think blocks")
    ap.add_argument("--delete", default="", help="Delete chat by chat_id and exit")
    args = ap.parse_args()

    # Keep config loading for compatibility; not strictly required by WS-only CLI
    try:
        _ = load_config(args.config)
    except Exception:
        # Do not hard-fail; WS-only CLI can still run
        pass

    http_base = _http_base(args.server)
    ws_url = _ws_url_from_http_base(http_base)

    # one-shot commands
    if args.list:
        _print_chat_list(http_base, limit=200)
        return 0

    if args.load:
        _print_chat_messages(http_base, chat_id=args.load.strip(), show_think=args.show_think)
        return 0

    if args.delete:
        _delete_chat(http_base, chat_id=args.delete.strip())
        return 0

    selected_chat_id = (args.chat_id or "").strip() or None

    print("Chat ready. Type /exit to quit.")
    print("Commands: /new, /list, /load <chat_id>, /del <chat_id>")

    while True:
        try:
            user_text = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return 0

        if not user_text:
            continue

        if user_text.lower() in {"/exit", "exit", "quit", "/quit"}:
            print("Bye.")
            return 0

        if user_text.lower() in {"/new", "new"}:
            selected_chat_id = None
            print("(new chat)")
            continue

        if user_text.lower() in {"/list", "list"}:
            try:
                _print_chat_list(http_base, limit=200)
            except Exception as e:
                print(f"[error] {e}")
            continue

        if user_text.lower().startswith("/load "):
            cid = user_text.split(None, 1)[1].strip()
            if not cid:
                print("[error] missing chat_id")
                continue
            try:
                _print_chat_messages(http_base, chat_id=cid, show_think=args.show_think)
                selected_chat_id = cid
            except Exception as e:
                print(f"[error] {e}")
            continue

        if user_text.lower().startswith("/del "):
            cid = user_text.split(None, 1)[1].strip()
            if not cid:
                print("[error] missing chat_id")
                continue
            try:
                _delete_chat(http_base, chat_id=cid)
                if selected_chat_id == cid:
                    selected_chat_id = None
            except Exception as e:
                print(f"[error] {e}")
            continue

        # normal message: send via WS
        try:
            import asyncio

            new_chat_id, think_ms, total_ms = asyncio.run(
                _ws_chat_once(
                    ws_url=ws_url,
                    session_id=str(args.session),
                    chat_id=selected_chat_id,
                    message=user_text,
                    debug_thinking=bool(args.debug_thinking),
                )
            )
            selected_chat_id = new_chat_id or selected_chat_id

            # keep a tiny footer for timing if you want; currently silent by default
            # print(f"[meta] think_ms={think_ms} total_ms={total_ms} chat_id={selected_chat_id}")

        except Exception as e:
            print(f"[error] {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
