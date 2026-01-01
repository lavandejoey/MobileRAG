#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WebSocket client for chat streaming.
src/clients/ws_client.py

@author: LIU Ziyi
@date: 2025-12-31
@license: Apache-2.0
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import AsyncIterator, Dict, Optional

import websockets


@dataclass
class WsEvent:
    event: str
    data: Dict


async def ws_chat_stream(
        server_base: str,
        session_id: str,
        chat_id: Optional[str],
        message: str,
) -> AsyncIterator[WsEvent]:
    base = server_base.rstrip("/")
    ws_url = base.replace("http://", "ws://").replace("https://", "wss://") + "/v1/chat/ws"

    async with websockets.connect(ws_url, ping_interval=20, ping_timeout=20) as ws:
        init = {"session_id": session_id, "message": message}
        if chat_id:
            init["chat_id"] = chat_id
        await ws.send(json.dumps(init, ensure_ascii=False))

        async for raw in ws:
            obj = json.loads(raw)
            ev = str(obj.get("event") or "")
            yield WsEvent(event=ev, data=obj)
