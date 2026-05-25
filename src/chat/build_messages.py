#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build messages for LLM input.
src/chat/build_messages.py

@author: LIU Ziyi
@date: 2025-12-31
@license: Apache-2.0
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Dict

from src.chat.system_prompt import SYSTEM_PROMPT
from src.rag.types import RagSnippet
from src.storage.history_db import HistoryDB

Message = Dict[str, str]


def format_rag_context(snips: List[RagSnippet], max_chars: int = 6000) -> str:
    parts: List[str] = []
    total = 0
    for snip in snips:
        cite = snip.citation_id or "FX"
        label = Path(snip.path).name
        location = f" ({snip.source_label})" if snip.source_label else ""
        header = f"[{cite}] {label}{location}\n"
        body = (snip.text or "").strip() + "\n"
        block = header + body + "\n"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "".join(parts).strip()


def build_llm_messages(
        db: HistoryDB,
        chat_id: str,
        rag_context: str = "",
) -> List[Message]:
    past = db.get_messages(chat_id=chat_id, limit=2000)
    history_msgs = [{"role": m.role, "content": m.content} for m in past if m.role in ("user", "assistant")]

    sys_content = SYSTEM_PROMPT
    if rag_context.strip():
        sys_content = sys_content + "\n\n## Retrieval Augmented Context\n" + rag_context.strip()

    return [{"role": "system", "content": sys_content}] + history_msgs
