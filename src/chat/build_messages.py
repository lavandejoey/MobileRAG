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

from typing import List, Dict

from src.chat.system_prompt import SYSTEM_PROMPT
from src.storage.history_db import HistoryDB

Message = Dict[str, str]


def build_llm_messages(
        db: HistoryDB,
        chat_id: str,
        user_message: str,
        rag_context: str = "",
) -> List[Message]:
    past = db.get_messages(chat_id=chat_id, limit=2000)
    history_msgs = [{"role": m.role, "content": m.content} for m in past if m.role in ("user", "assistant")]

    sys_content = SYSTEM_PROMPT
    if rag_context.strip():
        sys_content = sys_content + "\n\n## Retrieval Augmented Context\n" + rag_context.strip()

    return [{"role": "system", "content": sys_content}] + history_msgs
