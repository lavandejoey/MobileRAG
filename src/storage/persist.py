#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Persist chat turns to the history database.
src/storage/persist.py

@author: LIU Ziyi
@date: 2025-12-31
@license: Apache-2.0
"""
from __future__ import annotations

import json
import uuid
from typing import Dict, Any

from src.storage.history_db import HistoryDB


def persist_turn(
        db: HistoryDB,
        chat_id: str,
        assistant_answer: str,
        assistant_think: str,
        meta: Dict[str, Any],
) -> None:
    turn_id = str(uuid.uuid4())
    if assistant_think:
        db.add_message(chat_id=chat_id, role="assistant_think", content=assistant_think, turn_id=turn_id)
    db.add_message(chat_id=chat_id, role="meta", content=json.dumps(meta, ensure_ascii=False), turn_id=turn_id)
    db.add_message(chat_id=chat_id, role="assistant", content=assistant_answer, turn_id=turn_id)
