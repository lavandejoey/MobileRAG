#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chat protocol event types.
src/chat/events.py

@author: LIU Ziyi
@date: 2025-12-31
@license: Apache-2.0
"""
from __future__ import annotations

CHAT_CREATED = "chat_created"
STAGE = "stage"
RAG = "rag"

THINK_START = "think_start"
THINK_TOKEN = "think_token"
THINK_END = "think_end"

ANSWER_TOKEN = "answer_token"
DONE = "done"
ERROR = "error"
