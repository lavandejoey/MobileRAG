#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Splitting think and answer streams.
src/chat/think_split.py

@author: LIU Ziyi
@date: 2025-12-31
@license: Apache-2.0
"""
from __future__ import annotations

from typing import Tuple, Dict


def split_think_stream(token: str, state: Dict[str, str]) -> Tuple[str, str]:
    TAG_OPEN = "<think>"
    TAG_CLOSE = "</think>"

    buf = state["buf"] + token
    state["buf"] = ""

    think_out: list[str] = []
    answer_out: list[str] = []

    while buf:
        if state["mode"] == "answer":
            idx = buf.find(TAG_OPEN)
            if idx == -1:
                safe_len = max(len(buf) - (len(TAG_OPEN) - 1), 0)
                if safe_len:
                    answer_out.append(buf[:safe_len])
                    buf = buf[safe_len:]
                else:
                    break
            else:
                if idx:
                    answer_out.append(buf[:idx])
                buf = buf[idx + len(TAG_OPEN):]
                state["mode"] = "think"
        else:
            idx = buf.find(TAG_CLOSE)
            if idx == -1:
                safe_len = max(len(buf) - (len(TAG_CLOSE) - 1), 0)
                if safe_len:
                    think_out.append(buf[:safe_len])
                    buf = buf[safe_len:]
                else:
                    break
            else:
                if idx:
                    think_out.append(buf[:idx])
                buf = buf[idx + len(TAG_CLOSE):]
                state["mode"] = "answer"

    state["buf"] = buf
    return "".join(think_out), "".join(answer_out)
