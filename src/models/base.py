#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base classes and protocols for chat models.
src/models/base.py

@author: LIU Ziyi
@date: 2025-12-30
@license: Apache-2.0
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Protocol


@dataclass(frozen=True)
class GenerationParams:
    temperature: float = 0.2
    top_p: float = 0.9
    max_new_tokens: int = 512


Message = Dict[str, str]  # {"role": "...", "content": "..."}


class ChatModel(Protocol):
    def stream_chat(
            self, messages: List[Message], params: GenerationParams
    ) -> Iterable[str]: ...

    def chat(self, messages: List[Message], params: GenerationParams) -> str: ...
