#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ollama chat model implementation.
src/models/ollama.py

@author: LIU Ziyi
@date: 2025-12-30
@license: Apache-2.0
"""
from __future__ import annotations

import json
import logging
from typing import Iterable, List

import requests

from .base import ChatModel, GenerationParams, Message

logger = logging.getLogger("ollama")


class OllamaChatModel(ChatModel):
    def __init__(self, model: str, think: bool, base_url: str = "http://localhost:11434"):
        self.model = model
        self.think = think
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()
        self._model_ready = False

    def _ensure_model_ready(self) -> None:
        if self._model_ready:
            return
        try:
            resp = self._session.post(
                f"{self.base_url}/api/show", json={"model": self.model}, timeout=30
            )
            if resp.status_code == 404:
                raise ValueError(
                    f"Ollama model '{self.model}' not found. Run `ollama pull {self.model}` or update MODEL_NAME."
                )
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"Failed to verify Ollama model '{self.model}': {exc}") from exc
        self._model_ready = True

    def _raise_for_status(self, response: requests.Response) -> None:
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            if response.status_code == 404:
                raise ValueError(
                    f"Ollama model '{self.model}' not available at {self.base_url}."
                ) from exc
            raise

    def stream_chat(
            self, messages: List[Message], params: GenerationParams
    ) -> Iterable[str]:
        think_open = False

        self._ensure_model_ready()
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": params.temperature,
                "top_p": params.top_p,
                "num_predict": params.max_new_tokens,
            },
        }
        if self.think:
            payload["keep_alive"] = -1
            payload["think"] = True
        try:
            logger.debug("Ollama stream payload: %s", json.dumps(payload, ensure_ascii=False, indent=2)[:2000])
        except Exception:
            logger.exception("Failed to serialize Ollama payload for logging")
        with self._session.post(url, json=payload, stream=True, timeout=600) as r:
            self._raise_for_status(r)
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    logger.debug("Ollama stream non-json line: %s", line)
                    continue
                # logger.debug("Ollama stream line obj: %s", obj)
                # obj examples:
                # {'model': 'qwen3:4b-thinking', 'created_at': '2025-12-31T17:23:34.679427401Z', 'message': {'role': 'assistant', 'content': '', 'thinking': 'Okay'}, 'done': False}
                # {'model': 'qwen3:4b-thinking', 'created_at': '2025-12-31T17:23:47.066107705Z', 'message': {'role': 'assistant', 'content': 'Hello'}, 'done': False}
                # {'model': 'qwen3:4b-thinking', 'created_at': '2025-12-31T17:23:47.571839233Z', 'message': {'role': 'assistant', 'content': ''}, 'done': True, 'done_reason': 'stop', 'total_duration': 13262200995, 'load_duration': 88595972, 'prompt_eval_count': 34, 'prompt_eval_duration': 278191196, 'eval_count': 747, 'eval_duration': 12621372075}
                msg = obj.get("message") or {}
                think = msg.get("thinking") or ""
                content = msg.get("content") or ""

                if self.think and think:
                    # Emit a continuous <think> stream rather than wrapping each chunk.
                    if not think_open:
                        yield "<think>"
                        think_open = True
                    yield think
                if content:
                    # logger.debug(f"Ollama stream content: {content}")
                    if think_open:
                        yield "</think>"
                        think_open = False
                    yield content

                if obj.get("done"):
                    # logger.debug(f"Ollama stream done: {obj}")
                    if think_open:
                        yield "</think>"
                        think_open = False
                    break

    def chat(self, messages: List[Message], params: GenerationParams) -> str:
        self._ensure_model_ready()
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": params.temperature,
                "top_p": params.top_p,
                "num_predict": params.max_new_tokens,
            },
        }
        if self.think:
            payload["think"] = True
        try:
            logger.debug("Ollama chat payload: %s", json.dumps(payload, ensure_ascii=False)[:2000])
        except Exception:
            logger.exception("Failed to serialize Ollama payload for logging")
        r = self._session.post(url, json=payload, timeout=600)
        self._raise_for_status(r)
        try:
            obj = r.json()
        except Exception:
            logger.exception("Failed to parse Ollama chat response as JSON: %s", getattr(r, 'text', None))
            return ""
        logger.debug("Ollama chat response: %s", obj)
        return (obj.get("message") or {}).get("content") or ""
