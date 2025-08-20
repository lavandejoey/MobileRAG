# -*- coding: utf-8 -*-
"""
@file: core/graph/nodes/generator.py
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/14
@version: 0.11.0
"""

import re


def generator_node(state, llm_generator):
    """
    Generates a response using the LLM.
    """
    # --- 1) helper to strip any leaked <think>…</think> blocks ---
    # robust across newlines and partial weird spacing
    THINK_RE = re.compile(r"<\s*think\s*>.*?<\s*/\s*think\s*>", re.DOTALL | re.IGNORECASE)

    def strip_think(text: str) -> str:
        if not text:
            return text
        return THINK_RE.sub("", str(text))

    # --- 2) fetch budget parts safely & sanitize them ---
    budget = state.get("budget", {}) or {}
    summary = strip_think(budget.get("summary", "") or "")
    recent_messages = budget.get("recent_messages", []) or []
    memories = budget.get("memories", []) or []
    evidence = budget.get("evidence", []) or []
    normalized_query = strip_think(state.get("normalized_query", "") or "")

    # Sanitize lists → strings, strip any accidental <think> that got into history/memory
    recent_str = "\n".join(strip_think(str(m)) for m in recent_messages if m is not None)
    mem_str = "\n".join(strip_think(str(m)) for m in memories if m is not None)
    ev_str = "\n".join(strip_think(str(e)) for e in evidence if e is not None)

    # --- 3) build a disciplined, labelled prompt ---
    # This format helps the model stay on-topic and NOT guess unknown entities.
    prompt = (
        "## Summary\n"
        f"{summary}\n\n"
        "## Conversation (recent)\n"
        f"{recent_str}\n\n"
        "## Retrieved Memories (may be irrelevant)\n"
        f"{mem_str}\n\n"
        "## Evidence (citations/snippets)\n"
        f"{ev_str}\n\n"
        "## User Query\n"
        f"{normalized_query}\n\n"
        "## Instructions\n"
        "- Answer ONLY based on the query and relevant context above.\n"
        "- If the query mentions an ambiguous entity and the context does not disambiguate it,\n"
        "  ask one SHORT clarifying question instead of guessing.\n"
        "- Do NOT include chain-of-thought, hidden analysis, or any <think> tags."
        " Return ONLY the final answer.\n"
    )

    # --- 4) generate and strip any leaked <think> from the output as a final guard ---
    raw = llm_generator.generate(prompt)
    clean = strip_think(raw)
    return {"generation": clean}
