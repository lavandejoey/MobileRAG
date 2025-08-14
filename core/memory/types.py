# -*- coding: utf-8 -*-
"""
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/14
@version: 0.9.0
"""

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class MemoryCard:
    id: str
    content: str
    metadata: Dict[str, Any]

    def to_payload(self) -> Dict[str, Any]:
        payload = {"id": self.id, "content": self.content}
        payload.update(self.metadata)
        return payload

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]):
        memory_id = payload.pop("id")
        content = payload.pop("content")
        return cls(id=memory_id, content=content, metadata=payload)


@dataclass
class QueryResult:
    memory_card: MemoryCard
    score: float
