from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional


@dataclass
class IngestItem:
    path: str
    doc_id: str
    modality: Literal["text", "image"]
    meta: Dict[str, Any]


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    content: str
    lang: str
    meta: Dict[str, Any]
    # Optional fields for text/PDF
    page: Optional[int] = None
    bbox: Optional[tuple[int, int, int, int]] = None
