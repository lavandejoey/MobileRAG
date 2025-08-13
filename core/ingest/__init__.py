from typing import List

from core.types import Chunk, IngestItem

from .chunk import chunk as _chunk
from .scan import scan as _scan


class Ingestor:
    def scan(self, root_dir: str) -> List[IngestItem]:
        return _scan(root_dir)

    def chunk(self, ingest_items: List[IngestItem]) -> List[Chunk]:
        return _chunk(ingest_items)
