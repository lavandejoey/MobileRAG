# -*- coding: utf-8 -*-
"""
@file: core/ingest/chunk.py
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/14
@version: 0.5.0
"""

from typing import List

from unstructured.partition.auto import partition

from core.types import Chunk, IngestItem


def chunk(ingest_items: List[IngestItem]) -> List[Chunk]:
    """
    Chunks text and PDF files from IngestItems.
    """
    chunks: List[Chunk] = []

    for item in ingest_items:
        if item.modality == "text":
            # Handle text files (including PDFs treated as text by scan)
            try:
                elements = partition(filename=item.path)
                for i, element in enumerate(elements):
                    # For simplicity, using element.text as content and a basic chunk_id
                    # More sophisticated chunking might be needed based on element type, size, etc.
                    chunk_id = f"{item.doc_id}#chunk_{i}"
                    chunk_meta = item.meta.copy()
                    chunk_meta["modality"] = item.modality
                    chunks.append(
                        Chunk(
                            doc_id=item.doc_id,
                            chunk_id=chunk_id,
                            content=str(element.text),
                            lang="en",  # Placeholder, language detection can be added later
                            meta=chunk_meta,
                            page=(
                                element.metadata.page_number
                                if hasattr(element.metadata, "page_number")
                                else None
                            ),
                            bbox=(
                                tuple(element.metadata.bbox)
                                if hasattr(element.metadata, "bbox")
                                else None
                            ),
                        )
                    )
            except Exception as e:
                print(f"Error processing text/PDF file {item.path}: {e}")
        elif item.modality == "image":
            # Image files are not chunked in this phase, their captions will be handled later
            pass
    return chunks
