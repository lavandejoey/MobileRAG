# -*- coding: utf-8 -*-
"""
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/14
@version: 0.5.0
"""

import argparse
import json

import numpy as np

from core.config.settings import Settings
from core.ingest import Ingestor
from core.types import Chunk, IngestItem
from core.vecdb.client import VecDB


def _handle_scan(args, ingestor: Ingestor):
    print(f"Scanning directory: {args.root}")
    ingest_items = ingestor.scan(args.root)
    with open(args.output, "w") as f:
        json.dump([item.__dict__ for item in ingest_items], f, indent=4)
    print(f"Scanned {len(ingest_items)} items. Output to {args.output}")


def _handle_embed(args, ingestor: Ingestor):
    print(f"Embedding documents from: {args.input}")
    with open(args.input, "r") as f:
        scanned_data = json.load(f)
    ingest_items = [IngestItem(**item) for item in scanned_data]

    chunks = ingestor.chunk(ingest_items)
    print(f"Chunked {len(chunks)} chunks.")

    text_chunks = [c for c in chunks if c.meta.get("modality") == "text"]
    image_ingest_items = [item for item in ingest_items if item.modality == "image"]

    dense_embeddings = ingestor.embed_dense(text_chunks)
    sparse_embeddings = ingestor.embed_sparse(text_chunks)

    image_embeddings = np.array([])
    captions = []
    if image_ingest_items:
        image_embeddings = ingestor.embed_image(image_ingest_items)
        captions = ingestor.caption_images(image_ingest_items)

    embedded_data = []
    text_chunk_idx = 0
    image_item_idx = 0

    for chunk in chunks:
        data = chunk.__dict__.copy()
        if chunk.meta.get("modality") == "text":
            data["dense_embedding"] = dense_embeddings[text_chunk_idx]
            data["sparse_embedding"] = sparse_embeddings[text_chunk_idx]
            text_chunk_idx += 1
        elif chunk.meta.get("modality") == "image":
            if image_embeddings.size > 0:
                data["image_embedding"] = image_embeddings[image_item_idx].tolist()
            if captions:
                data["caption"] = captions[image_item_idx]
            image_item_idx += 1
        embedded_data.append(data)

    with open(args.output, "w") as f:
        json.dump(embedded_data, f, indent=4)
    print(f"Embedded {len(embedded_data)} chunks. Output to {args.output}")


def _handle_upsert(args, ingestor: Ingestor):
    print(f"Upserting documents from: {args.input}")
    with open(args.input, "r") as f:
        embedded_data = json.load(f)

    chunks_to_upsert = []
    dense_embeddings_to_upsert = []
    sparse_embeddings_to_upsert = []
    image_embeddings_to_upsert = []
    ingest_items_to_upsert = []
    captions_to_upsert = []

    for data in embedded_data:
        chunk = Chunk(
            doc_id=data["doc_id"],
            chunk_id=data["chunk_id"],
            content=data["content"],
            lang=data["lang"],
            meta=data["meta"],
            page=data.get("page"),
            bbox=data.get("bbox"),
        )
        chunks_to_upsert.append(chunk)
        dense_embeddings_to_upsert.append(data.get("dense_embedding"))
        sparse_embeddings_to_upsert.append(data.get("sparse_embedding"))

        modality = data["meta"].get("modality", "text")
        ingest_item = IngestItem(
            path=data["meta"]["file_path"],
            doc_id=data["doc_id"],
            modality=modality,
            meta=data["meta"],
        )
        ingest_items_to_upsert.append(ingest_item)

        if modality == "image":
            image_embeddings_to_upsert.append(data.get("image_embedding"))
            captions_to_upsert.append(data.get("caption"))
        else:
            image_embeddings_to_upsert.append(None)
            captions_to_upsert.append(None)

    actual_image_embeddings = []
    actual_captions = []
    actual_image_ingest_items = []

    for i, item in enumerate(ingest_items_to_upsert):
        if item.modality == "image":
            actual_image_ingest_items.append(item)
            actual_image_embeddings.append(image_embeddings_to_upsert[i])
            actual_captions.append(captions_to_upsert[i])

    ingestor.upsert(
        chunks=chunks_to_upsert,
        dense_embeddings=dense_embeddings_to_upsert,
        sparse_embeddings=sparse_embeddings_to_upsert,
        image_embeddings=(
            np.array(actual_image_embeddings) if actual_image_embeddings else np.array([])
        ),
        ingest_items=actual_image_ingest_items,
        captions=actual_captions,
    )
    print("Upsert command executed.")


def main():
    parser = argparse.ArgumentParser(description="MobileRAG Ingestion CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Scan command
    scan_parser = subparsers.add_parser("scan", help="Scan a directory for documents.")
    scan_parser.add_argument("--root", type=str, required=True, help="Root directory to scan.")
    scan_parser.add_argument(
        "--output",
        type=str,
        default="scanned_items.json",
        help="Output JSON file for scanned items.",
    )

    # Embed command
    embed_parser = subparsers.add_parser("embed", help="Embed scanned documents.")
    embed_parser.add_argument(
        "--input",
        type=str,
        default="scanned_items.json",
        help="Input JSON file with scanned items.",
    )
    embed_parser.add_argument(
        "--output",
        type=str,
        default="embedded_chunks.json",
        help="Output JSON file for embedded chunks.",
    )

    # Upsert command
    upsert_parser = subparsers.add_parser(
        "upsert", help="Upsert embedded chunks to vector database."
    )
    upsert_parser.add_argument(
        "--input",
        type=str,
        default="embedded_chunks.json",
        help="Input JSON file with embedded chunks.",
    )

    args = parser.parse_args()

    settings = Settings()
    vecdb = VecDB(settings)
    vecdb.create_collections()  # Ensure collections are created
    ingestor = Ingestor(settings, vecdb)

    if args.command == "scan":
        _handle_scan(args, ingestor)
    elif args.command == "embed":
        _handle_embed(args, ingestor)
    elif args.command == "upsert":
        _handle_upsert(args, ingestor)

    vecdb.close()


if __name__ == "__main__":
    main()
