# -*- coding: utf-8 -*-
"""
@file: scripts/ingest_cli.py
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/14
@version: 0.5.0
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from glob import glob

from tqdm import tqdm

from core.config.devices import resolve_devices
from core.config.settings import Settings
from core.ingest.caption import ImageCaptioner
from core.ingest.embed_dense import DenseEmbedder
from core.ingest.embed_image import ImageEmbedder
from core.ingest.embed_sparse import SparseEmbedder
from core.ingest.pipeline import IngestPipeline
from core.ingest.upsert import upsert_factory
from core.vecdb.client import VecDB


def main():
    parser = argparse.ArgumentParser(description="MobileRAG Ingestion CLI")
    parser.add_argument("--root", type=str, required=True, help="Root directory to scan.")
    args = parser.parse_args()

    settings = Settings()
    vecdb = VecDB(settings)
    vecdb.create_collections()  # Ensure collections are created

    resolved_devices = resolve_devices()
    dense_embedder = DenseEmbedder(str(resolved_devices["embed"]))
    image_embedder = ImageEmbedder(str(resolved_devices["embed"]))
    image_captioner = ImageCaptioner(str(resolved_devices["embed"]))
    sparse_embedder = SparseEmbedder()

    ingest_pipeline = IngestPipeline(
        vecdb=vecdb,
        dense_embedder=dense_embedder,
        sparse_embedder=sparse_embedder,
        image_embedder=image_embedder,
        image_captioner=image_captioner,
    )

    upsert = upsert_factory(vecdb_client=vecdb, ingest_pipeline=ingest_pipeline)

    files = glob(os.path.join(args.root, "**/*"), recursive=True)
    files = [f for f in files if os.path.isfile(f)]

    print(f"Found {len(files)} files to ingest.")

    prc_bar = tqdm(files, desc="Ingesting files", unit="file", dynamic_ncols=True)
    prc_bar.set_postfix(file="")
    for file_path in prc_bar:
        prc_bar.set_postfix(file=file_path.split("/")[-1])
        try:
            upsert(
                user_id="cli_user",
                collection_name=settings.vectorstore.collection,
                file_path=file_path,
            )
        except Exception as e:
            print(f"Failed to ingest {file_path}: {e}")

    vecdb.close()
    print("Ingestion complete.")


if __name__ == "__main__":
    main()
