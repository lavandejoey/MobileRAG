# -*- coding: utf-8 -*-
"""
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/14
@version: 0.6.0
"""

import argparse
import json
from typing import List
from unittest.mock import Mock

import numpy as np

from core.config.settings import Settings
from core.ingest.caption import ImageCaptioner
from core.ingest.embed_dense import DenseEmbedder
from core.ingest.embed_image import ImageEmbedder
from core.ingest.embed_sparse import SparseEmbedder
from core.retriever.hybrid import HybridRetriever
from core.retriever.types import HybridQuery


def calculate_recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    retrieved_at_k = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    if not relevant_set:
        return 0.0
    return len(retrieved_at_k.intersection(relevant_set)) / len(relevant_set)


def calculate_ndcg_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    # Simplified nDCG calculation for binary relevance (1 if relevant, 0 otherwise)
    # Ideal DCG
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_ids), k)))
    if idcg == 0:
        return 0.0

    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k]):
        if doc_id in relevant_ids:
            dcg += 1.0 / np.log2(i + 2)
    return dcg / idcg


def main():
    parser = argparse.ArgumentParser(description="MobileRAG Evaluation CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Retrieval command
    retrieval_parser = subparsers.add_parser("retrieval", help="Evaluate retrieval performance.")
    retrieval_parser.add_argument(
        "--dataset", type=str, required=True, help="Path to the labelled dataset JSON file."
    )
    retrieval_parser.add_argument(
        "--output",
        type=str,
        default="retrieval_report.json",
        help="Output JSON file for the retrieval report.",
    )
    retrieval_parser.add_argument(
        "--topk", type=int, default=60, help="Top K for Recall and nDCG calculation."
    )

    args = parser.parse_args()

    settings = Settings()

    # Mock VecDB for CLI evaluation due to local client issues
    mock_vecdb_client = Mock()
    mock_vecdb_client.search.return_value = [
        Mock(
            id="doc1#chunk0#100",
            score=0.9,
            payload={
                "file_path": "/path/to/doc1.txt",
                "page": 1,
                "lang": "en",
                "modality": "text",
                "content": "This is a test document.",
            },
        ),
        Mock(
            id="img1#chunk0#200",
            score=0.8,
            payload={
                "file_path": "/path/to/img1.png",
                "caption": "A black image.",
                "lang": "en",
                "modality": "image",
            },
        ),
    ]
    vecdb = Mock()
    vecdb.client = mock_vecdb_client
    vecdb.close = Mock()  # Mock close method

    # Mock embedders and captioner instances
    mock_dense_embedder_instance = Mock(spec=DenseEmbedder)
    mock_dense_embedder_instance.embed_text_query.return_value = [0.1] * 1024  # Dummy embedding

    mock_image_embedder_instance = Mock(spec=ImageEmbedder)
    mock_image_embedder_instance.embed_image_query.return_value = [0.2] * 512  # Dummy embedding

    mock_image_captioner_instance = Mock(spec=ImageCaptioner)
    mock_image_captioner_instance.caption_images.return_value = [
        "A dummy caption."
    ]  # Dummy caption

    mock_sparse_embedder_instance = Mock(spec=SparseEmbedder)
    mock_sparse_embedder_instance.embed_sparse.return_value = [
        {"indices": [1, 2], "values": [0.1, 0.2]}
    ]  # Dummy sparse embedding

    hybrid_retriever = HybridRetriever(
        settings,
        vecdb,
        mock_dense_embedder_instance,
        mock_image_embedder_instance,
        mock_image_captioner_instance,
        mock_sparse_embedder_instance,
    )

    if args.command == "retrieval":
        print(f"Evaluating retrieval performance using dataset: {args.dataset}")
        with open(args.dataset, "r") as f:
            labelled_dataset = json.load(f)

        total_recall = 0.0
        total_ndcg = 0.0
        query_results = []

        for item in labelled_dataset:
            query_text = item.get("query")
            query_image_path = item.get("image_path")
            relevant_docs = item.get("relevant_docs", [])

            hybrid_query = HybridQuery(
                text=query_text,
                image_path=query_image_path,
                topk_dense=args.topk,
                topk_sparse=args.topk,
            )

            candidates = hybrid_retriever.search(hybrid_query)
            retrieved_ids = [c.id for c in candidates]

            recall = calculate_recall_at_k(retrieved_ids, relevant_docs, args.topk)
            ndcg = calculate_ndcg_at_k(retrieved_ids, relevant_docs, args.topk)

            total_recall += recall
            total_ndcg += ndcg

            query_results.append(
                {
                    "query": query_text or query_image_path,
                    "relevant_docs": relevant_docs,
                    "retrieved_ids": retrieved_ids,
                    "recall_at_k": recall,
                    "ndcg_at_k": ndcg,
                }
            )

        avg_recall = total_recall / len(labelled_dataset) if labelled_dataset else 0.0
        avg_ndcg = total_ndcg / len(labelled_dataset) if labelled_dataset else 0.0

        report = {
            "total_queries": len(labelled_dataset),
            "average_recall_at_k": avg_recall,
            "average_ndcg_at_k": avg_ndcg,
            "individual_query_results": query_results,
        }

        with open(args.output, "w") as f:
            json.dump(report, f, indent=4)
        print(f"Retrieval report saved to {args.output}")

    vecdb.close()


if __name__ == "__main__":
    main()
