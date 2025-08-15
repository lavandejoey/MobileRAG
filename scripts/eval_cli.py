# -*- coding: utf-8 -*-
"""
@author: LIU Ziyi
@email: lavandejoey@outlook.com
@date: 2025/08/15
@version: 0.8.0
"""

import argparse
import json
import os
import sys
from typing import List

import numpy as np

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from core.config.devices import resolve_devices
from core.config.settings import Settings
from core.ingest.caption import ImageCaptioner
from core.ingest.embed_dense import DenseEmbedder
from core.ingest.embed_image import ImageEmbedder
from core.ingest.embed_sparse import SparseEmbedder
from core.reranker.reranker import Reranker
from core.retriever.hybrid import HybridRetriever
from core.retriever.types import Candidate, Evidence, HybridQuery
from core.vecdb.client import VecDB


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

    # Rerank command
    rerank_parser = subparsers.add_parser("rerank", help="Evaluate reranking performance.")
    rerank_parser.add_argument(
        "--dataset", type=str, required=True, help="Path to the labelled dataset JSON file."
    )
    rerank_parser.add_argument(
        "--output",
        type=str,
        default="rerank_report.json",
        help="Output JSON file for the rerank report.",
    )
    rerank_parser.add_argument(
        "--topk", type=int, default=10, help="Top K for nDCG calculation after reranking."
    )

    args = parser.parse_args()

    settings = Settings()
    resolved_devices = resolve_devices()

    # Initialize real components
    # Use in-memory Qdrant for evaluation CLI to avoid sqlite3.OperationalError
    vecdb = VecDB(settings, in_memory=True)
    vecdb.create_collections()  # Ensure collections exist for in-memory DB

    dense_embedder = DenseEmbedder(str(resolved_devices["embed"]))
    image_embedder = ImageEmbedder(str(resolved_devices["embed"]))
    image_captioner = ImageCaptioner(str(resolved_devices["embed"]))
    sparse_embedder = SparseEmbedder()
    reranker = Reranker(str(resolved_devices["reranker"]))

    hybrid_retriever = HybridRetriever(
        settings,
        vecdb,
        dense_embedder,
        image_embedder,
        image_captioner,
        sparse_embedder,
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

    elif args.command == "rerank":
        print(f"Evaluating reranking performance using dataset: {args.dataset}")
        with open(args.dataset, "r") as f:
            labelled_dataset = json.load(f)

        total_ndcg = 0.0
        query_results = []

        for item in labelled_dataset:
            query_text = item.get("query")
            # Convert dataset dictionaries to Candidate objects
            initial_candidates_data = item.get("candidates", [])
            initial_candidates = [
                Candidate(
                    id=c["id"],
                    text=c.get("text"),
                    score=c.get("score", 0.0),
                    evidence=Evidence(
                        file_path=c["evidence"]["file_path"],
                        page=c["evidence"].get("page"),
                        caption=c["evidence"].get("caption"),
                    ),
                    lang=c["lang"],
                    modality=c["modality"],
                )
                for c in initial_candidates_data
            ]
            relevant_docs = item.get("relevant_docs", [])

            if not query_text:
                print(f"Skipping item due to missing query: {item}")
                continue

            reranked_candidates = reranker.rank(query_text, initial_candidates, args.topk)
            reranked_ids = [c.id for c in reranked_candidates]

            ndcg = calculate_ndcg_at_k(reranked_ids, relevant_docs, args.topk)
            total_ndcg += ndcg

            query_results.append(
                {
                    "query": query_text,
                    "relevant_docs": relevant_docs,
                    "initial_candidates_ids": [c.id for c in initial_candidates],
                    "reranked_ids": reranked_ids,
                    "ndcg_at_k": ndcg,
                }
            )

        avg_ndcg = total_ndcg / len(labelled_dataset) if labelled_dataset else 0.0

        report = {
            "total_queries": len(labelled_dataset),
            "average_ndcg_at_k": avg_ndcg,
            "individual_query_results": query_results,
        }

        with open(args.output, "w") as f:
            json.dump(report, f, indent=4)
        print(f"Rerank report saved to {args.output}")

    vecdb.close()


if __name__ == "__main__":
    main()
