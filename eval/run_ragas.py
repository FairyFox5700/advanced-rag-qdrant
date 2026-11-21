"""Runs the 50-question dataset (eval/questions.json) through a chosen
retrieval pipeline and scores it with RAGAS metrics. Checklist item 6.

Usage:
    python -m eval.run_ragas --mode hybrid_rerank_parent
    python -m eval.run_ragas --mode adaptive
    python -m eval.run_ragas --mode basic --limit 10

Modes correspond to the strategies built elsewhere in this repo:
    basic                -> dense-only search (item 1)
    hybrid                -> hybrid RRF search, no reranker (item 2)
    hybrid_rerank_parent  -> hybrid + cross-encoder rerank + parent expansion (items 2,4,5)
    adaptive              -> full LangGraph adaptive pipeline (item 7)
"""
from __future__ import annotations

import argparse
import json
import os

from datasets import Dataset

from app.llm import synthesize_answer
from app.retrieval.adaptive_graph import run_adaptive_query
from app.retrieval.hybrid import dense_only_search, hybrid_search
from app.retrieval.parent_document import expand_to_parents
from app.retrieval.reranker import rerank

QUESTIONS_PATH = os.path.join(os.path.dirname(__file__), "questions.json")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def run_pipeline(question: str, mode: str) -> tuple[str, list[str]]:
    if mode == "basic":
        hits = dense_only_search(question, top_k=5)
    elif mode == "hybrid":
        hits = hybrid_search(question, top_k=5)
    elif mode == "hybrid_rerank_parent":
        hits = hybrid_search(question, top_k=20)
        hits = rerank(question, hits, top_k=5)
        hits = expand_to_parents(hits)
    elif mode == "adaptive":
        state = run_adaptive_query(question)
        contexts = [h.text for h in state.get("hits", [])]
        return state.get("answer", ""), contexts
    else:
        raise ValueError(f"unknown mode: {mode}")

    contexts = [h.text for h in hits]
    answer = synthesize_answer(question, contexts)
    return answer, contexts


def build_ragas_dataset(mode: str, limit: int | None) -> Dataset:
    with open(QUESTIONS_PATH, encoding="utf-8") as fh:
        questions = json.load(fh)
    if limit:
        questions = questions[:limit]

    rows = []
    for item in questions:
        answer, contexts = run_pipeline(item["question"], mode)
        rows.append(
            {
                "question": item["question"],
                "answer": answer,
                "contexts": contexts,
                "ground_truth": item["ground_truth"],
                "category": item["category"],
            }
        )
    return Dataset.from_list(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="hybrid_rerank_parent",
        choices=["basic", "hybrid", "hybrid_rerank_parent", "adaptive"],
    )
    parser.add_argument("--limit", type=int, default=None, help="only run the first N questions")
    args = parser.parse_args()

    dataset = build_ragas_dataset(args.mode, args.limit)

    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"{args.mode}.json")
    df = result.to_pandas()
    df.to_json(out_path, orient="records", indent=2)

    print(f"\n=== RAGAS results ({args.mode}, n={len(dataset)}) ===")
    print(result)
    print(f"\nPer-question results written to {out_path}")


if __name__ == "__main__":
    main()
