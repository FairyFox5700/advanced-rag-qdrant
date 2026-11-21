"""Pure Reciprocal Rank Fusion math, split out from app.retrieval.hybrid so
it (and the adaptive routing heuristic / graph extraction fallbacks) can be
unit-tested without pulling in fastembed/qdrant-client/sentence-transformers.
"""
from __future__ import annotations


def reciprocal_rank_fusion(
    dense_ranked_ids: list[str],
    sparse_ranked_ids: list[str],
    k: int = 60,
) -> dict[str, float]:
    """Standard RRF: score(d) = sum_over_rankers 1 / (k + rank)."""
    scores: dict[str, float] = {}
    for rank, doc_id in enumerate(dense_ranked_ids, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    for rank, doc_id in enumerate(sparse_ranked_ids, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores
