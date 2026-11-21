"""Cross-encoder reranking. Checklist item 4.

Bi-encoder (dense/hybrid) retrieval is fast but scores query and document
independently. A cross-encoder jointly attends over (query, document) pairs
for a much more accurate — if slower — relevance score, so we use it as a
second-stage reranker over a modest candidate pool (top_k_dense/sparse).
"""
from __future__ import annotations

from functools import lru_cache

from sentence_transformers import CrossEncoder

from app.config import settings
from app.retrieval.hybrid import HybridHit


@lru_cache(maxsize=1)
def get_cross_encoder() -> CrossEncoder:
    return CrossEncoder(settings.cross_encoder_model)


def rerank(query: str, hits: list[HybridHit], top_k: int | None = None) -> list[HybridHit]:
    if not hits:
        return hits
    top_k = top_k or settings.top_k_final

    model = get_cross_encoder()
    pairs = [(query, hit.text) for hit in hits]
    scores = model.predict(pairs)

    rescored = []
    for hit, score in zip(hits, scores):
        rescored.append(
            HybridHit(
                point_id=hit.point_id,
                text=hit.text,
                source=hit.source,
                parent_id=hit.parent_id,
                score=float(score),
                dense_rank=hit.dense_rank,
                sparse_rank=hit.sparse_rank,
            )
        )
    rescored.sort(key=lambda h: h.score, reverse=True)
    return rescored[:top_k]
