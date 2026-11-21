"""Hybrid search: fuse dense (semantic) and sparse (BM25-style) results with
Reciprocal Rank Fusion. Checklist item 2.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.config import settings
from app.embeddings import embed_dense, embed_sparse
from app.qdrant_store import search_dense, search_sparse
from app.rrf import reciprocal_rank_fusion

__all__ = ["HybridHit", "reciprocal_rank_fusion", "hybrid_search", "dense_only_search"]


@dataclass
class HybridHit:
    point_id: str
    text: str
    source: str
    parent_id: str
    score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None


def hybrid_search(query: str, top_k: int | None = None) -> list[HybridHit]:
    top_k = top_k or settings.top_k_final

    dense_vec = embed_dense([query])[0]
    sparse_vec = embed_sparse([query])[0]

    dense_hits = search_dense(dense_vec, limit=settings.top_k_dense)
    sparse_hits = search_sparse(
        sparse_vec.indices.tolist(), sparse_vec.values.tolist(), limit=settings.top_k_sparse
    )

    dense_ids = [str(h.id) for h in dense_hits]
    sparse_ids = [str(h.id) for h in sparse_hits]

    fused_scores = reciprocal_rank_fusion(dense_ids, sparse_ids, k=settings.rrf_k)

    payload_by_id = {str(h.id): h.payload for h in dense_hits}
    for h in sparse_hits:
        payload_by_id.setdefault(str(h.id), h.payload)

    dense_rank = {pid: r + 1 for r, pid in enumerate(dense_ids)}
    sparse_rank = {pid: r + 1 for r, pid in enumerate(sparse_ids)}

    ranked = sorted(fused_scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

    hits: list[HybridHit] = []
    for point_id, score in ranked:
        payload = payload_by_id.get(point_id, {})
        hits.append(
            HybridHit(
                point_id=point_id,
                text=payload.get("text", ""),
                source=payload.get("source", ""),
                parent_id=payload.get("parent_id", ""),
                score=score,
                dense_rank=dense_rank.get(point_id),
                sparse_rank=sparse_rank.get(point_id),
            )
        )
    return hits


def dense_only_search(query: str, top_k: int | None = None) -> list[HybridHit]:
    """Pure dense search path, used by the "basic RAG" endpoint (item 1)."""
    top_k = top_k or settings.top_k_final
    dense_vec = embed_dense([query])[0]
    hits = search_dense(dense_vec, limit=top_k)
    return [
        HybridHit(
            point_id=str(h.id),
            text=h.payload.get("text", ""),
            source=h.payload.get("source", ""),
            parent_id=h.payload.get("parent_id", ""),
            score=h.score,
            dense_rank=i + 1,
        )
        for i, h in enumerate(hits)
    ]
