"""Parent-document retrieval. Checklist item 5.

We embed small child chunks (good for precise semantic matching) but, once
the best children are identified, we expand each hit to its larger parent
chunk before handing context to the LLM — giving the generator more
surrounding context without diluting retrieval precision.
"""
from __future__ import annotations

from app.ingest import load_parent_store
from app.retrieval.hybrid import HybridHit


def expand_to_parents(hits: list[HybridHit]) -> list[HybridHit]:
    parent_store = load_parent_store()
    seen_parents: set[str] = set()
    expanded: list[HybridHit] = []

    for hit in hits:
        parent = parent_store.get(hit.parent_id)
        if parent is None:
            # Fall back to the child chunk itself if the parent store is stale.
            expanded.append(hit)
            continue
        if hit.parent_id in seen_parents:
            continue
        seen_parents.add(hit.parent_id)
        expanded.append(
            HybridHit(
                point_id=hit.parent_id,
                text=parent["text"],
                source=parent["source"],
                parent_id=hit.parent_id,
                score=hit.score,
                dense_rank=hit.dense_rank,
                sparse_rank=hit.sparse_rank,
            )
        )
    return expanded
