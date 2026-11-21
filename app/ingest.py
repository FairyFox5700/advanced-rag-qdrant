"""Ingestion pipeline: load docs -> chunk (parent+child) -> embed -> upsert.

This single pipeline feeds every retrieval strategy in the project:
- child chunks get dense + sparse vectors for basic/hybrid search (items 1-2)
- parent chunks are stored (payload-only, no vectors) for parent-document
  expansion (item 5) and as the unit GraphRAG extracts entities from (item 8)
"""
from __future__ import annotations

import json
import os
import uuid

from qdrant_client import models
from tqdm import tqdm

from app.chunking import ChildChunk, ParentChunk, chunk_document, load_documents
from app.config import settings
from app.embeddings import embed_dense, embed_sparse
from app.qdrant_store import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME, ensure_collection, upsert_points

PARENT_STORE_PATH = os.path.join("data", "parent_store.json")


def _uid(key: str) -> str:
    """Deterministic UUID5 so re-ingesting the same content upserts in place."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def _save_parent_store(parents: dict[str, ParentChunk]) -> None:
    os.makedirs(os.path.dirname(PARENT_STORE_PATH), exist_ok=True)
    serializable = {
        pid: {"source": p.source, "text": p.text} for pid, p in parents.items()
    }
    with open(PARENT_STORE_PATH, "w", encoding="utf-8") as fh:
        json.dump(serializable, fh, ensure_ascii=False, indent=2)


def load_parent_store() -> dict[str, dict]:
    if not os.path.exists(PARENT_STORE_PATH):
        return {}
    with open(PARENT_STORE_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def ingest(directory: str | None = None, recreate: bool = False, batch_size: int = 64) -> dict:
    ensure_collection(recreate=recreate)

    documents = load_documents(directory)
    all_parents: dict[str, ParentChunk] = {}
    all_children: list[ChildChunk] = []

    for doc in documents:
        parents, children = chunk_document(doc)
        for p in parents:
            all_parents[p.parent_id] = p
        all_children.extend(children)

    _save_parent_store(all_parents)

    points: list[models.PointStruct] = []
    for batch_start in tqdm(range(0, len(all_children), batch_size), desc="embedding+indexing"):
        batch = all_children[batch_start : batch_start + batch_size]
        texts = [c.text for c in batch]
        dense_vecs = embed_dense(texts)
        sparse_vecs = embed_sparse(texts)

        for child, dense_vec, sparse_vec in zip(batch, dense_vecs, sparse_vecs):
            points.append(
                models.PointStruct(
                    id=_uid(child.child_id),
                    vector={
                        DENSE_VECTOR_NAME: dense_vec,
                        SPARSE_VECTOR_NAME: models.SparseVector(
                            indices=sparse_vec.indices.tolist(),
                            values=sparse_vec.values.tolist(),
                        ),
                    },
                    payload={
                        "child_id": child.child_id,
                        "parent_id": child.parent_id,
                        "source": child.source,
                        "text": child.text,
                    },
                )
            )
        upsert_points(points)
        points = []

    return {
        "documents_ingested": len(documents),
        "chunks_indexed": len(all_children),
        "parent_chunks_stored": len(all_parents),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", default=None)
    parser.add_argument("--recreate", action="store_true")
    args = parser.parse_args()

    result = ingest(directory=args.directory, recreate=args.recreate)
    print(json.dumps(result, indent=2))
