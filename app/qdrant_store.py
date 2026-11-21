"""Thin wrapper around the Qdrant client: collection bootstrap, upsert, and
raw dense/sparse search primitives. Hybrid fusion (RRF) lives in
app/retrieval/hybrid.py and composes on top of these primitives.
"""
from __future__ import annotations

from qdrant_client import QdrantClient, models

from app.config import settings

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    return _client


def ensure_collection(recreate: bool = False) -> None:
    client = get_client()
    exists = client.collection_exists(settings.collection_name)
    if exists and not recreate:
        return
    if exists and recreate:
        client.delete_collection(settings.collection_name)

    client.create_collection(
        collection_name=settings.collection_name,
        vectors_config={
            DENSE_VECTOR_NAME: models.VectorParams(
                size=settings.dense_vector_size, distance=models.Distance.COSINE
            )
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=False)
            )
        },
    )


def upsert_points(points: list[models.PointStruct]) -> None:
    client = get_client()
    client.upsert(collection_name=settings.collection_name, points=points, wait=True)


def search_dense(vector: list[float], limit: int) -> list[models.ScoredPoint]:
    client = get_client()
    return client.query_points(
        collection_name=settings.collection_name,
        using=DENSE_VECTOR_NAME,
        query=vector,
        limit=limit,
        with_payload=True,
    ).points


def search_sparse(indices: list[int], values: list[float], limit: int) -> list[models.ScoredPoint]:
    client = get_client()
    return client.query_points(
        collection_name=settings.collection_name,
        using=SPARSE_VECTOR_NAME,
        query=models.SparseVector(indices=indices, values=values),
        limit=limit,
        with_payload=True,
    ).points


def fetch_by_ids(point_ids: list[str]) -> list[models.Record]:
    client = get_client()
    return client.retrieve(collection_name=settings.collection_name, ids=point_ids, with_payload=True)
