"""Embedding helpers. Uses `fastembed` for both dense (BGE) and sparse (BM25)
vectors so we don't need a GPU or external API for the base pipeline.
"""
from __future__ import annotations

from functools import lru_cache

from fastembed import SparseTextEmbedding, TextEmbedding

from app.config import settings


@lru_cache(maxsize=1)
def get_dense_model() -> TextEmbedding:
    return TextEmbedding(model_name=settings.dense_model_name)


@lru_cache(maxsize=1)
def get_sparse_model() -> SparseTextEmbedding:
    return SparseTextEmbedding(model_name=settings.sparse_model_name)


def embed_dense(texts: list[str]) -> list[list[float]]:
    model = get_dense_model()
    return [vec.tolist() for vec in model.embed(texts)]


def embed_sparse(texts: list[str]):
    """Returns fastembed SparseEmbedding objects with .indices/.values, which
    map directly onto Qdrant's sparse vector format."""
    model = get_sparse_model()
    return list(model.embed(texts))
