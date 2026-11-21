"""Pydantic request/response models shared across the API."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    directory: str | None = Field(
        default=None, description="Directory of .txt/.md files to ingest. Defaults to settings.data_dir."
    )


class IngestResponse(BaseModel):
    documents_ingested: int
    chunks_indexed: int
    parent_chunks_stored: int


class RetrievedChunk(BaseModel):
    id: str
    text: str
    score: float
    source: str
    rank_method: str = "dense"


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    use_hybrid: bool = True
    use_reranker: bool = True
    use_parent_expansion: bool = True
    use_query_rewrite: bool = False
    multi_query: bool = False


class QueryResponse(BaseModel):
    question: str
    rewritten_queries: list[str] = []
    retrieved: list[RetrievedChunk]
    answer: str


class AdaptiveQueryResponse(QueryResponse):
    route: Literal["simple", "multi_hop", "graph"] = "simple"
    trace: list[str] = []
