"""Central configuration for the Advanced RAG stack.

All tunables live here so the rest of the codebase stays declarative.
Reads from environment variables (see .env.example) with sane defaults
that work against a local Qdrant instance started via docker-compose.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    # Qdrant
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key: str | None = os.getenv("QDRANT_API_KEY") or None
    collection_name: str = os.getenv("QDRANT_COLLECTION", "advanced_rag_docs")

    # Embeddings
    dense_model_name: str = os.getenv("DENSE_MODEL", "BAAI/bge-small-en-v1.5")
    sparse_model_name: str = os.getenv("SPARSE_MODEL", "Qdrant/bm25")
    dense_vector_size: int = int(os.getenv("DENSE_VECTOR_SIZE", "384"))

    # Chunking
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "400"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "60"))
    parent_chunk_size: int = int(os.getenv("PARENT_CHUNK_SIZE", "1600"))
    parent_chunk_overlap: int = int(os.getenv("PARENT_CHUNK_OVERLAP", "100"))

    # Retrieval
    top_k_dense: int = int(os.getenv("TOP_K_DENSE", "20"))
    top_k_sparse: int = int(os.getenv("TOP_K_SPARSE", "20"))
    top_k_final: int = int(os.getenv("TOP_K_FINAL", "5"))
    rrf_k: int = int(os.getenv("RRF_K", "60"))

    # Reranker
    cross_encoder_model: str = os.getenv(
        "CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    # LLM (used for query rewriting, multi-query, GraphRAG extraction, answer synthesis)
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY") or None
    openai_base_url: str | None = os.getenv("OPENAI_BASE_URL") or None

    data_dir: str = os.getenv("DATA_DIR", "data/sample_docs")


settings = Settings()
