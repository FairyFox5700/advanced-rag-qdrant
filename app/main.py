"""FastAPI application exposing every retrieval strategy built in this repo.

Endpoints:
  POST /ingest                -> run the ingestion pipeline (item 1, 5)
  POST /query                 -> configurable query (items 1-5)
  POST /query/adaptive        -> LangGraph adaptive routing (item 7)
  POST /query/graphrag        -> direct GraphRAG traversal query (item 8)
  POST /graph/build           -> (re)build the knowledge graph
  GET  /health                -> liveness probe
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException

from app.config import settings
from app.graphrag.extract import build_graph, save_graph
from app.graphrag.retrieval import graph_retrieve
from app.ingest import ingest
from app.llm import synthesize_answer
from app.models import (
    AdaptiveQueryResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    RetrievedChunk,
)
from app.retrieval.adaptive_graph import run_adaptive_query
from app.retrieval.hybrid import dense_only_search, hybrid_search
from app.retrieval.parent_document import expand_to_parents
from app.retrieval.reranker import rerank
from app.retrieval.rewrite import generate_multi_queries, rewrite_query

app = FastAPI(
    title="Advanced RAG over Qdrant",
    description="Reference implementation of the Advanced RAG checklist: "
    "basic RAG, hybrid search + RRF, query rewriting/multi-query, "
    "cross-encoder reranking, parent-document retrieval, RAGAS evaluation, "
    "LangGraph adaptive retrieval, and GraphRAG.",
    version="1.0.0",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
def ingest_endpoint(req: IngestRequest) -> IngestResponse:
    result = ingest(directory=req.directory, recreate=True)
    return IngestResponse(**result)


@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest) -> QueryResponse:
    question = req.question
    rewritten_queries: list[str] = []

    if req.multi_query:
        base_query = rewrite_query(question) if req.use_query_rewrite else question
        sub_queries = generate_multi_queries(base_query, n=3)
        rewritten_queries = [base_query, *sub_queries]
        merged = {}
        for q in rewritten_queries:
            search_fn = hybrid_search if req.use_hybrid else dense_only_search
            for hit in search_fn(q, top_k=settings.top_k_dense):
                if hit.point_id not in merged or hit.score > merged[hit.point_id].score:
                    merged[hit.point_id] = hit
        hits = sorted(merged.values(), key=lambda h: h.score, reverse=True)
    else:
        search_query = rewrite_query(question) if req.use_query_rewrite else question
        if req.use_query_rewrite:
            rewritten_queries = [search_query]
        search_fn = hybrid_search if req.use_hybrid else dense_only_search
        hits = search_fn(search_query, top_k=settings.top_k_dense)

    if req.use_reranker:
        hits = rerank(question, hits, top_k=req.top_k)
    else:
        hits = hits[: req.top_k]

    if req.use_parent_expansion:
        hits = expand_to_parents(hits)

    answer = synthesize_answer(question, [h.text for h in hits])

    retrieved = [
        RetrievedChunk(
            id=h.point_id,
            text=h.text,
            score=h.score,
            source=h.source,
            rank_method="hybrid" if req.use_hybrid else "dense",
        )
        for h in hits
    ]
    return QueryResponse(
        question=question, rewritten_queries=rewritten_queries, retrieved=retrieved, answer=answer
    )


@app.post("/query/adaptive", response_model=AdaptiveQueryResponse)
def adaptive_query_endpoint(req: QueryRequest) -> AdaptiveQueryResponse:
    state = run_adaptive_query(req.question)
    retrieved = [
        RetrievedChunk(id=h.point_id, text=h.text, score=h.score, source=h.source, rank_method=state["route"])
        for h in state.get("hits", [])
    ]
    return AdaptiveQueryResponse(
        question=req.question,
        rewritten_queries=state.get("rewritten_queries", []),
        retrieved=retrieved,
        answer=state.get("answer", ""),
        route=state.get("route", "simple"),
        trace=state.get("trace", []),
    )


@app.post("/graph/build")
def build_graph_endpoint(use_llm: bool = True) -> dict:
    graph = build_graph(use_llm=use_llm)
    save_graph(graph)
    return {"nodes": graph.number_of_nodes(), "edges": graph.number_of_edges()}


@app.post("/query/graphrag", response_model=QueryResponse)
def graphrag_query_endpoint(req: QueryRequest) -> QueryResponse:
    results = graph_retrieve(req.question, hops=1, max_chunks=req.top_k)
    if not results:
        raise HTTPException(
            status_code=404,
            detail="No graph matches found. Build the graph via /graph/build and "
            "ensure the question mentions entities present in the documents.",
        )
    answer = synthesize_answer(req.question, [r["text"] for r in results])
    retrieved = [
        RetrievedChunk(
            id=r["chunk_id"], text=r["text"], score=float(len(r["entities"])), source=r["source"], rank_method="graph"
        )
        for r in results
    ]
    return QueryResponse(question=req.question, retrieved=retrieved, answer=answer)
