"""Adaptive retrieval via LangGraph. Checklist item 7.

A small state machine:
  classify -> {simple | multi_hop | graph} -> retrieve -> rerank -> answer

The classifier (LLM-based with a heuristic fallback) looks at the question
and decides which retrieval strategy is likely to work best:
  - "simple": a single-fact lookup -> hybrid search + rerank
  - "multi_hop": a comparison/aggregation question -> query rewrite +
     multi-query expansion, results merged then reranked
  - "graph": a relational "how are X and Y connected" question -> GraphRAG
     traversal, falling back to hybrid search if the graph has no match
"""
from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph

from app.config import settings
from app.graphrag.retrieval import graph_retrieve
from app.heuristics import Route, heuristic_route
from app.llm import synthesize_answer
from app.retrieval.hybrid import HybridHit, hybrid_search
from app.retrieval.parent_document import expand_to_parents
from app.retrieval.reranker import rerank
from app.retrieval.rewrite import generate_multi_queries, rewrite_query

CLASSIFY_PROMPT = """Classify the question into exactly one label:
- simple: a direct single-fact lookup
- multi_hop: requires comparing, combining, or reasoning over multiple pieces of information
- graph: asks how two or more named entities/concepts relate or connect to each other

Return only the label.

Question: {question}
Label:"""


def classify_route(question: str) -> Route:
    try:
        from app.llm import chat_completion

        label = chat_completion(CLASSIFY_PROMPT.format(question=question)).strip().lower()
        if label in ("simple", "multi_hop", "graph"):
            return label  # type: ignore[return-value]
    except Exception:
        pass
    return heuristic_route(question)


class RAGState(TypedDict, total=False):
    question: str
    route: Route
    rewritten_queries: list[str]
    hits: list[HybridHit]
    trace: list[str]
    answer: str


def node_classify(state: RAGState) -> RAGState:
    route = classify_route(state["question"])
    return {"route": route, "trace": state.get("trace", []) + [f"classified as '{route}'"]}


def node_simple_retrieve(state: RAGState) -> RAGState:
    hits = hybrid_search(state["question"], top_k=settings.top_k_dense)
    return {"hits": hits, "trace": state["trace"] + ["hybrid_search (simple)"]}


def node_multi_hop_retrieve(state: RAGState) -> RAGState:
    question = state["question"]
    rewritten = rewrite_query(question)
    sub_queries = generate_multi_queries(rewritten, n=3)

    merged: dict[str, HybridHit] = {}
    for q in [rewritten, *sub_queries]:
        for hit in hybrid_search(q, top_k=settings.top_k_dense):
            if hit.point_id not in merged or hit.score > merged[hit.point_id].score:
                merged[hit.point_id] = hit

    hits = sorted(merged.values(), key=lambda h: h.score, reverse=True)
    return {
        "hits": hits,
        "rewritten_queries": [rewritten, *sub_queries],
        "trace": state["trace"] + [f"multi_query x{len(sub_queries) + 1}"],
    }


def node_graph_retrieve(state: RAGState) -> RAGState:
    graph_hits = graph_retrieve(state["question"], hops=1, max_chunks=settings.top_k_final)
    if not graph_hits:
        # Graph had nothing relevant -> fall back to hybrid search.
        hits = hybrid_search(state["question"], top_k=settings.top_k_dense)
        trace_msg = "graph_retrieve empty -> fallback to hybrid_search"
    else:
        hits = [
            HybridHit(
                point_id=g["chunk_id"],
                text=g["text"],
                source=g["source"],
                parent_id=g["chunk_id"],
                score=float(len(g["entities"])),
            )
            for g in graph_hits
        ]
        trace_msg = f"graph_retrieve matched {len(graph_hits)} chunks via entities {sorted({e for g in graph_hits for e in g['entities']})}"
    return {"hits": hits, "trace": state["trace"] + [trace_msg]}


def node_rerank_and_expand(state: RAGState) -> RAGState:
    hits = rerank(state["question"], state["hits"], top_k=settings.top_k_final)
    hits = expand_to_parents(hits)
    return {"hits": hits, "trace": state["trace"] + ["reranked + expanded to parents"]}


def node_answer(state: RAGState) -> RAGState:
    contexts = [h.text for h in state["hits"]]
    answer = synthesize_answer(state["question"], contexts)
    return {"answer": answer, "trace": state["trace"] + ["answer synthesized"]}


def route_selector(state: RAGState) -> str:
    return state["route"]


def build_adaptive_graph():
    graph = StateGraph(RAGState)
    graph.add_node("classify", node_classify)
    graph.add_node("simple_retrieve", node_simple_retrieve)
    graph.add_node("multi_hop_retrieve", node_multi_hop_retrieve)
    graph.add_node("graph_retrieve", node_graph_retrieve)
    graph.add_node("rerank_and_expand", node_rerank_and_expand)
    graph.add_node("answer", node_answer)

    graph.set_entry_point("classify")
    graph.add_conditional_edges(
        "classify",
        route_selector,
        {
            "simple": "simple_retrieve",
            "multi_hop": "multi_hop_retrieve",
            "graph": "graph_retrieve",
        },
    )
    graph.add_edge("simple_retrieve", "rerank_and_expand")
    graph.add_edge("multi_hop_retrieve", "rerank_and_expand")
    graph.add_edge("graph_retrieve", "rerank_and_expand")
    graph.add_edge("rerank_and_expand", "answer")
    graph.add_edge("answer", END)

    return graph.compile()


_compiled_graph = None


def run_adaptive_query(question: str) -> RAGState:
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_adaptive_graph()
    return _compiled_graph.invoke({"question": question, "trace": []})
