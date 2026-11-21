"""Graph-based retrieval over the knowledge graph built in extract.py."""
from __future__ import annotations

import re

import networkx as nx

from app.graphrag.extract import load_graph
from app.ingest import load_parent_store


def _find_seed_entities(question: str, graph: nx.MultiDiGraph) -> list[str]:
    """Match capitalized phrases in the question against graph node names
    (case-insensitive substring match), falling back to fuzzy containment."""
    candidates = re.findall(r"\b[A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*\b", question)
    nodes_lower = {n.lower(): n for n in graph.nodes}

    seeds = []
    for cand in candidates:
        if cand.lower() in nodes_lower:
            seeds.append(nodes_lower[cand.lower()])

    if not seeds:
        # fall back to substring match over all node names
        q_lower = question.lower()
        seeds = [n for n in graph.nodes if n.lower() in q_lower]

    return list(dict.fromkeys(seeds))  # dedupe, preserve order


def graph_retrieve(question: str, hops: int = 1, max_chunks: int = 5) -> list[dict]:
    """Return a list of {chunk_id, source, text, entities} dicts gathered by
    walking `hops` steps out from entities mentioned in the question."""
    graph = load_graph()
    if graph.number_of_nodes() == 0:
        return []

    seeds = _find_seed_entities(question, graph)
    if not seeds:
        return []

    visited_nodes: set[str] = set(seeds)
    frontier = set(seeds)
    for _ in range(hops):
        next_frontier = set()
        for node in frontier:
            next_frontier |= set(graph.successors(node)) | set(graph.predecessors(node))
        visited_nodes |= next_frontier
        frontier = next_frontier

    chunk_ids: dict[str, set[str]] = {}
    for node in visited_nodes:
        for chunk_id in graph.nodes[node].get("chunks", []):
            chunk_ids.setdefault(chunk_id, set()).add(node)

    parent_store = load_parent_store()
    results = []
    for chunk_id, entities in sorted(chunk_ids.items(), key=lambda kv: -len(kv[1]))[:max_chunks]:
        parent = parent_store.get(chunk_id)
        if not parent:
            continue
        results.append(
            {
                "chunk_id": chunk_id,
                "source": parent["source"],
                "text": parent["text"],
                "entities": sorted(entities),
            }
        )
    return results
