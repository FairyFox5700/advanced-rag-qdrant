"""GraphRAG: lightweight entity/relation extraction + graph-based retrieval.
Checklist item 8.

For each parent chunk we ask the LLM to extract (entity, relation, entity)
triples. These are accumulated into a single NetworkX knowledge graph where
nodes are entities (with the chunk ids they appear in) and edges are
relations. At query time we extract entities from the question, pull their
local neighborhood out of the graph, and use the chunks attached to that
neighborhood as additional/alternative context to pure vector retrieval.
"""
from __future__ import annotations

import json
import os
import re

import networkx as nx

from app.config import settings
from app.heuristics import heuristic_triples
from app.ingest import load_parent_store
from app.llm import chat_completion

GRAPH_STORE_PATH = os.path.join("data", "knowledge_graph.json")

EXTRACTION_PROMPT = """Extract factual (subject, relation, object) triples from \
the text below. Keep entities short (proper nouns or key concepts). Return \
JSON only, a list of objects like {{"subject": "...", "relation": "...", "object": "..."}}. \
If nothing notable is found, return [].

Text:
{text}

JSON triples:"""


def _extract_triples(text: str) -> list[dict]:
    try:
        raw = chat_completion(EXTRACTION_PROMPT.format(text=text[:3000]))
    except Exception:
        return heuristic_triples(text)

    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return []
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return []



def build_graph(use_llm: bool = True) -> nx.MultiDiGraph:
    parents = load_parent_store()
    graph = nx.MultiDiGraph()

    for parent_id, parent in parents.items():
        triples = _extract_triples(parent["text"]) if use_llm else heuristic_triples(parent["text"])
        for triple in triples:
            subj = str(triple.get("subject", "")).strip()
            obj = str(triple.get("object", "")).strip()
            rel = str(triple.get("relation", "related_to")).strip() or "related_to"
            if not subj or not obj:
                continue
            graph.add_node(subj, chunks=set())
            graph.add_node(obj, chunks=set())
            graph.nodes[subj]["chunks"] = graph.nodes[subj].get("chunks", set()) | {parent_id}
            graph.nodes[obj]["chunks"] = graph.nodes[obj].get("chunks", set()) | {parent_id}
            graph.add_edge(subj, obj, relation=rel, chunk_id=parent_id)

    return graph


def save_graph(graph: nx.MultiDiGraph, path: str = GRAPH_STORE_PATH) -> None:
    data = nx.node_link_data(graph, edges="links")
    # sets aren't JSON serializable
    for node in data["nodes"]:
        if isinstance(node.get("chunks"), set):
            node["chunks"] = sorted(node["chunks"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def load_graph(path: str = GRAPH_STORE_PATH) -> nx.MultiDiGraph:
    if not os.path.exists(path):
        return nx.MultiDiGraph()
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return nx.node_link_graph(data, edges="links", multigraph=True, directed=True)


if __name__ == "__main__":
    g = build_graph()
    save_graph(g)
    print(f"Graph built: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges -> {GRAPH_STORE_PATH}")
