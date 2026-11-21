"""Pure-Python heuristic fallbacks, split out so they can be unit-tested
without pulling in networkx/langgraph/openai/fastembed/qdrant-client.
Used as offline fallbacks when no OPENAI_API_KEY is configured.
"""
from __future__ import annotations

import re
from typing import Literal

Route = Literal["simple", "multi_hop", "graph"]

GRAPH_KEYWORDS = ("relationship", "connect", "between", "related to", "compare", "versus", " vs ")
MULTI_HOP_KEYWORDS = ("and", "difference", "compare", "both", "list all", "pros and cons")


def heuristic_route(question: str) -> Route:
    q = question.lower()
    if any(k in q for k in GRAPH_KEYWORDS):
        return "graph"
    if any(k in q for k in MULTI_HOP_KEYWORDS):
        return "multi_hop"
    return "simple"


def heuristic_triples(text: str) -> list[dict]:
    """No-LLM fallback for GraphRAG extraction: co-occurrence of capitalized
    phrases within a chunk becomes a generic "related_to" edge. Weaker than
    LLM extraction but keeps the graph pipeline runnable offline."""
    capitalized = re.findall(r"\b[A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*\b", text)
    entities = sorted(set(capitalized))[:12]
    triples = []
    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            triples.append({"subject": entities[i], "relation": "related_to", "object": entities[j]})
    return triples[:20]
