"""Lightweight smoke tests that don't require a running Qdrant instance or an
OpenAI API key — they exercise the parts of the pipeline that are pure
Python logic (chunking, RRF math, heuristic routing/extraction fallbacks).
Run with: pytest -q  (or `python tests/test_smoke.py`)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.chunking import Document, chunk_document
from app.rrf import reciprocal_rank_fusion


def test_chunking_produces_parents_and_children():
    doc = Document(doc_id="d1", source="test.txt", text=("Sentence. " * 500))
    parents, children = chunk_document(doc)
    assert len(parents) >= 1
    assert len(children) >= len(parents)
    assert all(c.parent_id.startswith(p.parent_id) for p in parents for c in children if c.parent_id == p.parent_id)


def test_rrf_prefers_docs_ranked_well_in_both_lists():
    dense = ["a", "b", "c", "d"]
    sparse = ["b", "a", "d", "c"]
    scores = reciprocal_rank_fusion(dense, sparse, k=60)
    ranked = sorted(scores, key=scores.get, reverse=True)
    # "a" and "b" are top-2 in both lists, so they must outrank "c"/"d"
    assert set(ranked[:2]) == {"a", "b"}


def test_heuristic_route_detects_graph_questions():
    from app.heuristics import heuristic_route

    assert heuristic_route("How are X and Y connected?") == "graph"
    assert heuristic_route("What is the capital of France?") == "simple"
    assert heuristic_route("Compare A and B and list all differences") in {"multi_hop", "graph"}


def test_heuristic_triples_extracts_capitalized_entities():
    from app.heuristics import heuristic_triples

    triples = heuristic_triples("Elena Vasquez founded Northwind Robotics with Marcus Chen.")
    assert len(triples) > 0
    assert all("subject" in t and "object" in t for t in triples)


if __name__ == "__main__":
    test_chunking_produces_parents_and_children()
    test_rrf_prefers_docs_ranked_well_in_both_lists()
    test_heuristic_route_detects_graph_questions()
    test_heuristic_triples_extracts_capitalized_entities()
    print("All smoke tests passed.")
