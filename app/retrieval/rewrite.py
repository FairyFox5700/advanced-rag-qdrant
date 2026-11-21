"""Query rewriting and multi-query expansion. Checklist item 3.

Uses an LLM to (a) rewrite a possibly ambiguous/conversational query into a
clean standalone search query, and (b) generate several paraphrased
sub-queries that are each retrieved independently and merged, improving
recall for vague or multi-faceted questions.
"""
from __future__ import annotations

from app.config import settings
from app.llm import chat_completion

REWRITE_PROMPT = """You rewrite user questions into a single, clear, standalone \
search query optimized for retrieval in a vector database. Remove ambiguity, \
expand acronyms if obvious, and keep it concise. Return ONLY the rewritten query, \
nothing else.

Question: {question}
Rewritten query:"""

MULTI_QUERY_PROMPT = """Generate {n} different search queries that explore \
different phrasings or sub-aspects of the following question, to maximize \
recall when searching a vector database. Return one query per line, no \
numbering, no extra commentary.

Question: {question}"""


def rewrite_query(question: str) -> str:
    try:
        rewritten = chat_completion(REWRITE_PROMPT.format(question=question))
        return rewritten.strip().strip('"') or question
    except Exception:
        # Fail open: if no LLM is configured, just use the original question.
        return question


def generate_multi_queries(question: str, n: int = 3) -> list[str]:
    try:
        raw = chat_completion(MULTI_QUERY_PROMPT.format(question=question, n=n))
        queries = [line.strip("-* \t") for line in raw.splitlines() if line.strip()]
        queries = [q for q in queries if q]
        return (queries or [question])[:n]
    except Exception:
        return [question]
