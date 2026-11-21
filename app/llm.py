"""Minimal LLM wrapper used for query rewriting, multi-query generation,
GraphRAG entity/relation extraction, adaptive routing decisions, and final
answer synthesis. Centralizing this makes it trivial to swap providers
(OpenAI-compatible endpoints, Azure OpenAI, local vLLM, etc.) via
OPENAI_BASE_URL in .env.
"""
from __future__ import annotations

from functools import lru_cache

from app.config import settings


@lru_cache(maxsize=1)
def _client():
    from openai import OpenAI

    if not settings.openai_api_key:
        raise RuntimeError(
            "No OPENAI_API_KEY configured. Set it in .env to enable LLM-powered "
            "features (query rewriting, multi-query, GraphRAG extraction, "
            "answer synthesis). Retrieval-only features work without it."
        )
    return OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)


def chat_completion(prompt: str, system: str | None = None, temperature: float = 0.0) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = _client().chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content or ""


def synthesize_answer(question: str, contexts: list[str]) -> str:
    context_block = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    prompt = (
        "Answer the question using ONLY the provided context. Cite sources "
        "with bracketed numbers like [1]. If the answer isn't in the context, "
        "say so.\n\n"
        f"Context:\n{context_block}\n\nQuestion: {question}\nAnswer:"
    )
    try:
        return chat_completion(prompt)
    except Exception as exc:  # no LLM configured -> return extractive fallback
        if not contexts:
            return "No relevant context was retrieved."
        return (
            "[LLM not configured — returning top retrieved context verbatim] "
            + contexts[0][:600]
        )
