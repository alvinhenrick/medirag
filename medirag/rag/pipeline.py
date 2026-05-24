"""
Cache-layered streaming pipeline: cache lookup → DSPy RAG stream → cache save.

Composes the RAG module with the semantic cache. Owns the caching policy
(when to look up, when to save) but knows nothing about LLMs or DSPy internals.
"""

from typing import AsyncIterator

from loguru import logger

from medirag.cache.abc import SemanticCache
from medirag.rag.dspy import DspyRAG, stream_answer


async def answer_stream(
    rag: DspyRAG,
    cache: SemanticCache,
    query: str,
    cosine_threshold: float = 0.9,
) -> AsyncIterator[str]:
    """
    Stream an answer, serving from the semantic cache when possible.

    Cache hits are yielded as one chunk. Misses stream through the RAG and the accumulated answer is written back to the
    cache on success.
    """
    cached = cache.lookup(question=query, cosine_threshold=cosine_threshold)
    if cached:
        yield cached
        return

    accumulated = ""
    try:
        async for chunk in stream_answer(rag, query):
            accumulated += chunk
            yield chunk
    except Exception as e:
        logger.error(f"RAG error: {e}")
        if not accumulated:
            yield "An unexpected error occurred while answering your question."
        return

    if accumulated:
        cache.save(query, accumulated)
