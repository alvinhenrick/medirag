"""
Q&A runner: semantic cache + streaming DSPy RAG.

Single async path.
"""

from typing import AsyncGenerator

from loguru import logger

from medirag.cache.abc import SemanticCache
from medirag.rag.dspy import DspyRAG, stream_answer


class QuestionAnswerRunner:
    def __init__(self, sm: SemanticCache, rag: DspyRAG):
        self.semantic_cache = sm
        self.rag = rag

    async def ask(self, query: str, cosine_threshold: float = 0.9) -> AsyncGenerator[str, None]:
        """
        Stream an answer.

        Returns cached response in one chunk if found.
        """
        cached = self.semantic_cache.lookup(question=query, cosine_threshold=cosine_threshold)
        if cached:
            yield cached
            return

        accumulated = ""
        try:
            async for chunk in stream_answer(self.rag, query):
                accumulated += chunk
                yield chunk
        except Exception as e:
            logger.error(f"RAG error: {e}")
            if not accumulated:
                yield "An unexpected error occurred while answering your question."
            return

        if accumulated:
            self.semantic_cache.save(query, accumulated)
