from medirag.cache.abc import SemanticCache
from medirag.rag.dspy import DspyRAG
from medirag.rag.llama_index import WorkflowRAG

from loguru import logger
from typing import AsyncGenerator


class QuestionAnswerRunner:
    def __init__(self, sm: SemanticCache, rag: DspyRAG | WorkflowRAG):
        self.semantic_cache = sm
        self.rag = rag

    async def ask(
        self, query: str, cosine_threshold: float = 0.9, enable_stream: bool = False
    ) -> AsyncGenerator[str, None]:
        """
        Asynchronously ask a medical question, yielding the response directly if available or from a streaming source.
        """
        cached_response = self.semantic_cache.lookup(question=query, cosine_threshold=cosine_threshold)
        if cached_response:
            yield cached_response
        else:
            if enable_stream:
                async for response in self._handle_streaming_query(query):
                    yield response
            else:
                response = self._handle_non_streaming_query(query)
                yield response
                self.semantic_cache.save(query, response)

    async def _handle_streaming_query(self, query: str) -> AsyncGenerator[str, None]:
        try:
            response = await self.rag.run(query=query)
            if isinstance(response, str):
                yield response
                self.semantic_cache.save(query, response)
            if hasattr(response, "async_response_gen"):
                accumulated_response = ""
                async for chunk in response.async_response_gen():
                    accumulated_response += chunk
                    yield chunk
                self.semantic_cache.save(query, accumulated_response)
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            yield "An unexpected error occurred."

    def _handle_non_streaming_query(self, query: str) -> str:
        try:
            response = self.rag(query).answer
            return response
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return "Error processing request."
