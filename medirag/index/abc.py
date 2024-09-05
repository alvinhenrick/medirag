from abc import ABC, abstractmethod

from llama_index.core.postprocessor import LLMRerank
from loguru import logger


class Indexer(ABC):
    @property
    @abstractmethod
    def vector_store_index(self):
        """
        This should return the vector store index used by the subclass.
        """
        pass

    @abstractmethod
    def retrieve(self, query: str | list[str], top_k: int, with_reranker: bool) -> list:
        """
        Retrieve top_k results based on the query or queries.
        """
        pass

    def retrieve_common(self, query, top_k=3, with_reranker=False):
        """
        Common retrieve functionality used across different indexers.
        """
        if not self.vector_store_index:
            logger.error("Index is not initialized. Please build or load an index first.")
            raise ValueError("Index is not initialized.")

        retriever = self.vector_store_index.as_retriever(similarity_top_k=(top_k * 3 if with_reranker else top_k))
        nodes = retriever.retrieve(query)
        logger.info(f"Retrieved {len(nodes)} nodes.")

        if with_reranker:
            ranker = LLMRerank(choice_batch_size=top_k, top_n=top_k)
            ranked_nodes = ranker.postprocess_nodes(nodes, query_str=query)
            logger.info(f"Reranked nodes to {len(ranked_nodes)}")
            return ranked_nodes
        else:
            return nodes
