from typing import Protocol


# Define a protocol for indexers
class Indexer(Protocol):
    @property
    def vector_store_index(self):
        """
        This should return the vector store index used by the class.
        """
        ...

    def retrieve(self, query: str | list[str], top_k: int, with_reranker: bool) -> list:
        """
        Retrieve top_k results based on the query or queries.
        """
        ...
