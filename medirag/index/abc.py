from abc import ABC, abstractmethod


class Indexer(ABC):
    @abstractmethod
    def retrieve(self, query: str | list[str], top_k: int) -> list:
        """
        Retrieve top_k results based on the query or queries.
        """
        pass