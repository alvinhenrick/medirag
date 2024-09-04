from abc import ABC, abstractmethod


class SemanticCache(ABC):
    """
    Abstract base class for semantic caching mechanisms.
    """

    @abstractmethod
    def lookup(self, question: str, cosine_threshold: float):
        """
        Retrieve a response from the cache based on the question and cosine similarity threshold.
        """
        pass

    @abstractmethod
    def save(self, question: str, answer: str):
        """
        Save a question-answer pair to the cache.
        """
        pass
