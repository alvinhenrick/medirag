"""
Semantic cache backed by a normalized numpy matrix + cosine similarity.

Small enough that brute-force search beats maintaining an ANN index, and avoids a faiss-cpu BLAS conflict with
pyarrow/lancedb on macOS.
"""

from pathlib import Path

import json
import numpy as np
from pydantic import BaseModel, ValidationError
from sentence_transformers import SentenceTransformer
from loguru import logger

from medirag.cache.abc import SemanticCache


class SemanticCacheModel(BaseModel):
    questions: list[str] = []
    embeddings: list[list[float]] = []
    response_text: list[str] = []


class LocalSemanticCache(SemanticCache):
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        dimension: int = 768,
        json_file: str = "cache.json",
    ):
        self.model_name = model_name
        self.dimension = dimension
        self.json_file = json_file
        self.encoder = SentenceTransformer(model_name)
        self._cache = SemanticCacheModel()
        self._matrix: np.ndarray = np.zeros((0, dimension), dtype=np.float32)
        self.load_cache()

    def load_cache(self) -> None:
        try:
            with open(self.json_file, "r") as file:
                data = json.load(file)
            self._cache = SemanticCacheModel(**data)
            if self._cache.embeddings:
                self._matrix = np.asarray(self._cache.embeddings, dtype=np.float32)
        except FileNotFoundError:
            logger.info("Cache file not found, initializing new cache.")
        except ValidationError as e:
            logger.error(f"Error in cache data structure: {e}")
        except Exception as e:
            logger.error(f"Failed to load or process cache: {e}")

    def save_cache(self):
        data = self._cache.model_dump()
        with open(self.json_file, "w") as file:
            json.dump(data, file, indent=4)
        logger.info("Cache saved successfully.")

    def _encode(self, text: str) -> np.ndarray:
        vec = self.encoder.encode([text], show_progress_bar=False, normalize_embeddings=True)
        return np.asarray(vec, dtype=np.float32).reshape(1, -1)

    def lookup(self, question: str, cosine_threshold: float = 0.7) -> str | None:
        if self._matrix.shape[0] == 0:
            return None
        q = self._encode(question)
        sims = (self._matrix @ q.T).ravel()
        best = int(np.argmax(sims))
        if float(sims[best]) >= cosine_threshold:
            return self._cache.response_text[best]
        return None

    def save(self, question: str, response: str):
        q = self._encode(question)
        self._cache.questions.append(question)
        self._cache.embeddings.append(q.ravel().tolist())
        self._cache.response_text.append(response)
        self._matrix = np.vstack([self._matrix, q]) if self._matrix.size else q
        self.save_cache()
        logger.info("New response saved to cache.")

    def clear(self):
        self._cache = SemanticCacheModel()
        self._matrix = np.zeros((0, self.dimension), dtype=np.float32)
        cache_file_path = Path(self.json_file)
        try:
            cache_file_path.unlink(missing_ok=True)
            logger.info(f"Cache file {self.json_file} deleted successfully.")
        except Exception as e:
            logger.error(f"Failed to delete cache file {self.json_file}: {e}")
        logger.info("Cache cleared.")
