import faiss
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
        self.vector_index = faiss.IndexFlatIP(self.dimension)
        self.encoder = SentenceTransformer(model_name)
        self._cache = SemanticCacheModel()  # Initialize with a default SemanticCache to avoid NoneType issues
        self.load_cache()

    def load_cache(self) -> None:
        try:
            with open(self.json_file, "r") as file:
                data = json.load(file)
            self._cache = SemanticCacheModel(**data)  # Use unpacking to handle Pydantic validation
            for emb in self._cache.embeddings:
                np_emb = np.array(emb, dtype=np.float32)
                faiss.normalize_L2(np_emb.reshape(1, -1))
                self.vector_index.add(np_emb.reshape(1, -1))
        except FileNotFoundError:
            logger.info("Cache file not found, initializing new cache.")
        except ValidationError as e:
            logger.error(f"Error in cache data structure: {e}")
        except Exception as e:
            logger.error(f"Failed to load or process cache: {e}")

    def save_cache(self):
        data = self._cache.dict()
        with open(self.json_file, "w") as file:
            json.dump(data, file, indent=4)
        logger.info("Cache saved successfully.")

    def lookup(self, question: str, cosine_threshold: float = 0.7) -> str | None:
        embedding = self.encoder.encode([question], show_progress_bar=False)
        faiss.normalize_L2(embedding)
        data, index = self.vector_index.search(embedding, 1)
        if data[0][0] >= cosine_threshold:
            return self._cache.response_text[index[0][0]]
        return None

    def save(self, question: str, response: str):
        """
        Save a response to the cache.
        """
        embedding = self.encoder.encode([question], show_progress_bar=False)
        faiss.normalize_L2(embedding)
        self._cache.questions.append(question)
        self._cache.embeddings.append(embedding[0].tolist())
        self._cache.response_text.append(response)
        self.vector_index.add(embedding)  # noqa
        self.save_cache()
        logger.info("New response saved to cache.")

    def clear(self):
        self._cache = SemanticCacheModel()
        self.vector_index.reset()
        self.save_cache()
        logger.info("Cache cleared.")
