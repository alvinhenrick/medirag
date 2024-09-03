import faiss
import json
import numpy as np
from pydantic import BaseModel, ValidationError
from sentence_transformers import SentenceTransformer
from loguru import logger


class SemanticCache(BaseModel):
    questions: list[str] = []
    embeddings: list[list[float]] = []
    response_text: list[str] = []


class SemanticCaching:
    def __init__(self, model_name: str = 'sentence-transformers/all-mpnet-base-v2',
                 dimension: int = 768,
                 json_file: str = 'cache.json'):
        self._cache = None
        self.model_name = model_name
        self.dimension = dimension
        self.json_file = json_file
        self.vector_index = faiss.IndexFlatIP(self.dimension)
        self.encoder = SentenceTransformer(model_name)
        self.load_cache()  # Automatically attempt to load the cache upon initialization

    def load_cache(self) -> None:
        """Load cache from a JSON file."""
        try:
            with open(self.json_file, 'r') as file:
                data = json.load(file)
            # Create a SemanticCache instance from the data
            self._cache = SemanticCache.model_validate(data)
            # Convert embeddings to numpy arrays and add to FAISS
            for emb in self._cache.embeddings:
                np_emb = np.array(emb, dtype=np.float32)
                faiss.normalize_L2(np_emb.reshape(1, -1))  # Normalize before adding to FAISS
                self.vector_index.add(np_emb.reshape(1, -1))  # Reshape for FAISS
        except FileNotFoundError:
            logger.info("Cache file not found, initializing new cache.")
            self._cache = SemanticCache()
        except ValidationError as e:
            logger.error(f"Error in cache data structure: {e}")
            self._cache = SemanticCache()
        except Exception as e:
            logger.error(f"Failed to load or process cache: {e}")
            self._cache = SemanticCache()

    def save_cache(self):
        """Save the current cache to a JSON file."""
        data = self._cache.dict()
        with open(self.json_file, 'w') as file:
            json.dump(data, file, indent=4)  # Add indentation for better readability
        logger.info("Cache saved successfully.")

    def lookup(self, question: str, cosine_threshold: float = 0.7) -> str | None:
        """Check if a question is in the cache and return the cached response if it exists."""
        embedding = self.encoder.encode([question], show_progress_bar=False)
        faiss.normalize_L2(embedding)
        D, I = self.vector_index.search(embedding, 1)

        if D[0][0] >= cosine_threshold:
            row_id = I[0][0]
            return self._cache.response_text[row_id]
        return None

    def save(self, question: str, response: str):
        """Save a response to the cache."""
        embedding = self.encoder.encode([question], show_progress_bar=False)
        faiss.normalize_L2(embedding)
        self._cache.questions.append(question)
        self._cache.embeddings.append(embedding[0].tolist())  # Ensure embedding is flattened
        self._cache.response_text.append(response)
        self.vector_index.add(embedding)
        self.save_cache()
        logger.info("New response saved to cache.")

    def clear(self):
        """Clear the cache."""
        self._cache = SemanticCache()
        self.vector_index.reset()
        self.save_cache()
        logger.info("Cache cleared.")
