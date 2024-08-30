import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticCaching:
    def __init__(self,
                 model_name='sentence-transformers/all-mpnet-base-v2',
                 dimension=768,
                 json_file='cache.json',
                 cosine_threshold=0.7):

        self.model_name = model_name
        self.dimension = dimension
        self.cosine_threshold = cosine_threshold
        self.vector_index = faiss.IndexFlatIP(self.dimension)
        self.encoder = SentenceTransformer(self.model_name)
        self.json_file = json_file
        self.cache = self.load_cache()

    def load_cache(self):
        """Load cache from a JSON file."""
        local_cache = {'questions': [], 'embeddings': [], 'response_text': []}
        try:
            if self.json_file:
                with open(self.json_file, 'r') as file:
                    local_cache = json.load(file)
                    if 'embeddings' in local_cache and len(local_cache['embeddings']) > 0:
                        for embedding in local_cache['embeddings']:
                            _embedding = np.array(embedding, dtype=np.float32)
                            self.vector_index.add(_embedding)
                return local_cache
            else:
                return local_cache
        except FileNotFoundError:
            return local_cache
        except Exception as e:
            print(f"Failed to load or process cache: {e}")
            return local_cache

    def save_cache(self):
        """Save the current cache to a JSON file."""
        with open(self.json_file, 'w') as file:
            json.dump(self.cache, file)

    def lookup(self, question: str) -> str | None:
        """Check if a question is in the cache and return the cached response if it exists."""
        embedding = self.encoder.encode([question], show_progress_bar=False)
        faiss.normalize_L2(embedding)

        # Search in the index
        D, I = self.vector_index.search(embedding, 1)

        if D[0][0] >= self.cosine_threshold:
            row_id = I[0][0]
            return self.cache['response_text'][row_id]
        else:
            return None

    def save(self, question: str, response: str):
        """Save a response to the cache."""
        embedding = self.encoder.encode([question], show_progress_bar=False)
        faiss.normalize_L2(embedding)

        self.cache['questions'].append(question)
        self.cache['embeddings'].append(embedding.tolist())
        self.cache['response_text'].append(response)
        self.vector_index.add(embedding)
        self.save_cache()

    def clear(self):
        self.cache = {'questions': [], 'embeddings': [], 'response_text': []}
        self.vector_index.reset()
        self.save_cache()
