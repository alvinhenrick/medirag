import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer

from medirag.rag.qa import RAG


class SemanticCaching:
    def __init__(self,
                 model_name='sentence-transformers/all-mpnet-base-v2',
                 dimension=768,
                 json_file='cache.json',
                 cosine_threshold=0.7,
                 rag: RAG = None):

        self.model_name = model_name
        self.dimension = dimension
        self.cosine_threshold = cosine_threshold
        self.rag = rag
        # Initialize Faiss index for cosine similarity
        self.vector_index = faiss.IndexFlatIP(self.dimension)
        self.encoder = SentenceTransformer(self.model_name)
        self.json_file = json_file
        self.cache = self.load_cache()

    def load_cache(self):
        local_cache = {'questions': [], 'embeddings': [], 'response_text': []}
        try:
            if self.json_file:
                with open(self.json_file, 'r') as file:
                    local_cache = json.load(file)
                    if 'embeddings' in local_cache and len(local_cache['embeddings']) > 0:
                        for embedding in local_cache['embeddings']:
                            _embedding = np.array(embedding, dtype=np.float32)
                            # Add the embeddings to the Faiss index
                            self.vector_index.add(_embedding)
                return local_cache
            else:
                return local_cache
        except FileNotFoundError:
            return local_cache
        except Exception as e:
            print(f"Failed to load or process cache: {e}")
            # Reset the cache if there's an error
            return local_cache

    def save_cache(self):
        with open(self.json_file, 'w') as file:
            json.dump(self.cache, file)

    def ask(self, question: str) -> str:
        # Encode the question to get the embedding and normalize it
        embedding = self.encoder.encode([question], show_progress_bar=False)
        faiss.normalize_L2(embedding)

        # Search in the index
        D, I = self.vector_index.search(embedding, 1)

        if D[0][0] >= self.cosine_threshold:
            # Return the cached response
            row_id = I[0][0]
            return self.cache['response_text'][row_id]
        else:
            # Generate a new answer
            answer = self.invoke_rag(question)
            self.cache['questions'].append(question)
            self.cache['embeddings'].append(embedding.tolist())
            self.cache['response_text'].append(answer)

            # Add new normalized embedding to the index
            self.vector_index.add(embedding)
            self.save_cache()
            return answer

    def invoke_rag(self, question: str):
        response = self.rag(question)
        return response.answer
