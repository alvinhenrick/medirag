from pathlib import Path

import faiss
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from sentence_transformers import SentenceTransformer


class SemanticCaching:
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2", dimension=768, cosine_threshold=0.7):
        self.model_name = model_name
        self.dimension = dimension
        self.cosine_threshold = cosine_threshold
        self.index = None
        self.storage_context = None
        self.encoder = SentenceTransformer(self.model_name)

    def create_faiss_index(self):
        index = faiss.IndexFlatIP(self.dimension)
        vector_store = FaissVectorStore(faiss_index=index)
        self.storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return vector_store

    def build_index(self, documents):
        embed_model = HuggingFaceEmbedding(model_name=self.model_name)
        Settings.embed_model = embed_model
        self.index = VectorStoreIndex.from_documents(documents, storage_context=self.storage_context)

    def save_cache(self, persist_dir="./cache"):
        if self.index:
            self.index.storage_context.persist(Path(persist_dir))

    def load_cache(self, persist_dir="./cache"):
        vector_store = FaissVectorStore.from_persist_dir(persist_dir)
        self.storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=persist_dir)
        self.index = VectorStoreIndex(storage_context=self.storage_context)

    def ask(self, question: str) -> str:
        if not self.index:
            raise ValueError("Index is not initialized. Please build or load an index first.")
        # TODO RAG
        query_engine = self.index.as_query_engine()
        response = query_engine.query(question)
        return response
