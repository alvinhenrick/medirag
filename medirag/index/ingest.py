from pathlib import Path

import faiss
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore


class DailyMedIndexer:
    def __init__(self,
                 model_name="sentence-transformers/all-mpnet-base-v2",
                 dimension=768):

        self.model_name = model_name
        self.dimension = dimension
        self.vector_store_index = None
        self.storage_context = self.create_faiss_index()

    def create_faiss_index(self):
        index = faiss.IndexFlatIP(self.dimension)
        vector_store = FaissVectorStore(faiss_index=index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return storage_context

    def build_index(self, documents):
        embed_model = HuggingFaceEmbedding(model_name=self.model_name)
        Settings.embed_model = embed_model
        self.vector_store_index = VectorStoreIndex.from_documents(documents, storage_context=self.storage_context)

    def save_index(self, persist_dir="./storage"):
        if self.vector_store_index:
            self.vector_store_index.storage_context.persist(Path(persist_dir))

    def load_index(self, persist_dir="./storage"):
        vector_store = FaissVectorStore.from_persist_dir(persist_dir)
        self.storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=persist_dir)
        self.vector_store_index = VectorStoreIndex(storage_context=self.storage_context)

    def retrieve(self, query, top_k=2):
        if self.vector_store_index:
            retriever = self.vector_store_index.as_retriever(similarity_top_k=top_k)
            return retriever.retrieve(query)
        else:
            raise ValueError("Index is not initialized. Please build or load an index first.")
