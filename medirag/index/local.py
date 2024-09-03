from loguru import logger
import faiss
from llama_index.core import VectorStoreIndex, StorageContext, Settings, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

from medirag.index.common import Indexer


class LocalIndexer(Indexer):
    def __init__(self, model_name="nuvocare/WikiMedical_sent_biobert",
                 dimension=768, persist_dir="./storage"):
        self.vector_store_index = None
        self.model_name = model_name
        self.dimension = dimension
        self.persist_dir = persist_dir
        self.storage_context = self._initialize_storage_context()
        self._initialize_embedding_model()

    def _initialize_storage_context(self):
        try:
            return self._load_storage_context()
        except Exception as e:
            logger.warning(f"Failed to load storage context from disk: {e}. Creating a new storage context.")
            return self._create_faiss_index()

    def _load_storage_context(self):
        vector_store = FaissVectorStore.from_persist_dir(self.persist_dir)
        storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=self.persist_dir)
        logger.info("Storage context loaded successfully from disk.")
        return storage_context

    def _create_faiss_index(self):
        index = faiss.IndexFlatIP(self.dimension)
        vector_store = FaissVectorStore(faiss_index=index)
        return StorageContext.from_defaults(vector_store=vector_store)

    def _initialize_embedding_model(self):
        try:
            embed_model = HuggingFaceEmbedding(model_name=self.model_name)
            Settings.embed_model = embed_model
            logger.info("Embedding model initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")

    def load_index(self, documents=None):
        if documents:
            logger.info("Building index from documents...")
            self.storage_context = self._create_faiss_index()
            self.vector_store_index = VectorStoreIndex.from_documents(documents, storage_context=self.storage_context)
            return self.vector_store_index
        else:
            return self._load_existing_index()

    def _load_existing_index(self):
        try:
            self.vector_store_index = load_index_from_storage(storage_context=self.storage_context)
            logger.info("Index loaded from storage context.")
            return self.vector_store_index
        except Exception as e:
            logger.error(f"Failed to load index from storage context: {e}")
            raise ValueError("Failed to load index from storage context.")

    def save_index(self):
        if not self.vector_store_index:
            logger.error("No index to save. Load or build an index first.")
            raise ValueError("No index to save.")
        self.vector_store_index.storage_context.persist(self.persist_dir)
        logger.info(f"Index saved to {self.persist_dir}.")

    def retrieve(self, query, top_k=3):
        if not self.vector_store_index:
            logger.error("Index is not initialized. Please build or load an index first.")
            raise ValueError("Index is not initialized.")
        retriever = self.vector_store_index.as_retriever(similarity_top_k=top_k)
        return retriever.retrieve(query)
