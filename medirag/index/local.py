from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import VectorStoreIndex, Settings, load_index_from_storage, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from loguru import logger
import faiss

from medirag.index.ptc import Indexer
from medirag.index.utils import retrieve_common


class LocalIndexer(Indexer):
    def __init__(self, model_name="nuvocare/WikiMedical_sent_biobert", dimension=768, persist_dir="./storage"):
        self._vector_store_index = None
        self.model_name = model_name
        self.dimension = dimension
        self.persist_dir = persist_dir
        self._initialize()

    @property
    def vector_store_index(self):
        return self._vector_store_index

    def _initialize(self):
        # Initialize embedding model
        embed_model = HuggingFaceEmbedding(model_name=self.model_name)
        Settings.embed_model = embed_model
        logger.info("Embedding model initialized.")

        # Load or create FAISS index
        try:
            self.storage_context = self._load_storage_context()
        except Exception as e:
            logger.warning(f"Failed to load storage context from disk: {e}. Creating a new FAISS index.")
            self.storage_context = self._create_faiss_index()

    def _load_storage_context(self):
        vector_store = FaissVectorStore.from_persist_dir(self.persist_dir)
        storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=self.persist_dir)
        logger.info("Storage context loaded successfully from disk.")
        return storage_context

    def _create_faiss_index(self):
        index = faiss.IndexFlatIP(self.dimension)
        vector_store = FaissVectorStore(faiss_index=index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return storage_context

    def load_index(self, documents=None):
        if documents:
            logger.info("Building index from documents...")
            self._vector_store_index = VectorStoreIndex.from_documents(documents, storage_context=self.storage_context)
        else:
            self._load_existing_index()

    def _load_existing_index(self):
        try:
            # Using `load_index_from_storage` to load an existing index
            self._vector_store_index = load_index_from_storage(storage_context=self.storage_context)
            logger.info("Index loaded from storage context.")
        except Exception as e:
            logger.error(f"Failed to load index from storage context: {e}")
            raise ValueError("Failed to load index from storage context.")

    def retrieve(self, query, top_k: int = 3, with_reranker: bool = False):
        """
        Retrieve the top-k results based on the query.
        """
        return retrieve_common(self.vector_store_index, query, top_k, with_reranker)
