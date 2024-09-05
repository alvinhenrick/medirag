from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.postprocessor import LLMRerank
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.kdbai import KDBAIVectorStore
import kdbai_client as kdbai
import os
from loguru import logger

from medirag.index.abc import Indexer


class KDBAIDailyMedIndexer(Indexer):
    def __init__(self, model_name="nuvocare/WikiMedical_sent_biobert", table_name="daily_med"):
        self.model_name = model_name
        self.table_name = table_name
        self._initialize_embedding_model()
        self.session = self._initialize_kdbai_session()
        self.vector_store = self._initialize_vector_store()

    def _initialize_embedding_model(self):
        # Initialize the embedding model and assign to settings
        embed_model = HuggingFaceEmbedding(model_name=self.model_name)
        Settings.embed_model = embed_model
        logger.info("Embedding model initialized.")

    @staticmethod
    def _initialize_kdbai_session():
        # Initialize KDBAI session
        api_key = os.getenv("KDBAI_API_KEY")
        endpoint = os.getenv("KDBAI_ENDPOINT")
        session = kdbai.Session(api_key=api_key, endpoint=endpoint)
        logger.debug("KDBAI session initialized.")
        return session

    def _initialize_vector_store(self):
        # Initialize vector store using the session
        vector_store = KDBAIVectorStore(self.session.table(self.table_name), batch_size=100)
        logger.debug(f"Vector store initialized for table: {self.table_name}.")
        return vector_store

    def load_index(self, documents=None):
        """
        Load the index from existing storage or create a new one from provided documents.
        """
        if documents:
            return self._build_index_from_documents(documents)
        else:
            return self._load_existing_index()

    def _build_index_from_documents(self, documents):
        logger.info("Building index from documents...")
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        chunk = SemanticSplitterNodeParser(
            buffer_size=1, breakpoint_percentile_threshold=95, embed_model=Settings.embed_model
        )
        self.vector_store_index = VectorStoreIndex.from_documents(
            documents, storage_context=storage_context, transformations=[chunk]
        )
        return self.vector_store_index

    def _load_existing_index(self):
        try:
            logger.info("Loading index from storage context...")
            self.vector_store_index = VectorStoreIndex.from_vector_store(self.vector_store)
            return self.vector_store_index
        except Exception as e:
            logger.error(f"Failed to load index from storage context: {e}")
            raise

    def retrieve(self, query, top_k: int = 3, with_reranker: bool = False):
        """
        Retrieve the top-k results based on the query.
        """
        if not self.vector_store_index:
            logger.error("Vector store is not initialized. Please index documents first.")
            raise ValueError("Vector store is not initialized. Please index documents first.")

        retriever = self.vector_store_index.as_retriever(similarity_top_k=(top_k * 3 if with_reranker else top_k))
        nodes = retriever.retrieve(query)
        logger.info(f"Retrieved {len(nodes)} nodes.")

        if with_reranker:
            ranker = LLMRerank(choice_batch_size=top_k, top_n=top_k)
            ranked_nodes = ranker.postprocess_nodes(nodes, query_str=query)
            logger.info(f"Reranked nodes to {len(ranked_nodes)}")
            return ranked_nodes
        else:
            return nodes
