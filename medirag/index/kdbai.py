from llama_index.core import VectorStoreIndex, StorageContext, Settings, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.kdbai import KDBAIVectorStore
import kdbai_client as kdbai
import os


class DailyMedIndexer:
    def __init__(self, model_name="nuvocare/WikiMedical_sent_biobert"):
        self.model_name = model_name
        self._initialize_embedding_model()
        self.session = kdbai.Session(api_key=os.getenv('KDBAI_API_KEY'),
                                     endpoint=os.getenv('KDBAI_ENDPOINT'))
        self.vector_store = KDBAIVectorStore(self.session.table("daily_med"), batch_size=1000)
        self.vector_store_index = None

    def _initialize_embedding_model(self):
        # Initialize the embedding model
        embed_model = HuggingFaceEmbedding(model_name=self.model_name)
        Settings.embed_model = embed_model

    def load_index(self, documents=None):
        """
        Load the index from existing storage or create a new one from provided documents.
        """
        self._initialize_embedding_model()

        if documents:
            return self._build_index_from_documents(documents)
        else:
            return self._load_existing_index()

    def _build_index_from_documents(self, documents):
        print("Building index from documents...")

        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        chunk = SemanticSplitterNodeParser(buffer_size=1, breakpoint_percentile_threshold=95,
                                           embed_model=Settings.embed_model)
        self.vector_store_index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            transformations=[chunk]
        )
        return self.vector_store_index

    def _load_existing_index(self):
        try:
            print("Loading index from storage context...")
            self.vector_store_index = VectorStoreIndex.from_vector_store(self.vector_store)
            return self.vector_store_index
        except Exception as e:
            raise ValueError(f"Failed to load index from storage context: {e}")

    def retrieve(self, query, top_k=3):
        """
        Retrieve the top-k results based on the query.
        """
        if not self.vector_store_index:
            raise ValueError("Vector store is not initialized. Please index documents first.")

        retriever = self.vector_store_index.as_retriever(similarity_top_k=top_k)
        return retriever.retrieve(query)
