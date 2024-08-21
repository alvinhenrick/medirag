import faiss
from llama_index.core import VectorStoreIndex, StorageContext, Settings, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore


class DailyMedIndexer:
    def __init__(self,
                 model_name="dmis-lab/biobert-base-cased-v1.2",
                 dimension=768,
                 persist_dir="./storage"):

        self.model_name = model_name
        self.dimension = dimension
        self.persist_dir = persist_dir

        self.storage_context = self._initialize_storage_context()
        self.vector_store_index = None

    def _initialize_storage_context(self):
        """
        Attempt to load the storage context from disk; otherwise, create a new one.
        """
        try:
            return self._load_storage_context()
        except Exception as e:
            print(f"Failed to load storage context from disk: {e}. Creating a new storage context.")
            return self._create_faiss_index()

    def _load_storage_context(self):
        vector_store = FaissVectorStore.from_persist_dir(self.persist_dir)
        storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=self.persist_dir)
        print("Storage context loaded successfully from disk.")
        return storage_context

    def _create_faiss_index(self):
        index = faiss.IndexFlatIP(self.dimension)
        vector_store = FaissVectorStore(faiss_index=index)
        return StorageContext.from_defaults(vector_store=vector_store)

    def load_index(self, documents=None):
        """
        Load the index from existing storage or create a new one from provided documents.
        """
        self._initialize_embedding_model()

        if documents:
            return self._build_index_from_documents(documents)
        else:
            return self._load_existing_index()

    def _initialize_embedding_model(self):
        embed_model = HuggingFaceEmbedding(model_name=self.model_name)
        Settings.embed_model = embed_model

    def _build_index_from_documents(self, documents):
        print("Building index from documents...")
        self.vector_store_index = VectorStoreIndex.from_documents(documents, storage_context=self.storage_context)
        return self.vector_store_index

    def _load_existing_index(self):
        try:
            print("Loading index from storage context...")
            self.vector_store_index = load_index_from_storage(storage_context=self.storage_context)
            return self.vector_store_index
        except Exception as e:
            raise ValueError(f"Failed to load index from storage context: {e}")

    def save_index(self, persist_dir=None):
        """
        Persist the index to disk.
        """
        if not self.vector_store_index:
            raise ValueError("No index to save. Load or build an index first.")

        persist_dir = persist_dir or self.persist_dir
        self.vector_store_index.storage_context.persist(persist_dir)
        print(f"Index saved to {persist_dir}.")

    def retrieve(self, query, top_k=3):
        """
        Retrieve the top-k results based on the query.
        """
        if not self.vector_store_index:
            raise ValueError("Index is not initialized. Please build or load an index first.")

        retriever = self.vector_store_index.as_retriever(similarity_top_k=top_k)
        return retriever.retrieve(query)
