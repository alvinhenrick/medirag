import faiss
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore


class DailyMedIndexer:
    def __init__(self,
                 model_name="sentence-transformers/all-mpnet-base-v2",
                 dimension=768,
                 persist_dir="./storage"):
        self.model_name = model_name
        self.dimension = dimension
        self.persist_dir = persist_dir

        # Initialize storage context and load the index
        self.storage_context = self.build_storage_context()
        self.vector_store_index = None

    def build_storage_context(self):
        try:
            vector_store = FaissVectorStore.from_persist_dir(self.persist_dir)
            storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=self.persist_dir)
            print("Storage context loaded successfully from disk.")
        except Exception as e:
            # If loading fails, create a new Faiss index
            print(f"Failed to load storage context from disk: {e}. Creating a new storage context.")
            storage_context = self._create_faiss_index()

        return storage_context

    def _create_faiss_index(self):
        index = faiss.IndexFlatIP(self.dimension)
        vector_store = FaissVectorStore(faiss_index=index)
        return StorageContext.from_defaults(vector_store=vector_store)

    def load_index(self, documents=None):
        # Initialize the embedding model and set it in the settings
        embed_model = HuggingFaceEmbedding(model_name=self.model_name)
        Settings.embed_model = embed_model

        if documents is not None:
            print("Building index from documents...")
            self.vector_store_index = VectorStoreIndex.from_documents(documents, storage_context=self.storage_context)
        else:
            try:
                print("Loading index from storage context...")
                self.vector_store_index = VectorStoreIndex(storage_context=self.storage_context)
            except Exception as e:
                raise ValueError(f"Failed to load index from storage context: {e}")

        return self.vector_store_index

    def save_index(self, persist_dir=None):
        if self.vector_store_index:
            persist_dir = persist_dir if persist_dir else self.persist_dir
            self.vector_store_index.storage_context.persist(persist_dir)

    def retrieve(self, query, top_k=2):
        if not self.vector_store_index:
            raise ValueError("Index is not initialized. Please build or load an index first.")

        retriever = self.vector_store_index.as_retriever(similarity_top_k=top_k)
        return retriever.retrieve(query)
