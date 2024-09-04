from medirag.cache.local import LocalSemanticCache
from medirag.index.local import LocalIndexer

# from medirag.index.kdbai import KDBAIDailyMedIndexer
from medirag.rag.qa import RAG, DailyMedRetrieve
import dspy

from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.


def ask_med_question(sm, rag, query):
    response = sm.lookup(question=query, cosine_threshold=0.9)
    if not response:
        response = rag(query).answer
        sm.save(query, response)
    return response


def test_rag_with_example(data_dir):
    # Example usage:
    index_path = data_dir.joinpath("daily_bio_bert_indexed")
    # Ensure the path is correct and the directory exists
    assert index_path.exists(), f"Directory not found: {index_path}"

    # Index and query documents
    indexer = LocalIndexer(persist_dir=index_path)
    indexer.load_index()
    rm = DailyMedRetrieve(indexer=indexer)

    query = "What information do you have about Clopidogrel?"
    turbo = dspy.OpenAI(model="gpt-3.5-turbo")

    dspy.settings.configure(lm=turbo, rm=rm)

    rag = RAG(k=3)

    sm = LocalSemanticCache(
        model_name="sentence-transformers/all-mpnet-base-v2", dimension=768, json_file="rag_test_cache.json"
    )

    result1 = ask_med_question(sm, rag, query)
    print(result1)
    result2 = ask_med_question(sm, rag, query)

    assert result1 == result2
