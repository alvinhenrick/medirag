from medirag.cache.local import SemanticCaching
from medirag.index.local import DailyMedIndexer
from medirag.rag.qa import RAG, DailyMedRetrieve
import dspy

from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.


def test_rag_with_example(data_dir):
    # Example usage:
    index_path = data_dir.joinpath("daily_med_indexed")
    # Ensure the path is correct and the directory exists
    assert index_path.exists(), f"Directory not found: {index_path}"

    # Index and query documents
    indexer = DailyMedIndexer(persist_dir=index_path)
    indexer.load_index()
    rm = DailyMedRetrieve(daily_med_indexer=indexer)

    query = "What are the key things about the drug's usage?"
    turbo = dspy.OpenAI(model='gpt-3.5-turbo')

    dspy.settings.configure(lm=turbo, rm=rm)

    rag = RAG(k=3)

    sm = SemanticCaching(model_name='all-mpnet-base-v2', dimension=768,
                         json_file='rag_test_cache.json', rag=rag)
    sm.load_cache()

    result1 = sm.ask(query)
    print(result1)
    result2 = sm.ask(query)

    assert result1 == result2
