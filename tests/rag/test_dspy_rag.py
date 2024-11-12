import pytest

from medirag.cache.local import LocalSemanticCache
from medirag.index.local import LocalIndexer

# from medirag.index.kdbai import KDBAIDailyMedIndexer
from medirag.rag.dspy import DspyRAG, DailyMedRetrieve
import dspy

from medirag.rag.qa_rag import QuestionAnswerRunner


@pytest.mark.asyncio
async def test_rag_with_example(data_dir):
    # Example usage:
    index_path = data_dir.joinpath("daily_bio_bert_indexed")
    # Ensure the path is correct and the directory exists
    assert index_path.exists(), f"Directory not found: {index_path}"

    # Index and query documents
    # indexer = KDBAIDailyMedIndexer()
    indexer = LocalIndexer(persist_dir=index_path)
    indexer.load_index()
    rm = DailyMedRetrieve(indexer=indexer)

    query = "What information do you have about Clopidogrel?"
    turbo = dspy.OpenAI(model="gpt-4o-mini", max_tokens=4000)

    top_k = 3  # Adjust the number of documents to retrieve
    dspy.settings.configure(lm=turbo, rm=rm)

    rag = DspyRAG(k=top_k)

    sm = LocalSemanticCache(
        model_name="sentence-transformers/all-mpnet-base-v2", dimension=768, json_file="test_dspy_rag.json"
    )

    qa = QuestionAnswerRunner(sm=sm, rag=rag)

    response_1 = qa.ask(query, enable_stream=False)
    result_1 = ""
    async for chunk in response_1:
        result_1 += chunk

    response_2 = qa.ask(query, enable_stream=False)
    result_2 = ""
    async for chunk in response_2:
        result_2 += chunk

    assert result_1 == result_2

    sm.clear()
