import pytest
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings

from medirag.cache.local import LocalSemanticCache

# from medirag.index.kdbai import KDBAIDailyMedIndexer
from medirag.index.local import LocalIndexer

from medirag.rag.llama_index import WorkflowRAG
from medirag.rag.qa_rag import QuestionAnswerRunner


@pytest.mark.asyncio
async def test_wf_with_example(data_dir):
    # Example usage:
    index_path = data_dir.joinpath("daily_bio_bert_indexed")
    # Ensure the path is correct and the directory exists
    assert index_path.exists(), f"Directory not found: {index_path}"

    # Initialize the indexer and load the index
    indexer = LocalIndexer(persist_dir=index_path)
    indexer.load_index()

    top_k = 6  # Adjust the number of documents to retrieve
    top_n = 3  # Adjust the number of top-ranked documents to select
    Settings.llm = OpenAI(model="gpt-3.5-turbo")

    # Pass the indexer to the workflow
    rag = WorkflowRAG(indexer=indexer, timeout=60, top_k=top_k, top_n=top_n)

    sm = LocalSemanticCache(
        model_name="sentence-transformers/all-mpnet-base-v2", dimension=768, json_file="test_llama_index_wf.json"
    )

    query = "What information do you have about Clopidogrel?"

    qa = QuestionAnswerRunner(sm=sm, rag=rag)

    response_1 = qa.ask(query, enable_stream=True)
    result_1 = ""
    async for chunk in response_1:
        result_1 += chunk

    response_2 = qa.ask(query, enable_stream=True)
    result_2 = ""
    async for chunk in response_2:
        result_2 += chunk

    assert result_1 == result_2

    sm.clear()
