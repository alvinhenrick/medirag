import pytest
from dotenv import load_dotenv

from medirag.index.local import DailyMedIndexer
from medirag.rag.wf import RAGWorkflow

load_dotenv()  # take environment variables from .env.


@pytest.mark.asyncio
async def test_wf_with_example(data_dir):
    # Example usage:
    index_path = data_dir.joinpath("daily_bio_bert_indexed")
    # Ensure the path is correct and the directory exists
    assert index_path.exists(), f"Directory not found: {index_path}"

    # Initialize the indexer and load the index
    indexer = DailyMedIndexer(persist_dir=index_path)
    indexer.load_index()

    top_k = 3  # Adjust the number of documents to retrieve
    top_n = 3  # Adjust the number of top-ranked documents to select

    # Pass the indexer to the workflow
    workflow = RAGWorkflow(indexer=indexer, timeout=60, top_k=top_k, top_n=top_n)
    query = "What information do you have about Clopidogrel?"

    result = await workflow.run(query=query)
    accumulated_response = ""
    if hasattr(result, 'async_response_gen'):
        async for chunk in result.async_response_gen():
            accumulated_response += chunk

    assert len(accumulated_response) > 0
