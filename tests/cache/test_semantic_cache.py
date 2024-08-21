import pytest

from medirag.cache.local import SemanticCaching


# Test the SemanticCaching class with real embeddings and index interactions
@pytest.fixture(scope="module")
def semantic_caching():
    # This will actually initialize the model and the index
    return SemanticCaching(model_name='dmis-lab/biobert-base-cased-v1.2', dimension=768,
                           json_file='real_test_cache.json')


def test_ask_cached_interaction(mocker, semantic_caching):
    # Mock the `invoke_rag` method to control its behavior and monitor its usage
    mock_invoke_rag = mocker.patch.object(semantic_caching, 'invoke_rag',
                                          return_value="Paris")

    # First invocation: this should lead to `invoke_rag` being called
    first_response = semantic_caching.ask("What is the capital of France?")
    assert first_response == "Paris"
    mock_invoke_rag.assert_called_once()  # Confirm it was called

    # Second invocation: this should use the cached result, not call `invoke_rag` again
    second_response = semantic_caching.ask("What is the capital of France?")
    assert second_response == "Paris"
    # Check that `invoke_rag` was still only called once (i.e., no additional calls)
    mock_invoke_rag.assert_called_once()

    # The response should be the same, and `invoke_rag` should not have been called again
    assert first_response == second_response
