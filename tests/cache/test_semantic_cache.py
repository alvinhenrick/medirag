import pytest
from medirag.cache.local import LocalSemanticCache


# Fixture to initialize the SemanticCaching object
@pytest.fixture(scope="module")
def semantic_cache():
    # Initialize the SemanticCaching class with a test cache file
    return LocalSemanticCache(
        model_name="sentence-transformers/all-mpnet-base-v2", dimension=768, json_file="real_test_cache.json"
    )


def test_save_and_lookup_in_cache(semantic_cache):
    # Clear any existing cache data
    semantic_cache.clear()

    # Step 1: Lookup should return None for a question not in the cache
    initial_lookup = semantic_cache.lookup("What is the capital of France?")
    assert initial_lookup is None

    # Step 2: Save a response to the cache
    semantic_cache.save("What is the capital of France?", "Paris")

    # Step 3: Lookup the same question; it should now return the cached response
    cached_response = semantic_cache.lookup("What is the capital of France?")
    assert cached_response is not None
    assert cached_response == "Paris"

    # Cleanup: Clear the cache after test
    semantic_cache.clear()
