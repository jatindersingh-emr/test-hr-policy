
import pytest
import time
from unittest.mock import patch, MagicMock

@pytest.fixture
def mock_policy_cache():
    """
    Fixture to provide a mock in-memory cache with TTL simulation.
    """
    class MockCache:
        def __init__(self, ttl=1.0):
            self._store = {}
            self._ttl = ttl

        def get(self, key):
            entry = self._store.get(key)
            if not entry:
                return None
            value, timestamp = entry
            if time.time() - timestamp > self._ttl:
                # Simulate cache expiry
                self._store.pop(key)
                return None
            return value

        def set(self, key, value):
            self._store[key] = (value, time.time())

        def clear(self):
            self._store.clear()

    return MockCache(ttl=1.0)  # 1 second TTL for test

@pytest.fixture
def mock_policy_retriever():
    """
    Fixture to provide a mock policy retriever function.
    """
    retriever = MagicMock()
    retriever.return_value = {"policy": "fresh_policy_content"}
    return retriever

@pytest.fixture
def api_ask_handler(mock_policy_cache, mock_policy_retriever):
    """
    Fixture to provide a simulated /api/ask handler that uses the mock cache and retriever.
    """
    def handler(user_input):
        # Simulate cache lookup
        cached = mock_policy_cache.get(user_input)
        if cached:
            return {"source": "cache", "policy": cached}
        # Simulate retrieval and cache update
        policy = mock_policy_retriever(user_input)["policy"]
        mock_policy_cache.set(user_input, policy)
        return {"source": "retrieved", "policy": policy}
    return handler

@pytest.mark.performance
def test_performance_policy_cache_expiry_and_refresh(api_ask_handler, mock_policy_cache, mock_policy_retriever):
    """
    Tests that cached policy content expires after the configured TTL and is refreshed on subsequent queries.

    Success criteria:
    - First query results in cache miss and triggers retrieval.
    - Second query (before TTL) results in cache hit.
    - After TTL, cache miss triggers retrieval and cache update.
    - Cache is updated with new content.
    """
    user_input = "What is the vacation policy?"

    # First query: should be a cache miss, triggers retrieval
    response1 = api_ask_handler(user_input)
    assert response1["source"] == "retrieved", "First query should retrieve policy (cache miss)"
    assert response1["policy"] == "fresh_policy_content"
    assert mock_policy_retriever.call_count == 1

    # Second query: within TTL, should be a cache hit
    response2 = api_ask_handler(user_input)
    assert response2["source"] == "cache", "Second query should hit cache"
    assert response2["policy"] == "fresh_policy_content"
    assert mock_policy_retriever.call_count == 1, "Policy retriever should not be called again before TTL"

    # Wait for TTL to expire
    time.sleep(1.1)

    # Third query: after TTL, should be a cache miss and trigger retrieval
    # Simulate retriever returning new content
    mock_policy_retriever.return_value = {"policy": "new_policy_content"}
    response3 = api_ask_handler(user_input)
    assert response3["source"] == "retrieved", "After TTL, should be cache miss and retrieve new policy"
    assert response3["policy"] == "new_policy_content"
    assert mock_policy_retriever.call_count == 2, "Policy retriever should be called again after TTL"

    # Fourth query: should be a cache hit with the new content
    response4 = api_ask_handler(user_input)
    assert response4["source"] == "cache", "After refresh, should hit cache with new content"
    assert response4["policy"] == "new_policy_content"
    assert mock_policy_retriever.call_count == 2

    # Error scenario: Cache does not expire as expected
    # (Simulated by forcibly setting a very long TTL and checking that cache is not refreshed)
    long_ttl_cache = type(mock_policy_cache)(ttl=100.0)
    long_ttl_handler = lambda inp: (
        long_ttl_cache.get(inp) or long_ttl_cache.set(inp, "long_ttl_policy") or {"source": "retrieved", "policy": "long_ttl_policy"}
    )
    long_ttl_handler(user_input)  # Populate cache
    time.sleep(1.1)
    # Should still be cache hit
    assert long_ttl_cache.get(user_input) == "long_ttl_policy", "Cache should not expire with long TTL"

    # Error scenario: Cache update fails (simulate by raising exception in set)
    class FailingCache:
        def get(self, key): return None
        def set(self, key, value): raise RuntimeError("Cache update failed")
    failing_cache = FailingCache()
    def failing_handler(user_input):
        try:
            cached = failing_cache.get(user_input)
            if cached:
                return {"source": "cache", "policy": cached}
            policy = mock_policy_retriever(user_input)["policy"]
            failing_cache.set(user_input, policy)
            return {"source": "retrieved", "policy": policy}
        except Exception as e:
            return {"error": str(e)}
    result = failing_handler(user_input)
    assert "Cache update failed" in result["error"], "Should report cache update failure"

