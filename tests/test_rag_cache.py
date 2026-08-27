"""Cache key isolation for retrieval inputs and index versions."""

from src.cache.rag_cache import RAGCache


def _cache_without_redis():
    cache = object.__new__(RAGCache)
    cache._prefix = "test"
    cache._redis = None
    cache._ttl = 60
    return cache


def test_cache_key_normalizes_query_whitespace():
    cache = _cache_without_redis()
    assert cache._make_key("RTX  4060", 5) == cache._make_key("RTX 4060", 5)


def test_cache_key_isolates_filters_index_and_algorithm_versions():
    cache = _cache_without_redis()
    baseline = cache._make_key("laptop", 5)
    assert baseline != cache._make_key("laptop", 5, {"max_price": 8000})
    assert baseline != cache._make_key("laptop", 5, index_version="20260827")
    assert baseline != cache._make_key("laptop", 5, algorithm_version="rrf-v1")


def test_legacy_get_set_signatures_remain_valid():
    cache = _cache_without_redis()
    assert cache.get("laptop", 5) is None
    cache.set("laptop", 5, "result")
