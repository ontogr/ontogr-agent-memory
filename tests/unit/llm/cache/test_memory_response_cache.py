"""In-memory completion caching, including its least-recently-used bound."""

import pytest

from ontogr_mem.llm.cache import CacheEntry, MemoryResponseCache
from tests.contracts.response_cache import ResponseCacheContract


class TestMemoryResponseCache(ResponseCacheContract):
    """The in-memory cache implements the shared cache contract."""

    @pytest.fixture
    def response_cache(self) -> MemoryResponseCache:
        """Return a fresh in-memory cache for one test."""
        return MemoryResponseCache()

    async def test_get_marks_a_key_recently_used_before_eviction(self) -> None:
        """An accessed entry survives when the cache exceeds its capacity."""
        cache = MemoryResponseCache(max_entries=2)
        first = CacheEntry("one", "one", 1, 1)
        second = CacheEntry("two", "two", 1, 1)
        await cache.put("one", first)
        await cache.put("two", second)

        assert await cache.get("one") == first
        await cache.put("three", CacheEntry("three", "three", 1, 1))

        assert await cache.get("one") == first
        assert await cache.get("two") is None
