"""SQLite response-cache contract and persistence across cache instances."""

import pytest

from ontogr_mem.llm.cache import CacheEntry, SqliteResponseCache
from tests.contracts.response_cache import ResponseCacheContract


class TestSqliteResponseCache(ResponseCacheContract):
    """The SQLite adapter implements the cache contract on a temporary file."""

    @pytest.fixture
    async def response_cache(self, tmp_path):
        """Open and close an isolated SQLite response cache."""
        cache = SqliteResponseCache(str(tmp_path / "responses.sqlite"))
        try:
            yield cache
        finally:
            await cache.close()

    async def test_entries_survive_opening_a_second_cache_instance(self, tmp_path):
        """A committed cache entry remains readable after reopening the file."""
        path = str(tmp_path / "responses.sqlite")
        first = SqliteResponseCache(path)
        entry = CacheEntry("prompt", "answer", 3, 2, frozenset({"tag"}))
        await first.put("key", entry)
        await first.close()

        second = SqliteResponseCache(path)
        try:
            assert await second.get("key") == entry
        finally:
            await second.close()
