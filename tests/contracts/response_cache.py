"""Shared behavior required of each completion-cache adapter."""

from ontogr_mem.llm.cache import CacheEntry, ResponseCache


class ResponseCacheContract:
    """Behavior every response-cache adapter must preserve."""

    async def test_stored_entry_round_trips_all_fields(
        self, response_cache: ResponseCache
    ) -> None:
        """A cache entry preserves text, token counts, prompt, and tags."""
        entry = CacheEntry("prompt", "answer", 4, 2, frozenset({"tag"}))

        await response_cache.put("key", entry)

        assert await response_cache.get("key") == entry

    async def test_unknown_key_returns_none(
        self, response_cache: ResponseCache
    ) -> None:
        """A cache miss has no entry."""
        assert await response_cache.get("missing") is None

    async def test_delete_removes_only_the_requested_key(
        self, response_cache: ResponseCache
    ) -> None:
        """Deleting one key leaves another entry intact."""
        first = CacheEntry("one", "first", 1, 1)
        second = CacheEntry("two", "second", 2, 2)
        await response_cache.put("first", first)
        await response_cache.put("second", second)

        await response_cache.delete("first")

        assert await response_cache.get("first") is None
        assert await response_cache.get("second") == second

    async def test_evict_tag_removes_exactly_tagged_entries(
        self, response_cache: ResponseCache
    ) -> None:
        """Tag eviction removes matching entries and reports their count."""
        await response_cache.put("one", CacheEntry("a", "A", 1, 1, frozenset({"t"})))
        await response_cache.put(
            "two", CacheEntry("b", "B", 1, 1, frozenset({"t", "other"}))
        )
        await response_cache.put("three", CacheEntry("c", "C", 1, 1))

        assert await response_cache.evict_tag("t") == 2
        assert await response_cache.get("one") is None
        assert await response_cache.get("two") is None
        assert await response_cache.get("three") is not None

    async def test_count_tag_counts_without_removing_entries(
        self, response_cache: ResponseCache
    ) -> None:
        """Tag counts report matching entries without changing the cache."""
        entry = CacheEntry("a", "A", 1, 1, frozenset({"t"}))
        await response_cache.put("one", entry)

        assert await response_cache.count_tag("t") == 1
        assert await response_cache.get("one") == entry

    async def test_evict_text_removes_prompt_or_reply_matches(
        self, response_cache: ResponseCache
    ) -> None:
        """Text eviction matches prompts and replies, regardless of tags."""
        await response_cache.put("one", CacheEntry("contains needle", "A", 1, 1))
        await response_cache.put(
            "two", CacheEntry("b", "needle in reply", 1, 1, frozenset({"tag"}))
        )
        await response_cache.put("three", CacheEntry("c", "C", 1, 1))

        assert await response_cache.evict_text("needle") == 2
        assert await response_cache.get("one") is None
        assert await response_cache.get("two") is None
        assert await response_cache.get("three") is not None

    async def test_scan_counts_matches_without_removing_entries(
        self, response_cache: ResponseCache
    ) -> None:
        """Text scans report matches and leave cache entries in place."""
        entry = CacheEntry("needle", "answer", 1, 1)
        await response_cache.put("one", entry)

        assert await response_cache.scan("needle") == 1
        assert await response_cache.get("one") == entry

    async def test_put_replaces_an_existing_entry(
        self, response_cache: ResponseCache
    ) -> None:
        """Writing the same key replaces its prior value."""
        await response_cache.put("key", CacheEntry("old", "old", 1, 1))
        replacement = CacheEntry("new", "new", 2, 2)

        await response_cache.put("key", replacement)

        assert await response_cache.get("key") == replacement
