"""In-memory response cache for one event loop."""

from collections import OrderedDict

from ontogr_mem.llm.cache.port import CacheEntry, holds


class MemoryResponseCache:
    """Keep completions in least-recently-used order."""

    def __init__(self, max_entries: int = 2048) -> None:
        """Create an in-memory cache with a fixed entry capacity."""
        self._max_entries = max_entries
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()

    async def get(self, key: str) -> CacheEntry | None:
        """Return an entry and mark it as recently used."""
        entry = self._entries.get(key)
        if entry is not None:
            self._entries.move_to_end(key)
        return entry

    async def put(self, key: str, entry: CacheEntry) -> None:
        """Store an entry and evict the least recently used excess entries."""
        self._entries[key] = entry
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

    async def delete(self, key: str) -> None:
        """Remove one entry if it exists."""
        self._entries.pop(key, None)

    async def evict_tag(self, tag: str) -> int:
        """Remove every entry carrying ``tag`` and return the count."""
        keys = [key for key, entry in self._entries.items() if tag in entry.tags]
        for key in keys:
            del self._entries[key]
        return len(keys)

    async def count_tag(self, tag: str) -> int:
        """Count entries carrying ``tag`` without changing their order."""
        return sum(tag in entry.tags for entry in self._entries.values())

    async def scan(self, needle: str) -> int:
        """Count entries containing ``needle`` in their prompt or completion."""
        return sum(holds(entry, needle) for entry in self._entries.values())

    async def evict_text(self, needle: str) -> int:
        """Remove matching entries and return the number removed."""
        keys = [key for key, entry in self._entries.items() if holds(entry, needle)]
        for key in keys:
            del self._entries[key]
        return len(keys)
