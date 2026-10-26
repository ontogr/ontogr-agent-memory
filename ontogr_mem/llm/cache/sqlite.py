"""SQLite-backed completion cache with async thread-offloaded operations."""

import asyncio
import json
import sqlite3
import threading
from collections.abc import Callable

from ontogr_mem.llm.cache.port import CacheEntry, holds


class SqliteResponseCache:
    """Persist completion entries in a SQLite file that the cache owns."""

    def __init__(self, path: str = ":memory:") -> None:
        """Open a response cache at ``path`` and create its table if needed."""
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._connection.execute(
                "CREATE TABLE IF NOT EXISTS llm_cache "
                "(key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )
            self._connection.commit()

    async def get(self, key: str) -> CacheEntry | None:
        """Read an entry from SQLite without blocking the event loop."""
        return await asyncio.to_thread(self._get, key)

    async def put(self, key: str, entry: CacheEntry) -> None:
        """Write or replace an entry in SQLite."""
        await asyncio.to_thread(self._put, key, entry)

    async def delete(self, key: str) -> None:
        """Remove one entry if it exists."""
        await asyncio.to_thread(self._delete, key)

    async def evict_tag(self, tag: str) -> int:
        """Remove entries carrying ``tag`` and return the number removed."""
        return await asyncio.to_thread(self._evict, lambda entry: tag in entry.tags)

    async def count_tag(self, tag: str) -> int:
        """Count entries carrying ``tag`` without removing them."""
        return await asyncio.to_thread(self._count, lambda entry: tag in entry.tags)

    async def scan(self, needle: str) -> int:
        """Count entries containing ``needle`` without removing them."""
        return await asyncio.to_thread(self._count, lambda entry: holds(entry, needle))

    async def evict_text(self, needle: str) -> int:
        """Remove entries containing ``needle`` and return the number removed."""
        return await asyncio.to_thread(self._evict, lambda entry: holds(entry, needle))

    async def close(self) -> None:
        """Close the cache connection after outstanding calls complete."""
        await asyncio.to_thread(self._close)

    def _get(self, key: str) -> CacheEntry | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT value FROM llm_cache WHERE key = ?", (key,)
            ).fetchone()
        return None if row is None else _decode(row[0])

    def _put(self, key: str, entry: CacheEntry) -> None:
        value = json.dumps(
            {
                "prompt": entry.prompt,
                "text": entry.text,
                "input_tokens": entry.input_tokens,
                "output_tokens": entry.output_tokens,
                "tags": sorted(entry.tags),
            },
            sort_keys=True,
        )
        with self._lock:
            self._connection.execute(
                "INSERT INTO llm_cache (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )
            self._connection.commit()

    def _delete(self, key: str) -> None:
        with self._lock:
            self._connection.execute("DELETE FROM llm_cache WHERE key = ?", (key,))
            self._connection.commit()

    def _count(self, predicate: Callable[[CacheEntry], bool]) -> int:
        with self._lock:
            values = self._connection.execute("SELECT value FROM llm_cache").fetchall()
        return sum(predicate(_decode(row[0])) for row in values)

    def _evict(self, predicate: Callable[[CacheEntry], bool]) -> int:
        with self._lock:
            values = self._connection.execute(
                "SELECT key, value FROM llm_cache"
            ).fetchall()
            keys = [key for key, value in values if predicate(_decode(value))]
            self._connection.executemany(
                "DELETE FROM llm_cache WHERE key = ?", ((key,) for key in keys)
            )
            self._connection.commit()
        return len(keys)

    def _close(self) -> None:
        with self._lock:
            self._connection.close()


def _decode(value: str) -> CacheEntry:
    """Decode one serialized completion entry."""
    item = json.loads(value)
    return CacheEntry(
        prompt=item["prompt"],
        text=item["text"],
        input_tokens=item["input_tokens"],
        output_tokens=item["output_tokens"],
        tags=frozenset(item["tags"]),
    )
