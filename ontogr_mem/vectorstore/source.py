"""Lazy loading for embedding providers."""

import asyncio
from collections.abc import Awaitable, Callable
from enum import StrEnum

from ontogr_mem.vectorstore.embedder import Embedder


class EmbedderState(StrEnum):
    """Loading state of an embedder source."""

    NOT_LOADED = "not_loaded"
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"


class EmbedderSource:
    """Load and retain an embedder on its first call."""

    def __init__(self, loader: Callable[[], Awaitable[Embedder]]) -> None:
        self._loader = loader
        self._embedder: Embedder | None = None
        self._lock = asyncio.Lock()
        self.state = EmbedderState.NOT_LOADED

    async def __call__(self) -> Embedder:
        """Return the loaded embedder, retrying after a failed load."""
        if self._embedder is not None:
            return self._embedder
        async with self._lock:
            if self._embedder is not None:
                return self._embedder
            self.state = EmbedderState.LOADING
            try:
                embedder = await self._loader()
            except BaseException:
                self.state = EmbedderState.FAILED
                raise
            self._embedder = embedder
            self.state = EmbedderState.READY
            return embedder
