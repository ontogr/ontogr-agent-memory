"""A deterministic embedder for vector store tests."""

import hashlib
import math
import random
from collections.abc import Mapping, Sequence

from ontogr_mem.vectorstore import (
    EmbeddingGateway,
    PermanentEmbeddingError,
    TransientEmbeddingError,
    VectorStoreSettings,
)


DIM = 3


class MockEmbedder:
    """Unit vectors seeded by the SHA-256 of the text, unless the test fixes them.

    Args:
        dim: Vector length.
        model_id: Model identity string.
        max_input_tokens: Longest input the model accepts.
        vectors: Vectors to return for exact texts instead of the seeded ones.
    """

    def __init__(
        self,
        dim: int = DIM,
        model_id: str = "mock",
        max_input_tokens: int = 8192,
        vectors: Mapping[str, Sequence[float]] | None = None,
    ) -> None:
        """Create the embedder."""
        self.dim = dim
        self.model_id = model_id
        self.max_input_tokens = max_input_tokens
        self._vectors = dict(vectors or {})

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed deterministically."""
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        if text in self._vectors:
            return list(self._vectors[text])
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest(), "big")
        rng = random.Random(seed)
        vec = [rng.gauss(0.0, 1.0) for _ in range(self.dim)]
        length = math.sqrt(sum(x * x for x in vec))
        return [x / length for x in vec]


class FailingEmbedder(MockEmbedder):
    """An embedder whose failures the test scripts.

    Args:
        transient_over: Batches longer than this raise a transient error.
        permanent: Every call raises a permanent error.
    """

    def __init__(
        self, *, transient_over: int | None = None, permanent: bool = False
    ) -> None:
        """Create the embedder."""
        super().__init__()
        self._transient_over = transient_over
        self._permanent = permanent

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed, or fail as scripted."""
        if self._permanent:
            msg = "bad credentials"
            raise PermanentEmbeddingError(msg)
        if self._transient_over is not None and len(texts) > self._transient_over:
            msg = "rate limited"
            raise TransientEmbeddingError(msg)
        return await super().embed_batch(texts)


class FlakyEmbedder(MockEmbedder):
    """An embedder that fails a fixed number of calls before recovering."""

    def __init__(self, *, failures: int) -> None:
        """Fail the first `failures` calls, then embed normally."""
        super().__init__()
        self.failures = failures
        self.calls = 0

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Count the call and raise a transient error while failures remain."""
        self.calls += 1
        if self.calls <= self.failures:
            raise TransientEmbeddingError("rate limited")
        return await super().embed_batch(texts)


class RecordingEmbedder(MockEmbedder):
    """An embedder that records the texts it embedded and can be made to fail.

    Args:
        failing: Whether every call fails.
        vectors: Vectors to return for exact texts instead of the seeded ones.
    """

    def __init__(
        self,
        *,
        failing: bool = False,
        vectors: Mapping[str, Sequence[float]] | None = None,
    ) -> None:
        """Create the embedder."""
        super().__init__(vectors=vectors)
        self.failing = failing
        self.embedded: list[str] = []
        self.batches: list[tuple[str, ...]] = []

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed the texts, or fail while the embedder is failing."""
        if self.failing:
            msg = "provider is unavailable"
            raise PermanentEmbeddingError(msg)
        self.embedded.extend(texts)
        self.batches.append(tuple(texts))
        return await super().embed_batch(texts)


async def no_sleep(seconds: float) -> None:
    """Skip the retry delay."""


def gateway(embedder: MockEmbedder, **settings: float) -> EmbeddingGateway:
    """Wrap `embedder` in a gateway that never sleeps between retries."""
    return EmbeddingGateway(embedder, VectorStoreSettings(**settings), sleep=no_sleep)
