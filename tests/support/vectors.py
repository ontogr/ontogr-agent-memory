"""Known vectors and harness types shared by the vector store tests."""

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass

from ontogr_mem.vectorstore import VectorKind, VectorStore
from tests.support.embedder import MockEmbedder


A = [1.0, 0.0, 0.0]
HALF = [0.5, 0.75**0.5, 0.0]
ORTHOGONAL = [0.0, 1.0, 0.0]
OPPOSITE = [-1.0, 0.0, 0.0]
FIXED = {
    "a": A,
    "half": HALF,
    "orthogonal": ORTHOGONAL,
    "opposite": OPPOSITE,
}


def fixed_embedder(**kwargs: object) -> MockEmbedder:
    """Return an embedder with known vectors for the texts in `FIXED`."""
    return MockEmbedder(vectors=FIXED, **kwargs)  # type: ignore[arg-type]


class ProbeMismatchEmbedder(MockEmbedder):
    """An embedder whose probe returns fewer values than its declared dimension."""

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Return a two-value vector for every text."""
        return [[1.0, 0.0] for _ in texts]


@dataclass
class VectorHarness:
    """A store on a fresh tenant plus the hooks a backend needs.

    Attributes:
        store: The store under test.
        seed: Creates the graph objects that will hold vectors.
        reopen: Builds a second store over the same data with another embedder.
        drop_index: Removes a vector index out of band, on backends that have one.
    """

    store: VectorStore
    seed: Callable[..., Awaitable[None]]
    reopen: Callable[[MockEmbedder], VectorStore]
    drop_index: Callable[[VectorKind], Awaitable[None]] | None = None


Factory = Callable[[MockEmbedder | None], Awaitable[VectorHarness]]


async def ready(
    factory: Factory, embedder: MockEmbedder | None = None
) -> VectorHarness:
    """Create a harness whose store is ready."""
    harness = await factory(embedder or fixed_embedder())
    await harness.store.ensure_ready()
    return harness
