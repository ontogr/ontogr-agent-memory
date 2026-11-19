"""The vector store contract on the in-memory twin."""

from collections.abc import Sequence

import pytest

from ontogr_mem.vectorstore import (
    EdgeExtras,
    EmbeddingPurpose,
    EmbedItem,
    Hit,
    IndexDriftError,
    SearchFilters,
    SpaceMismatchError,
    SpaceSpec,
    TemporalFilter,
    Validity,
    VectorKind,
    VectorStore,
)
from ontogr_mem.vectorstore.memory import InMemoryBackend
from tests.contracts.vector_store import VectorStoreContract
from tests.support.embedder import MockEmbedder, embedder_source
from tests.support.vectors import A, Factory, VectorHarness, fixed_embedder


class InvalidQueryProbeEmbedder(MockEmbedder):
    """Return a bad vector only for the query encoding purpose."""

    def __init__(self, invalid_vector: list[float]) -> None:
        """Keep the query vector invalid while document vectors remain valid."""
        super().__init__()
        self.invalid_vector = invalid_vector

    async def embed_batch(
        self, texts: Sequence[str], *, purpose: EmbeddingPurpose
    ) -> list[list[float]]:
        """Return the invalid vector only for query-purpose probes."""
        if purpose == "query":
            return [self.invalid_vector.copy() for _ in texts]
        return await super().embed_batch(texts, purpose=purpose)


class GraphView:
    """Graph facts used to check search and hydration against graph state."""

    def __init__(self) -> None:
        """Start with both facts present and one valid at the requested time."""
        self.ids = {"f1", "f2"}

    async def exists(self, kind, ids):
        """Return graph objects that still exist."""
        return set(ids) & self.ids

    async def iter_items(self, kind, after, limit):
        """Return no backfill items."""
        return []

    async def still_valid(self, ids, temporal):
        """Return the graph facts believed at the requested time."""
        return set(ids) & {"f1"}


class FlakyIndexBackend(InMemoryBackend):
    """An in-memory twin whose index can drift and refuse searches."""

    def __init__(self) -> None:
        """Start with an operational index and no pending recheck failure."""
        super().__init__()
        self.index_operational = True
        self.fail_recheck = False

    async def ensure_ready(self, specs: Sequence[SpaceSpec]) -> None:
        """Fail on demand, otherwise restore the index to an operational state."""
        if self.fail_recheck:
            raise IndexDriftError("index is not operational")
        await super().ensure_ready(specs)
        self.index_operational = True

    async def search(
        self,
        kind: VectorKind,
        vector: Sequence[float],
        k: int,
        filters: SearchFilters,
        min_score: float | None,
    ) -> list[tuple[str, float]]:
        """Refuse searches while the index is not operational."""
        if not self.index_operational:
            raise IndexDriftError("index is not operational")
        return await super().search(kind, vector, k, filters, min_score)


class TestMemoryVectorStore(VectorStoreContract):
    """VectorStore over the in-memory backend satisfies the contract."""

    @pytest.fixture
    def make_harness(self) -> Factory:
        """Return a factory for twins that stand in for fresh tenants."""

        async def make(embedder: MockEmbedder | None) -> VectorHarness:
            assert embedder is not None
            backend = InMemoryBackend()

            def store_for(other: MockEmbedder) -> VectorStore:
                return VectorStore(backend, embedder_source(other))

            async def seed(kind, ids, **props: object) -> None:
                return None

            return VectorHarness(store_for(embedder), seed, store_for)

        return make

    async def test_graph_rechecks_known_at_search_and_hydrates_current_objects(self):
        """One graph view drives past-belief filtering and hydration."""
        graph = GraphView()
        backend = InMemoryBackend()
        store = VectorStore(backend, embedder_source(fixed_embedder()), graph=graph)
        await store.ensure_ready()
        await store.upsert(
            VectorKind.EDGE,
            [
                EmbedItem(
                    "f1", "first", EdgeExtras("s1", "o1", Validity("active", 0, 0))
                ),
                EmbedItem(
                    "f2", "second", EdgeExtras("s2", "o2", Validity("active", 0, 0))
                ),
            ],
        )

        historical = await store.search(
            VectorKind.EDGE,
            A,
            k=10,
            filters=SearchFilters(temporal=TemporalFilter(known_at_us=5)),
        )
        graph.ids = {"f1"}
        kept, missing = await store.hydrate(
            VectorKind.EDGE, [*historical, Hit("f2", 0.5, 1)]
        )

        assert [hit.id for hit in historical] == ["f1"]
        assert [hit.id for hit in kept] == ["f1"]
        assert missing == ["f2"]

    @pytest.mark.parametrize(
        "invalid_vector",
        [[0.0, 0.0, 0.0], [float("nan"), 0.0, 0.0]],
        ids=["zero_norm", "non_finite"],
    )
    async def test_startup_rejects_an_invalid_query_purpose_probe(
        self, invalid_vector: list[float]
    ) -> None:
        """A valid document probe cannot hide an invalid query probe."""
        store = VectorStore(
            InMemoryBackend(),
            embedder_source(InvalidQueryProbeEmbedder(invalid_vector)),
        )

        with pytest.raises(SpaceMismatchError, match="query"):
            await store.ensure_ready()

    async def test_dimension_mismatch_requires_compatible_vector_indexes(self) -> None:
        """Dimension changes require re-ingestion and compatible indexes."""
        backend = InMemoryBackend()
        stored = VectorStore(backend, embedder_source(MockEmbedder(dim=3)))
        await stored.ensure_ready()
        configured = VectorStore(backend, embedder_source(MockEmbedder(dim=4)))

        with pytest.raises(SpaceMismatchError) as error:
            await configured.ensure_ready()

        assert str(error.value) == (
            "node: Existing memory uses a different embedding dimension. "
            "Changing the embedding dimension requires re-embedding the source "
            "data into a new Tenant with compatible vector indexes. Shared-index "
            "backends may require a separate database or instance. Automatic "
            "re-embedding is not supported."
        )

    async def test_failed_recheck_forces_lazy_reinitialization_before_search(self):
        """A failed explicit recheck must not leave the store on a bad index."""
        backend = FlakyIndexBackend()
        store = VectorStore(backend, embedder_source(fixed_embedder()))

        await store.ensure_ready()
        await store.upsert(VectorKind.NODE, [EmbedItem("n1", "a")])
        backend.index_operational = False
        backend.fail_recheck = True
        with pytest.raises(IndexDriftError):
            await store.ensure_ready()

        backend.fail_recheck = False
        hits = await store.search(VectorKind.NODE, A, k=5)

        assert [hit.id for hit in hits] == ["n1"]
        assert backend.index_operational is True
