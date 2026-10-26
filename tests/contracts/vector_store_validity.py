"""Validity-mirror checks for backends that keep fact history in the store."""

import pytest

from ontogr_mem.vectorstore import (
    CapabilityError,
    SearchFilters,
    TemporalFilter,
    Validity,
    ValidityUpdate,
    VectorKind,
)
from tests.contracts.vector_store import edge_item, seed_edges
from tests.support.pg_graph import FakeGraph
from tests.support.vectors import A, VectorHarness, ready


EDGE = VectorKind.EDGE
ACTIVE = Validity("active", 0, 0)


def graph_of(harness: VectorHarness) -> FakeGraph:
    """Return the fake graph behind `harness`."""
    graph = getattr(harness, "graph", None)
    assert isinstance(graph, FakeGraph)
    return graph


async def seed_current_edges(
    harness: VectorHarness, ids: list[str], text: str = "a"
) -> None:
    """Seed the graph edges and write their vectors with an active validity."""
    await seed_edges(harness, ids, validity=ACTIVE)
    await harness.store.upsert(
        EDGE, [edge_item(object_id, text, validity=ACTIVE) for object_id in ids]
    )


class ValidityMirrorContract:
    """Checks only a backend with `validity_mirror=True` can pass."""

    @pytest.fixture
    def make_harness(self):
        """Return a factory for graph-backed harnesses on fresh tenants."""
        raise NotImplementedError

    async def test_update_validity_closes_only_the_mirror(self, make_harness):
        """A closed fact leaves current search and stays in history, vector intact."""
        h = await ready(make_harness)
        await seed_current_edges(h, ["f1"])
        before = await h.store.fetch_vectors(EDGE, ["f1"])

        changed = await h.store.update_validity(
            [ValidityUpdate("f1", Validity("closed", 0, 0))]
        )
        current = await h.store.search(EDGE, A, k=5)
        history = await h.store.search(
            EDGE,
            A,
            k=5,
            filters=SearchFilters(temporal=TemporalFilter(history=True)),
        )

        assert changed == 1
        assert current == []
        assert [hit.id for hit in history] == ["f1"]
        assert await h.store.fetch_vectors(EDGE, ["f1"]) == before

    async def test_supersession_hides_the_old_belief_from_current_search(
        self, make_harness
    ):
        """An expired fact is not current belief but is still history."""
        h = await ready(make_harness)
        await seed_current_edges(h, ["f1"])

        await h.store.update_validity(
            [ValidityUpdate("f1", Validity("active", 0, 0, expired_at_us=900))]
        )

        assert await h.store.search(EDGE, A, k=5) == []
        history = await h.store.search(
            EDGE,
            A,
            k=5,
            filters=SearchFilters(temporal=TemporalFilter(history=True)),
        )
        assert [hit.id for hit in history] == ["f1"]

    async def test_retraction_then_restoration_returns_the_fact(self, make_harness):
        """A withdrawn fact keeps its vector and comes back when restored."""
        h = await ready(make_harness)
        await seed_current_edges(h, ["f1"])
        before = await h.store.fetch_vectors(EDGE, ["f1"])

        await h.store.update_validity(
            [ValidityUpdate("f1", Validity("withdrawn", 0, 0))]
        )
        assert await h.store.search(EDGE, A, k=5) == []
        assert await h.store.exists(EDGE, ["f1"]) == {"f1"}

        await h.store.update_validity([ValidityUpdate("f1", ACTIVE)])

        assert [hit.id for hit in await h.store.search(EDGE, A, k=5)] == ["f1"]
        assert await h.store.fetch_vectors(EDGE, ["f1"]) == before

    async def test_rewritten_fact_gets_a_new_vector_under_a_new_id(self, make_harness):
        """A rewrite deletes the old fact id and stores the new one."""
        h = await ready(make_harness)
        await seed_current_edges(h, ["f1", "f2"])

        await h.store.delete(EDGE, ["f1"])
        await h.store.upsert(EDGE, [edge_item("f2", "half", validity=ACTIVE)])

        assert await h.store.exists(EDGE, ["f1", "f2"]) == {"f2"}
        history = await h.store.search(
            EDGE,
            A,
            k=5,
            filters=SearchFilters(temporal=TemporalFilter(history=True)),
        )
        assert [hit.id for hit in history] == ["f2"]

    async def test_known_at_in_the_past_is_decided_by_the_graph(self, make_harness):
        """The mirror says what is current now; the graph says what was believed."""
        h = await ready(make_harness)
        await seed_current_edges(h, ["f0", "f1", "f2"])
        await h.store.update_validity([ValidityUpdate("f0", Validity("closed", 0, 0))])
        graph = graph_of(h)
        graph.valid = lambda object_id, _temporal: object_id in {"f0", "f2"}

        hits = await h.store.search(
            EDGE,
            A,
            k=5,
            filters=SearchFilters(temporal=TemporalFilter(known_at_us=250)),
        )

        assert {hit.id for hit in hits} == {"f0", "f2"}
        assert graph.valid_calls

    async def test_known_at_over_fetches_and_doubles_until_enough(self, make_harness):
        """A recheck that removes the nearest hits fetches more, then stops."""
        h = await ready(make_harness)
        ids = [f"f{i}" for i in range(6)]
        texts = ["a", "a", "a", "half", "half", "half"]
        await seed_edges(h, ids, validity=ACTIVE)
        await h.store.upsert(
            EDGE,
            [
                edge_item(object_id, text, validity=ACTIVE)
                for object_id, text in zip(ids, texts, strict=True)
            ],
        )
        graph = graph_of(h)
        top3 = {"f0", "f1", "f2"}
        graph.valid = lambda object_id, _temporal: object_id not in top3

        hits = await h.store.search(
            EDGE,
            A,
            k=2,
            filters=SearchFilters(temporal=TemporalFilter(known_at_us=250)),
        )

        assert len(hits) == 2
        assert not ({hit.id for hit in hits} & top3)
        assert [len(call) for call in graph.valid_calls] == [4, 6]

    async def test_known_at_stops_at_the_recheck_cap(self, make_harness):
        """When the graph rejects every hit the search returns empty, not loops."""
        h = await ready(make_harness)
        await seed_current_edges(h, [f"f{i}" for i in range(6)])
        graph = graph_of(h)
        graph.valid = lambda _object_id, _temporal: False

        hits = await h.store.search(
            EDGE,
            A,
            k=2,
            filters=SearchFilters(temporal=TemporalFilter(known_at_us=250)),
        )

        assert hits == []
        assert len(graph.valid_calls) <= h.store.settings.max_rechecks + 1

    async def test_known_at_without_a_graph_raises(self, make_harness):
        """Past belief cannot be answered without the graph re-check."""
        h = await ready(make_harness)
        await seed_current_edges(h, ["f1"])
        h.store.graph = None

        with pytest.raises(CapabilityError):
            await h.store.search(
                EDGE,
                A,
                k=1,
                filters=SearchFilters(temporal=TemporalFilter(known_at_us=1)),
            )

    async def test_hydrate_drops_ids_the_graph_no_longer_has(self, make_harness):
        """Hydration trusts the graph, not the vector table."""
        h = await ready(make_harness)
        await seed_current_edges(h, ["f0", "f1", "f2"])
        graph_of(h).remove(EDGE, "f1")
        hits = await h.store.search(EDGE, A, k=3)

        kept, missing = await h.store.hydrate(EDGE, hits)

        assert {hit.id for hit in kept} == {"f0", "f2"}
        assert missing == ["f1"]
