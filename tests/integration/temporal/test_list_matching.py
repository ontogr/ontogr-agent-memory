"""Temporal statements that match long lists of ids, on both servers.

A list-valued parameter used in a match is where a server fails silently or
crashes, so the statements that take one run here with a list far longer than
a typical commit.
"""

import pytest

from ontogr_mem.graphstore.falkordb import FalkorDBGraphStore
from ontogr_mem.graphstore.neo4j import Neo4jGraphStore
from ontogr_mem.graphstore.schemas import EPISODES, RESOLUTION, TEMPORAL
from ontogr_mem.graphstore.temporal.store import GraphTemporalStore
from ontogr_mem.temporal.ledger import GENESIS
from ontogr_mem.temporal.query import TemporalQuery
from ontogr_mem.temporal.temporal_store import SlotCommit
from tests.contracts.temporal_store import (
    NOW,
    SLOT,
    ledger_for,
    observation,
    version,
)
from tests.support.temporal import TemporalHarness, graph_temporal_harness
from tests.support.tenants import new_tenant


LIST_SIZE = 60


async def _long_list_round_trip(harness: TemporalHarness) -> None:
    objects = [f"obj-{index:03d}" for index in range(LIST_SIZE)]
    await harness.seed_entities(["ana", *objects])
    await harness.seed_episode("ep-1")
    observations = [
        observation(f"obs-{index:03d}", obj=obj) for index, obj in enumerate(objects)
    ]

    inserted = await harness.store.append_observations(observations)
    again = await harness.store.append_observations(observations)
    revision = await harness.store.commit_slot(
        SlotCommit(
            slot_key=SLOT,
            base_revision=0,
            new_facts=tuple(
                version(f"fact-{index:03d}", 1, obj=obj, evidence=(f"obs-{index:03d}",))
                for index, obj in enumerate(objects)
            ),
            ledger=(ledger_for(SLOT, (0, GENESIS)),),
        )
    )

    state = await harness.store.load_slot(SLOT)
    views = await harness.store.query(TemporalQuery(now=NOW))
    observed = await harness.store.observations_of_episode("ep-1")

    assert inserted == LIST_SIZE
    assert again == 0
    assert revision == 1
    assert len(state.facts) == LIST_SIZE
    assert sorted(view.object_id for view in views) == objects
    assert len(observed) == LIST_SIZE


class TestFalkorDBListMatching:
    """Long id lists through the FalkorDB temporal statements."""

    @pytest.fixture
    async def make_harness(self, falkordb_store: FalkorDBGraphStore):
        """Return a factory of harnesses on fresh tenants, erased afterwards."""
        tenants: list[str] = []

        async def make() -> TemporalHarness:
            handle = await falkordb_store.open(
                new_tenant(), [EPISODES, RESOLUTION, TEMPORAL]
            )
            tenants.append(handle.tenant)
            return graph_temporal_harness(
                GraphTemporalStore(falkordb_store, handle),
                falkordb_store,
                handle,
                tenant_keyed=False,
            )

        yield make
        for tenant in tenants:
            await falkordb_store.erase_tenant(tenant)

    async def test_long_id_lists_round_trip(self, make_harness):
        """Entity gate, observation lookup and facts of a 60-entity commit."""
        await _long_list_round_trip(await make_harness())


class TestNeo4jListMatching:
    """Long id lists through the Neo4j temporal statements."""

    @pytest.fixture
    async def make_harness(self, neo4j_store: Neo4jGraphStore):
        """Return a factory of harnesses on fresh tenants, erased afterwards."""
        tenants: list[str] = []

        async def make() -> TemporalHarness:
            handle = await neo4j_store.open(
                new_tenant(), [EPISODES, RESOLUTION, TEMPORAL]
            )
            tenants.append(handle.tenant)
            return graph_temporal_harness(
                GraphTemporalStore(neo4j_store, handle),
                neo4j_store,
                handle,
                tenant_keyed=True,
            )

        yield make
        for tenant in tenants:
            await neo4j_store.erase_tenant(tenant)

    async def test_long_id_lists_round_trip(self, make_harness):
        """Entity gate, observation lookup and facts of a 60-entity commit."""
        await _long_list_round_trip(await make_harness())
