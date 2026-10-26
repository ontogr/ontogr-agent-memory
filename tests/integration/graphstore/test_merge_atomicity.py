"""Competing resolver writers on the same tenant graph."""

import asyncio

import pytest

from ontogr_mem.data_models.entities import EntityStatus
from ontogr_mem.graphstore.resolver_store import GraphResolverStore
from ontogr_mem.graphstore.schemas import RESOLUTION
from ontogr_mem.resolution.errors import MergeRefusedError
from tests.contracts.resolver_store import entity, merge_plan
from tests.support.tenants import new_tenant


@pytest.mark.parametrize("backend", ["falkordb", "neo4j"])
async def test_independent_writers_cannot_absorb_the_same_entity_twice(
    backend, falkordb_store, neo4j_store
):
    """Database guards refuse a competing merge without creating its record."""
    graph = neo4j_store if backend == "neo4j" else falkordb_store
    handle = await graph.open(new_tenant(), [RESOLUTION])
    try:
        first = GraphResolverStore(graph.runner(handle))
        second = GraphResolverStore(graph.runner(handle))
        survivor = entity("survivor")
        absorbed = entity("absorbed")
        await first.write_entity_batch([survivor, absorbed], [])
        plans = [
            merge_plan("first", survivor, absorbed),
            merge_plan("second", survivor, absorbed),
        ]

        results = await asyncio.gather(
            first.apply_merge(plans[0]),
            second.apply_merge(plans[1]),
            return_exceptions=True,
        )

        assert sum(not isinstance(result, BaseException) for result in results) == 1
        assert sum(isinstance(result, MergeRefusedError) for result in results) == 1
        records = await asyncio.gather(
            first.merge_record("first"), first.merge_record("second")
        )
        assert sum(record is not None for record in records) == 1
        stored = await first.get_entities(["absorbed"])
        assert stored[0].status is EntityStatus.MERGED
        assert stored[0].merged_into == "survivor"
    finally:
        await graph.erase_tenant(handle.tenant)
