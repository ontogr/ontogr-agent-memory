"""The alias text is stored as a property of the alias node."""

import pytest

from ontogr_mem.graphstore.resolver_store import GraphResolverStore
from ontogr_mem.graphstore.schemas import RESOLUTION
from tests.contracts.resolver_store import alias, entity
from tests.support.tenants import new_tenant


@pytest.mark.parametrize("backend", ["falkordb", "neo4j"])
async def test_alias_node_stores_the_alias_text(backend, falkordb_store, neo4j_store):
    """A written alias carries the alias text as a node property."""
    graph = neo4j_store if backend == "neo4j" else falkordb_store
    handle = await graph.open(new_tenant(), [RESOLUTION])
    try:
        store = GraphResolverStore(graph.runner(handle))
        written = alias("e1", "Acme Incorporated")

        await store.write_entity_batch([entity("e1")], [written])

        nodes = await graph.get_nodes(handle, "Alias", [written.alias_id])
        assert len(nodes) == 1
        assert nodes[0].props.get("text") == "Acme Incorporated"
    finally:
        await graph.erase_tenant(handle.tenant)
