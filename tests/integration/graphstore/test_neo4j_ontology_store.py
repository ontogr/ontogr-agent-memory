"""The ontology store contract on Neo4j."""

import pytest

from ontogr_mem.graphstore.schemas import EPISODES, ONTOLOGY
from tests.contracts.ontology_store import OntologyStoreContract
from tests.support.ontology import graph_ontology_harness
from tests.support.tenants import new_tenant


class TestNeo4jOntologyStore(OntologyStoreContract):
    """GraphOntologyStore satisfies the ontology store contract on Neo4j."""

    @pytest.fixture
    async def make_harness(self, neo4j_store):
        """Return a factory for stores on fresh tenants, erased afterwards."""
        tenants: list[str] = []

        async def make():
            handle = await neo4j_store.open(new_tenant(), [EPISODES, ONTOLOGY])
            tenants.append(handle.tenant)
            return graph_ontology_harness(neo4j_store, handle, tenant_keyed=True)

        yield make
        for tenant in tenants:
            await neo4j_store.erase_tenant(tenant)
