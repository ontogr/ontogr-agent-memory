"""The ontology store contract on FalkorDB."""

import pytest

from ontogr_mem.graphstore.schemas import EPISODES, ONTOLOGY
from tests.contracts.ontology_store import OntologyStoreContract
from tests.support.ontology import graph_ontology_harness
from tests.support.tenants import new_tenant


class TestFalkorDBOntologyStore(OntologyStoreContract):
    """GraphOntologyStore satisfies the ontology store contract on FalkorDB."""

    @pytest.fixture
    async def make_harness(self, falkordb_store):
        """Return a factory for stores on fresh tenants, erased afterwards."""
        tenants: list[str] = []

        async def make():
            handle = await falkordb_store.open(new_tenant(), [EPISODES, ONTOLOGY])
            tenants.append(handle.tenant)
            return graph_ontology_harness(falkordb_store, handle, tenant_keyed=False)

        yield make
        for tenant in tenants:
            await falkordb_store.erase_tenant(tenant)
