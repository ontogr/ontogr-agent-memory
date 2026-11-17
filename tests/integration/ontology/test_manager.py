"""Tests for the ontology manager over a live graph tenant."""

import pytest

from ontogr_mem.graphstore.ontology_store import GraphOntologyStore
from ontogr_mem.ontology.errors import VersionConflict
from ontogr_mem.ontology.manager import OntologyManager
from ontogr_mem.ontology.models import AddEntity, Ontology
from tests.support.clock import FixedClock
from tests.support.graph_backend import GraphBackend
from tests.support.ontology import base, cs, load_catalog


async def make_store(graph_backend: GraphBackend) -> GraphOntologyStore:
    """Open a fresh tenant and return its ontology store."""
    handle = await graph_backend.open_tenant()
    return GraphOntologyStore(
        graph_backend.store, handle, tenant_keyed=graph_backend.tenant_keyed
    )


async def make_manager(
    graph_backend: GraphBackend,
    catalog: Ontology | None = None,
) -> OntologyManager:
    """Build a manager on a fresh tenant, optionally loaded with a catalog."""
    manager = OntologyManager(
        await make_store(graph_backend),
        FixedClock(),
    )
    if catalog is not None:
        await load_catalog(manager, catalog)
    return manager


class TestOntologyManager:
    """Behavior of OntologyManager."""

    async def test_apply_with_a_stale_base_conflicts(self, graph_backend: GraphBackend):
        """Apply with a stale base conflicts."""
        manager = await make_manager(graph_backend, base())
        stale = await manager.catalog()
        await manager.apply(cs(stale, AddEntity(label="Place")))

        with pytest.raises(VersionConflict):
            await manager.apply(cs(stale, AddEntity(label="Other")))
