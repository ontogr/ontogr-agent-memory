"""Tests for the ontology manager over a live graph tenant."""

import pytest

from ontogr_mem.graphstore.ontology_store import GraphOntologyStore
from ontogr_mem.ontology.errors import InvalidChange, VersionConflict
from ontogr_mem.ontology.manager import OntologyManager
from ontogr_mem.ontology.models import AddEntity, Ontology
from tests.support.clock import FixedClock
from tests.support.graph_backend import GraphBackend
from tests.support.ontology import base, cs


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
        await manager.load_catalog(catalog)
    return manager


class TestOntologyManager:
    """Behavior of OntologyManager."""

    async def test_load_catalog_only_into_version_zero(
        self, graph_backend: GraphBackend
    ):
        """Load catalog only into version zero."""
        manager = await make_manager(graph_backend, base())

        head = await manager.catalog()
        assert head.version == 1
        assert {e.label for e in head.entities} == {
            "Unknown",
            "Person",
            "Employee",
            "Organization",
        }
        with pytest.raises(InvalidChange):
            await manager.load_catalog(base())

    async def test_apply_with_a_stale_base_conflicts(self, graph_backend: GraphBackend):
        """Apply with a stale base conflicts."""
        manager = await make_manager(graph_backend, base())
        stale = await manager.catalog()
        await manager.apply(cs(stale, AddEntity(label="Place")))

        with pytest.raises(VersionConflict):
            await manager.apply(cs(stale, AddEntity(label="Other")))
