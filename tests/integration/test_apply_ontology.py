"""Tests for initializing and updating tenant ontologies."""

import pytest

from ontogr_mem.graphstore.ontology_store import GraphOntologyStore
from ontogr_mem.ontology.declaration import EntityDeclaration, OntologyDeclaration
from ontogr_mem.ontology.manager import OntologyManager
from ontogr_mem.service.errors import InvalidArgumentError
from ontogr_mem.service.ontologies import apply_ontology
from tests.support.clock import FixedClock
from tests.support.graph_backend import GraphBackend


async def new_manager(backend: GraphBackend) -> OntologyManager:
    """Build a manager for a fresh graph tenant."""
    handle = await backend.open_tenant()
    return OntologyManager(
        GraphOntologyStore(backend.store, handle, tenant_keyed=backend.tenant_keyed),
        FixedClock(),
    )


class TestApplyOntology:
    """Tenant ontology initialization follows declaration presence."""

    async def test_a_new_tenant_gets_the_default_ontology(
        self, graph_backend: GraphBackend
    ):
        """An empty tenant receives the default ontology."""
        manager = await new_manager(graph_backend)
        result = await apply_ontology(manager, None)
        assert result.version == 1
        assert result.relation_by_label("WORKS_AT") is not None

    async def test_a_tenant_with_an_ontology_keeps_it_when_none_is_given(
        self, graph_backend: GraphBackend
    ):
        """Opening without a declaration preserves a stored custom type."""
        manager = await new_manager(graph_backend)
        custom = OntologyDeclaration(entities=[EntityDeclaration(label="Widget")])
        stored = await apply_ontology(manager, custom)
        result = await apply_ontology(manager, None)
        assert result.version == stored.version
        assert [entity.label for entity in result.entities] == ["Unknown", "Widget"]

    async def test_a_custom_ontology_replaces_the_default(
        self, graph_backend: GraphBackend
    ):
        """A custom declaration is used for a new tenant."""
        manager = await new_manager(graph_backend)
        result = await apply_ontology(
            manager, OntologyDeclaration(entities=[EntityDeclaration(label="Widget")])
        )
        assert [entity.label for entity in result.entities] == ["Unknown", "Widget"]
        assert result.relation_by_label("PREFERS") is None

    async def test_a_declaration_that_does_not_apply_is_an_invalid_argument(
        self, graph_backend: GraphBackend
    ):
        """A declaration conflicting with a retired name is rejected."""
        manager = await new_manager(graph_backend)
        await apply_ontology(
            manager, OntologyDeclaration(entities=[EntityDeclaration(label="A")])
        )
        await apply_ontology(manager, OntologyDeclaration())
        with pytest.raises(InvalidArgumentError):
            await apply_ontology(
                manager,
                OntologyDeclaration(
                    entities=[EntityDeclaration(label="B", aliases=["A"])]
                ),
            )
