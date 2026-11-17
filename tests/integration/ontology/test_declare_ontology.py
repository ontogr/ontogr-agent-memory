"""Tests for applying declarations to a live ontology store."""

import asyncio

from ontogr_mem.graphstore.ontology_store import GraphOntologyStore
from ontogr_mem.ontology.declaration import EntityDeclaration, OntologyDeclaration
from ontogr_mem.ontology.manager import OntologyManager
from tests.support.clock import FixedClock
from tests.support.graph_backend import GraphBackend


def declaration() -> OntologyDeclaration:
    """Return a small declaration used by the manager tests."""
    return OntologyDeclaration(entities=[EntityDeclaration(label="Person")])


async def store_for(backend: GraphBackend) -> GraphOntologyStore:
    """Open the ontology store of a fresh tenant."""
    handle = await backend.open_tenant()
    return GraphOntologyStore(backend.store, handle, tenant_keyed=backend.tenant_keyed)


class TestDeclareOntology:
    """Declaration updates persist through the graph ontology store."""

    async def test_the_first_declaration_creates_one_version_and_the_second_none(
        self, graph_backend: GraphBackend
    ):
        """Repeated declarations leave the ontology version unchanged."""
        manager = OntologyManager(await store_for(graph_backend), FixedClock())
        first = await manager.declare(declaration())
        second = await manager.declare(declaration())
        assert first.version == 1
        assert second.version == first.version

    async def test_two_managers_declaring_at_once_end_on_one_version(
        self, graph_backend: GraphBackend
    ):
        """Concurrent equivalent declarations converge on one version."""
        store = await store_for(graph_backend)
        first = OntologyManager(store, FixedClock())
        second = OntologyManager(store, FixedClock())
        declarations = declaration()
        await asyncio.gather(first.declare(declarations), second.declare(declarations))
        assert (await first.catalog()).version == 1

    async def test_a_changed_declaration_adds_one_version_and_keeps_ids(
        self, graph_backend: GraphBackend
    ):
        """A rename and addition retain existing type ids and record provenance."""
        manager = OntologyManager(await store_for(graph_backend), FixedClock())
        initial = await manager.declare(declaration())
        person_id = next(
            entity.type_id for entity in initial.entities if entity.label == "Person"
        )
        changed = OntologyDeclaration(
            entities=[
                EntityDeclaration(label="Human", renamed_from=["Person"]),
                EntityDeclaration(label="Organization"),
            ]
        )
        updated = await manager.declare(changed)
        assert updated.version == initial.version + 1
        assert (
            next(
                entity.type_id for entity in updated.entities if entity.label == "Human"
            )
            == person_id
        )
        persisted = await manager.catalog()
        organization_id = next(
            item.type_id for item in persisted.entities if item.label == "Organization"
        )
        assert organization_id != person_id
        history = await manager.history()
        assert history[-1].channel.value == "import"
        assert history[-1].actor == "declare"
