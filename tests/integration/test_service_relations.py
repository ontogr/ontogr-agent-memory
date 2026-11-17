"""Ontology-backed relation behavior for a tenant service."""

import pytest

from ontogr_mem.graphstore.ontology_store import GraphOntologyStore
from ontogr_mem.ontology.manager import OntologyManager
from ontogr_mem.ontology.models import ChangeSet, DeprecateType, Kind
from ontogr_mem.service.ontologies import apply_ontology
from ontogr_mem.service.relations import relation_policy
from tests.support.clock import FixedClock
from tests.support.graph_backend import GraphBackend


@pytest.fixture
async def manager(graph_backend: GraphBackend) -> OntologyManager:
    """Build a manager over a fresh tenant, without an LLM client."""
    handle = await graph_backend.open_tenant()
    return OntologyManager(
        GraphOntologyStore(
            graph_backend.store, handle, tenant_keyed=graph_backend.tenant_keyed
        ),
        FixedClock(),
    )


async def test_defaults_load_once_and_single_relations_follow_the_ontology(manager):
    """Defaults are idempotent and relation rules follow the current head."""
    await apply_ontology(manager, None)
    head = await manager.catalog()
    await apply_ontology(manager, None)
    assert (await manager.catalog()).version == head.version

    catalog = await manager.catalog()
    works_at = next(r for r in catalog.relations if r.label == "WORKS_AT")
    policy = relation_policy(catalog)
    for label in ("WORKS_AT", "LIVES_IN", "MARRIED_TO", "LOCATED_IN"):
        relation = next(item for item in catalog.relations if item.label == label)
        assert policy.rules(relation.type_id).cardinality.value == "one"
    for label in ("PREFERS", "USES"):
        relation = next(item for item in catalog.relations if item.label == label)
        assert policy.rules(relation.type_id).cardinality.value == "many"

    head = await manager.catalog()
    await manager.apply(
        ChangeSet(
            base_version=head.version,
            ops=[DeprecateType(kind=Kind.RELATION, target="WORKS_AT")],
        )
    )
    assert (
        relation_policy(await manager.catalog())
        .rules(works_at.type_id)
        .cardinality.value
        == "many"
    )
