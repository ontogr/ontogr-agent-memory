"""Ontology-backed relation behavior for a tenant service."""

import pytest

from ontogr_mem.graphstore.ontology_store import GraphOntologyStore
from ontogr_mem.ontology.manager import OntologyManager
from ontogr_mem.ontology.models import (
    AddAlias,
    ChangeSet,
    DeprecateType,
    Kind,
    Mode,
    SetModes,
)
from ontogr_mem.ontology.settings import OntologySettings
from ontogr_mem.service.errors import InvalidArgumentError
from ontogr_mem.service.relations import (
    ensure_relation,
    load_defaults,
    relation_policy,
)
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
        None,
        OntologySettings(),
        FixedClock(),
    )


async def test_defaults_load_once_and_relation_updates_follow_ontology(manager):
    """Defaults are idempotent and relation rules follow the current head."""
    await load_defaults(manager)
    head = await manager.catalog()
    await load_defaults(manager)
    assert (await manager.catalog()).version == head.version

    relation = await ensure_relation(manager, "LIVES IN", single_valued=True)
    same = await ensure_relation(manager, "lives_in", single_valued=True)
    assert relation.type_id == same.type_id

    head = await manager.catalog()
    await manager.apply(
        ChangeSet(
            base_version=head.version,
            ops=[SetModes(relation_mode=Mode.PREFERRED)],
        )
    )
    mentor = await ensure_relation(manager, "Mentors")
    assert mentor.label == "Mentors"
    assert (
        relation_policy(await manager.catalog()).rules(mentor.type_id).cardinality.value
        == "many"
    )

    head = await manager.catalog()
    await manager.apply(
        ChangeSet(
            base_version=head.version,
            ops=[SetModes(relation_mode=Mode.STRICT)],
        )
    )
    with pytest.raises(InvalidArgumentError, match="Mentor Unknown"):
        await ensure_relation(manager, "Mentor Unknown")

    works_at = await ensure_relation(manager, "WORKS_AT", single_valued=False)
    assert (
        relation_policy(await manager.catalog())
        .rules(works_at.type_id)
        .cardinality.value
        == "many"
    )

    head = await manager.catalog()
    await manager.apply(
        ChangeSet(
            base_version=head.version,
            ops=[
                AddAlias(kind=Kind.RELATION, target="works_at", alias="employs"),
                DeprecateType(kind=Kind.RELATION, target="related_to"),
            ],
        )
    )
    assert (await ensure_relation(manager, "EMPLOYS")).type_id == works_at.type_id
    with pytest.raises(InvalidArgumentError, match="related_to"):
        await ensure_relation(manager, "related_to")
