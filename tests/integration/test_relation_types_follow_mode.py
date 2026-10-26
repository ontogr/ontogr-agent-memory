"""End-to-end: fact relations follow the ontology mode on every store."""

from collections.abc import AsyncGenerator
from dataclasses import dataclass

import pytest

from ontogr_mem.data_models import Method
from ontogr_mem.data_models.facts import (
    ExtractedFact,
    FactKind,
    Modality,
    Polarity,
    TimeBasis,
)
from ontogr_mem.graphstore.memory import MemoryOntologyStore
from ontogr_mem.graphstore.ontology_store import GraphOntologyStore
from ontogr_mem.graphstore.schemas import ONTOLOGY
from ontogr_mem.ontology.extraction_items import stabilize_facts
from ontogr_mem.ontology.manager import OntologyManager
from ontogr_mem.ontology.models import (
    AddRelation,
    DeprecateType,
    Mode,
    Origin,
    SetModes,
    UnmatchedKind,
)
from ontogr_mem.ontology.models import (
    Kind as OntologyKind,
)
from ontogr_mem.ontology.models import (
    Status as TypeStatus,
)
from ontogr_mem.ontology.ontology_store import OntologyStore
from ontogr_mem.ontology.settings import OntologySettings
from tests.support.clock import FixedClock
from tests.support.llm import ScriptedLLM
from tests.support.ontology import base, cs
from tests.support.tenants import new_tenant


def fact(
    *,
    fact_id: str = "f1",
    relation_type_id: int | None = None,
    relation_label: str | None = None,
    episode_id: str = "ep1",
) -> ExtractedFact:
    """Build one extracted fact for stabilization."""
    text = "Ana works at Acme"
    return ExtractedFact(
        fact_id=fact_id,
        text=text,
        entity_refs=("e0", "e1"),
        relation_type_id=relation_type_id,
        relation_label=relation_label,
        fact_kind=FactKind.STATE,
        polarity=Polarity.AFFIRMED,
        modality=Modality.ACTUAL,
        asserted_by="u1",
        valid_from=None,
        valid_to=None,
        precision=None,
        time_expression=None,
        time_basis=TimeBasis.NONE,
        evidence_quote=text,
        evidence_span=(0, len(text)),
        confidence=0.9,
        method=Method.LLM,
        episode_id=episode_id,
    )


@dataclass
class World:
    """A manager on one store."""

    store: OntologyStore

    def manager(self) -> OntologyManager:
        """Build a manager over this store."""
        return OntologyManager(
            self.store,
            ScriptedLLM([]),
            OntologySettings(min_support=3),
            FixedClock(),
        )


@pytest.fixture(params=["memory", "falkordb", "neo4j"])
async def world(request, falkordb_store, neo4j_store) -> AsyncGenerator[World]:
    """An ontology store on the twin and on each real backend."""
    if request.param == "memory":
        yield World(MemoryOntologyStore(set[str]))
        return
    store = falkordb_store if request.param == "falkordb" else neo4j_store
    handle = await store.open(new_tenant(), [ONTOLOGY])
    keyed = request.param == "neo4j"
    try:
        yield World(GraphOntologyStore(store, handle, tenant_keyed=keyed))
    finally:
        await store.erase_tenant(handle.tenant)


async def manager_with_mode(world: World, mode: Mode) -> OntologyManager:
    """Load the base catalog and set the relation mode."""
    manager = world.manager()
    await manager.load_catalog(base())
    await manager.apply(cs(await manager.catalog(), SetModes(relation_mode=mode)))
    return manager


class TestRelationTypesFollowMode:
    """An unknown relation label is handled by the ontology mode."""

    async def test_strict_drops_and_records_nothing(self, world):
        """The fact is dropped and counted; nothing is stored or registered."""
        manager = await manager_with_mode(world, Mode.STRICT)
        before = await manager.catalog()
        view = await manager.extraction_view()
        label_fact = fact(relation_label="mentors")

        result = await stabilize_facts(manager, [label_fact], view)

        assert result.typed == ()
        assert result.dropped == (label_fact,)
        assert result.counts == {"unknown_relation": 1}
        assert await manager.unmatched() == []
        assert await manager.catalog() == before

    async def test_preferred_keeps_the_fact_untyped_and_replaces_rows(self, world):
        """The label is recorded per episode; a rerun replaces the episode's rows."""
        manager = await manager_with_mode(world, Mode.PREFERRED)
        view = await manager.extraction_view()

        first = await stabilize_facts(
            manager, [fact(fact_id="f1", relation_label="mentors")], view
        )

        assert first.dropped == ()
        (typed,) = first.typed
        assert typed.relation_type is None
        assert [
            (row.kind, row.raw_label, row.episode_id)
            for row in await manager.unmatched()
        ] == [(UnmatchedKind.RELATION_TYPE, "mentors", "ep1")]

        await stabilize_facts(
            manager, [fact(fact_id="f2", relation_label="guides")], view
        )

        assert [row.raw_label for row in await manager.unmatched()] == ["guides"]

    async def test_open_registers_the_label_and_reuses_it(self, world):
        """The first label registers a type; a later fact reuses its id."""
        manager = await manager_with_mode(world, Mode.OPEN)
        view = await manager.extraction_view()

        first = await stabilize_facts(
            manager, [fact(fact_id="f1", relation_label="mentors")], view
        )

        (typed,) = first.typed
        catalog = await manager.catalog()
        (mentors,) = [r for r in catalog.relations if r.label == "mentors"]
        assert typed.relation_type == mentors.type_id
        assert mentors.origin is Origin.AUTO_ACCEPTED
        assert mentors.status is TypeStatus.ACTIVE
        assert mentors.patterns == []

        second_view = await manager.extraction_view()
        second = await stabilize_facts(
            manager, [fact(fact_id="f2", relation_label="mentors")], second_view
        )

        assert second.typed[0].relation_type == mentors.type_id
        labels = [r.label for r in (await manager.catalog()).relations]
        assert labels.count("mentors") == 1

    async def test_dense_ids_map_to_stable_ids_across_deprecation(self, world):
        """A dense id is mapped through the projection that issued it."""
        manager = world.manager()
        await manager.load_catalog(base())
        await manager.apply(cs(await manager.catalog(), AddRelation(label="uses")))
        projection = await manager.extraction_view()
        assert projection.relation_ids == (0, 1)

        first = fact(fact_id="f0", relation_type_id=0)
        second = fact(fact_id="f1", relation_type_id=1)
        result = await stabilize_facts(manager, [first, second], projection)

        assert [typed.relation_type for typed in result.typed] == [0, 1]
        assert result.dropped == ()

        await manager.apply(
            cs(
                await manager.catalog(),
                DeprecateType(kind=OntologyKind.RELATION, target=0),
            )
        )
        after = await manager.extraction_view()
        assert after.relation_ids == (1,)

        moved = fact(fact_id="f2", relation_type_id=0)
        moved_result = await stabilize_facts(manager, [moved], after)

        assert moved_result.typed[0].relation_type == 1
        # Facts stabilized earlier keep the stable id they were given.
        assert result.typed[0].relation_type == 0
        assert result.typed[1].relation_type == 1

    async def test_typed_fact_records_the_projection_catalog_version(self, world):
        """A typed fact names the catalog version its dense ids came from."""
        manager = await manager_with_mode(world, Mode.PREFERRED)
        projection = await manager.extraction_view()

        result = await stabilize_facts(manager, [fact()], projection)

        assert result.typed[0].catalog_version == str(projection.catalog_version)
