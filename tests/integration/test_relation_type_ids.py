"""End-to-end: dense relation ids map to stable ids across catalog changes."""

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
from ontogr_mem.graphstore.ontology_store import GraphOntologyStore
from ontogr_mem.ontology.extraction_items import stabilize_facts
from ontogr_mem.ontology.manager import OntologyManager
from ontogr_mem.ontology.models import (
    AddRelation,
    DeprecateType,
)
from ontogr_mem.ontology.models import (
    Kind as OntologyKind,
)
from ontogr_mem.ontology.ontology_store import OntologyStore
from tests.support.clock import FixedClock
from tests.support.graph_backend import GraphBackend
from tests.support.ontology import base, cs, load_catalog


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
        return OntologyManager(self.store, FixedClock())


@pytest.fixture
async def world(graph_backend: GraphBackend) -> AsyncGenerator[World]:
    """An ontology store on each real backend."""
    handle = await graph_backend.open_tenant()
    yield World(
        GraphOntologyStore(
            graph_backend.store, handle, tenant_keyed=graph_backend.tenant_keyed
        )
    )


class TestRelationTypeIds:
    """Stable relation ids and the catalog version carried by typed facts."""

    async def test_dense_ids_map_to_stable_ids_across_deprecation(self, world):
        """A dense id is mapped through the projection that issued it."""
        manager = world.manager()
        await load_catalog(manager, base())
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
        manager = world.manager()
        await load_catalog(manager, base())
        projection = await manager.extraction_view()

        result = await stabilize_facts(manager, [fact()], projection)

        assert result.typed[0].catalog_version == str(projection.catalog_version)
