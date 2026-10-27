"""End-to-end: entity extraction follows the ontology mode on every store."""

import json
import re
from collections.abc import AsyncGenerator
from dataclasses import dataclass

import pytest

from ontogr_mem.data_models import Episode
from ontogr_mem.entity_extraction import EntityExtraction, EntityExtractionSettings
from ontogr_mem.entity_extraction.extractors import LLMExtractor, RecordMapper
from ontogr_mem.graphstore.ontology_store import GraphOntologyStore
from ontogr_mem.ontology.extraction_items import items_from_entities
from ontogr_mem.ontology.manager import OntologyManager
from ontogr_mem.ontology.models import UNKNOWN_ID, Mode, Reason, SetModes
from ontogr_mem.ontology.ontology_store import OntologyStore
from ontogr_mem.ontology.settings import OntologySettings
from ontogr_mem.ontology.validate import Context, ValidationReport
from tests.support.clock import FixedClock
from tests.support.extraction import gate, text_episode
from tests.support.graph_backend import GraphBackend
from tests.support.llm import ScriptedLLM
from tests.support.ontology import base, cs


DRAFT = json.dumps(
    {"ops": [{"op": "add_entity", "label": "Gadget", "description": "A device."}]}
)
TYPES_BLOCK = re.compile(r"<entity_types>\n(.*?)\n</entity_types>", re.DOTALL)


def answer_with_zorbit(request) -> dict:
    """Reply as a model would: type Zorbit as Gadget when the catalog lists it."""
    listed = json.loads(TYPES_BLOCK.search(request.prompt).group(1))
    gadget = next((t["id"] for t in listed if t["label"] == "Gadget"), None)
    entity = {"name": "Zorbit", "confidence": 0.95}
    if gadget is None:
        entity |= {"type_id": -1, "type_label": "Gadget"}
    else:
        entity["type_id"] = gadget
    return {"entities": [entity]}


@dataclass
class World:
    """A manager on one store, and the extraction that reads its catalog."""

    store: OntologyStore
    draft_llm: ScriptedLLM

    def manager(self) -> OntologyManager:
        """Build a manager that needs three unmatched rows to propose."""
        return OntologyManager(
            self.store, self.draft_llm, OntologySettings(min_support=3), FixedClock()
        )


@pytest.fixture
async def world(graph_backend: GraphBackend) -> AsyncGenerator[World]:
    """An ontology store on each real backend."""
    handle = await graph_backend.open_tenant()
    yield World(
        GraphOntologyStore(
            graph_backend.store, handle, tenant_keyed=graph_backend.tenant_keyed
        ),
        ScriptedLLM([DRAFT]),
    )


async def extract_and_validate(
    manager: OntologyManager, episode: Episode
) -> ValidationReport:
    """Extract an episode against the current catalog and validate the result."""
    projection = await manager.extraction_view()
    extraction = EntityExtraction(
        [LLMExtractor(gate(ScriptedLLM(answer=answer_with_zorbit)))],
        RecordMapper({}),
        projection.view.catalog,
        settings=EntityExtractionSettings(max_previous=0),
    )
    report = await extraction.extract([episode])
    items = items_from_entities(report.results[0].entities, projection.view)
    return await manager.process_extraction(items, Context(episode.id))


def episodes(count: int) -> list[Episode]:
    """Build episodes that all mention Zorbit."""
    return [
        text_episode(f"Ana bought a Zorbit, number {n}.", f"ep{n}", minutes=n)
        for n in range(count)
    ]


def zorbit(report: ValidationReport):
    """Return the verdict for the Zorbit entity."""
    return next(v for v in report.entities if v.local_id == "e0")


class TestEntityTypesFollowMode:
    """An unknown type name is handled by the ontology mode."""

    async def test_strict_drops_and_records_nothing(self, world):
        """The entity is dropped and counted, and nothing is stored or registered."""
        manager = world.manager()
        await manager.load_catalog(base())
        await manager.apply(
            cs(await manager.catalog(), SetModes(entity_mode=Mode.STRICT))
        )
        before = await manager.catalog()

        report = await extract_and_validate(manager, episodes(1)[0])

        assert not zorbit(report).kept
        assert report.counters[Reason.UNKNOWN_TYPE] == 1
        assert await manager.unmatched() == []
        assert await manager.catalog() == before

    async def test_preferred_records_a_row_per_episode_then_types_after_acceptance(
        self, world
    ):
        """Three episodes support a proposal; accepting it types the entity."""
        manager = world.manager()
        await manager.load_catalog(base())
        eps = episodes(3)

        for episode in eps:
            report = await extract_and_validate(manager, episode)
            assert zorbit(report).type_id == UNKNOWN_ID
        assert len(await manager.unmatched()) == 3

        (proposal,) = (await manager.propose()).proposals
        await manager.decide(
            proposal.proposal_id, accept=True, actor="ana", reason="ok"
        )
        gadget = next(
            e for e in (await manager.catalog()).entities if e.label == "Gadget"
        )
        reports = [await extract_and_validate(manager, e) for e in eps]

        assert [zorbit(r).type_id for r in reports] == [gadget.type_id] * 3
        assert await manager.unmatched() == []

    async def test_open_registers_at_once_and_keeps_the_id_stable(self, world):
        """The first mention registers the type; later mentions reuse its id."""
        manager = world.manager()
        await manager.load_catalog(base())
        await manager.apply(
            cs(await manager.catalog(), SetModes(entity_mode=Mode.OPEN))
        )
        first, second = episodes(2)

        first_report = await extract_and_validate(manager, first)
        second_report = await extract_and_validate(manager, second)

        labels = [e.label for e in (await manager.catalog()).entities]
        assert labels.count("Gadget") == 1
        assert zorbit(first_report).type_id is not None
        assert zorbit(first_report).type_id == zorbit(second_report).type_id
