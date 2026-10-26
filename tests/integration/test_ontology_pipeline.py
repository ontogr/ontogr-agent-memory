"""End-to-end: the ontology manager on each graph backend."""

import asyncio
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass

import pytest

from ontogr_mem.graphstore.ontology_store import GraphOntologyStore
from ontogr_mem.graphstore.protocol import GraphStore
from ontogr_mem.graphstore.schemas import EPISODES, ONTOLOGY
from ontogr_mem.graphstore.types import GraphHandle
from ontogr_mem.ingestion import IngestionPipeline, default_registry
from ontogr_mem.ontology.manager import OntologyManager
from ontogr_mem.ontology.models import Mode, SetModes
from ontogr_mem.ontology.settings import OntologySettings
from ontogr_mem.ontology.validate import Context, EntityItem, ExtractedItems
from tests.integration.test_pipeline import CHAT, DOCUMENT
from tests.support.clock import FixedClock
from tests.support.episodes import graph_harness
from tests.support.llm import ScriptedLLM
from tests.support.ontology import base, cs
from tests.support.tenants import new_tenant


GOOD = json.dumps(
    {"ops": [{"op": "add_entity", "label": "Gadget", "description": "A device."}]}
)


@dataclass
class Backend:
    """A graph backend with one opened tenant."""

    store: GraphStore
    handle: GraphHandle
    keyed: bool

    def manager(self, llm: ScriptedLLM | None = None) -> OntologyManager:
        """Build a manager on the tenant's graph."""
        return OntologyManager(
            GraphOntologyStore(self.store, self.handle, tenant_keyed=self.keyed),
            llm or ScriptedLLM(),
            OntologySettings(min_support=2),
            FixedClock(),
        )


@pytest.fixture(params=["falkordb", "neo4j"])
async def backend(request, falkordb_store, neo4j_store) -> AsyncGenerator[Backend]:
    """An opened tenant on each backend, erased afterwards."""
    store = falkordb_store if request.param == "falkordb" else neo4j_store
    handle = await store.open(new_tenant(), [EPISODES, ONTOLOGY])
    yield Backend(store, handle, request.param == "neo4j")
    await store.erase_tenant(handle.tenant)


class TestOntologyPipeline:
    """The ontology manager works on both graph backends."""

    async def test_ingest_extract_draft_decide_and_register(self, backend):
        """Episodes, unmatched rows, a decided proposal and a registration."""
        runner = backend.store.runner(backend.handle)
        episodes = graph_harness(runner, tenant_keyed=backend.keyed).store
        pipeline = IngestionPipeline(episodes, default_registry(), FixedClock())
        await pipeline.ingest("message", CHAT)
        await pipeline.ingest("document", [DOCUMENT])
        manager = backend.manager(ScriptedLLM([GOOD]))
        await manager.load_catalog(base())

        first, second = sorted(
            await GraphOntologyStore(
                backend.store, backend.handle, tenant_keyed=backend.keyed
            ).known_episode_ids()
        )[:2]
        for episode_id in (first, second):
            await manager.process_extraction(
                ExtractedItems(entities=(EntityItem("a", raw_type="Gadget"),)),
                Context(episode_id),
            )
        (proposal,) = (await manager.propose()).proposals
        await manager.decide(
            proposal.proposal_id, accept=True, actor="ana", reason="ok"
        )
        await manager.apply(
            cs(await manager.catalog(), SetModes(entity_mode=Mode.OPEN))
        )
        report = await manager.process_extraction(
            ExtractedItems(entities=(EntityItem("a", raw_type="Kestrel"),)),
            Context(first),
        )

        assert report.entities[0].kept and report.entities[0].type_id is not None
        assert [c.channel.value for c in await manager.history()] == [
            "import",
            "proposal",
            "user",
            "open_mode",
        ]
        labels = {entity.label for entity in (await manager.catalog()).entities}
        assert {"Gadget", "Kestrel"} <= labels

    async def test_two_managers_racing_both_apply(self, backend):
        """Both writers end applied; the one that lost retried on the new head."""
        first, second = backend.manager(), backend.manager()
        await first.load_catalog(base())
        await first.apply(cs(await first.catalog(), SetModes(entity_mode=Mode.OPEN)))

        async def register(manager: OntologyManager, label: str) -> None:
            """Register."""
            await manager.process_extraction(
                ExtractedItems(entities=(EntityItem("a", raw_type=label),)),
                Context(label),
            )

        await asyncio.gather(register(first, "Kestrel"), register(second, "Tundra"))

        labels = [e.label for e in (await first.catalog()).entities]
        assert {"Kestrel", "Tundra"} <= set(labels)
        assert len(labels) == len(set(labels))
        history = await first.history()
        assert [c.base_version for c in history] == list(range(len(history)))

    async def test_many_concurrent_registrations_lose_nothing(self, backend):
        """Twenty extractions with one shared label register every label once."""
        manager = backend.manager()
        await manager.load_catalog(base())
        await manager.apply(
            cs(await manager.catalog(), SetModes(entity_mode=Mode.OPEN))
        )
        words = [
            "Alpha", "Bravo", "Cobalt", "Dune", "Ember", "Fjord", "Garnet", "Harbor",
            "Iris", "Jasper", "Kestrel", "Lumen", "Meadow", "Nimbus", "Onyx",
            "Pebble", "Quartz", "Raven", "Summit", "Tundra",
        ]  # fmt: skip

        async def extract(i: int) -> None:
            """Extract."""
            items = ExtractedItems(
                entities=(
                    EntityItem("a", raw_type=words[i]),
                    EntityItem("b", raw_type="Shared"),
                )
            )
            await manager.process_extraction(items, Context(f"ep{i}"))

        await asyncio.gather(*(extract(i) for i in range(20)))

        labels = [e.label for e in (await manager.catalog()).entities]
        assert set(words) | {"Shared"} <= set(labels)
        assert len(labels) == len(set(labels))
        history = await manager.history()
        assert [c.base_version for c in history] == list(range(len(history)))
