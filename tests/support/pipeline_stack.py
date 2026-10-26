"""Build the temporal extraction stack used by integration scenarios."""

import re
from dataclasses import dataclass
from datetime import UTC, datetime

from ontogr_mem.data_models import Episode
from ontogr_mem.data_models.extraction import ExtractedEntity, Method
from ontogr_mem.fact_extraction import (
    FactExtraction,
    InMemoryFactStore,
    LLMFactExtractor,
)
from ontogr_mem.graphstore.memory import MemoryOntologyStore
from ontogr_mem.graphstore.resolver_store import GraphResolverStore
from ontogr_mem.graphstore.temporal.store import GraphTemporalStore
from ontogr_mem.ingestion.store import EpisodeStore
from ontogr_mem.ontology.extraction_items import stabilize_facts
from ontogr_mem.ontology.manager import OntologyManager
from ontogr_mem.ontology.models import AddRelation, PatternSpec
from ontogr_mem.ontology.settings import OntologySettings
from ontogr_mem.resolution import EntityResolver, ResolutionPolicy
from ontogr_mem.resolution.config import ResolverSettings
from ontogr_mem.resolution.policy import ResolutionMode, TypePolicy
from ontogr_mem.temporal import TemporalEngine, from_type_view
from ontogr_mem.vectorstore import EmbeddingGateway, VectorStore
from ontogr_mem.vectorstore.memory import InMemoryBackend
from tests.support.clock import FixedClock
from tests.support.embedder import MockEmbedder
from tests.support.episodes import graph_harness
from tests.support.extraction import gate
from tests.support.facts import fact, reply
from tests.support.llm import ScriptedLLM
from tests.support.ontology import base, cs


TEMPORAL_TEXTS = {
    "ep-2024": "Ana works at Acme since March 2024.",
    "ep-2025": "Ana works at Globex since January 2025.",
    "ep-employs": "Acme employs Ana since January 2026.",
}
NOW = datetime(2026, 10, 22, tzinfo=UTC)
EPISODE_TEXT = re.compile(r"<episode_text>\n(.*?)\n</episode_text>", re.DOTALL)


def relation_id(view, label: str) -> int | None:
    """Return the dense id of a relation label in the view, if it is offered."""
    relation = view.catalog.relation_by_label(label)
    return None if relation is None else relation.id


def temporal_replies(view) -> dict[str, str]:
    """Return the scripted fact reply for each episode text in the view."""
    works = relation_id(view, "works_at")
    employs = relation_id(view, "employs")
    replies: dict[str, str] = {}
    if works is not None:
        replies[TEMPORAL_TEXTS["ep-2024"]] = reply(
            fact(
                text="Ana works at Acme",
                fact_kind="state",
                entity_refs=["e0", "e1"],
                relation_type_id=works,
                evidence_quote="Ana works at Acme",
                valid_from="2024-03",
                precision="month",
                time_expression="March 2024",
            )
        )
        replies[TEMPORAL_TEXTS["ep-2025"]] = reply(
            fact(
                text="Ana works at Globex",
                fact_kind="state",
                entity_refs=["e0", "e1"],
                relation_type_id=works,
                evidence_quote="Ana works at Globex",
                valid_from="2025-01",
                precision="month",
                time_expression="January 2025",
            )
        )
    if employs is not None:
        replies[TEMPORAL_TEXTS["ep-employs"]] = reply(
            fact(
                text="Acme employs Ana",
                fact_kind="state",
                entity_refs=["e0", "e1"],
                relation_type_id=employs,
                evidence_quote="Acme employs Ana since January 2026",
                valid_from="2026-01",
                precision="month",
                time_expression="January 2026",
            )
        )
    return replies


def scripted_facts(replies: dict[str, str]):
    """Answer each fact request with the reply for its episode text."""

    def answer(request) -> str:
        text = EPISODE_TEXT.search(request.prompt).group(1)
        return replies[text]

    return answer


def entity(local_id: str, name: str, type_id: int, label: str, episode_id: str):
    """Build an entity as extraction hands it to resolution."""
    return ExtractedEntity(
        local_id=local_id,
        name=name,
        type_id=type_id,
        type_label=label,
        confidence=0.9,
        method=Method.LLM,
        episode_id=episode_id,
    )


@dataclass
class TemporalStack:
    """Every component of the temporal path, wired on one tenant."""

    store: object
    handle: object
    keyed: bool
    temporal: GraphTemporalStore
    episodes: EpisodeStore
    resolver: EntityResolver
    manager: OntologyManager

    async def ingest(self, episode: Episode, entities: list[ExtractedEntity]):
        """Run one episode through extraction, resolution and the engine."""
        await self.episodes.write_episodes([episode])
        await self.resolver.resolve(entities)
        view = await self.manager.extraction_view()
        llm = ScriptedLLM(answer=scripted_facts(temporal_replies(view.view)))
        extraction = FactExtraction(
            LLMFactExtractor(gate(llm)), InMemoryFactStore(), view.view
        )
        report = await extraction.extract([episode], {episode.id: entities})
        stabilized = await stabilize_facts(self.manager, report.results[0].facts, view)
        engine = await self.engine()
        return await engine.ingest(stabilized.typed, episode)

    async def engine(self) -> TemporalEngine:
        """Build a temporal engine over the tenant with the current relation rules."""
        view = await self.manager.extraction_view()
        exclusive = [
            dense
            for dense in (relation_id(view.view, "works_at"),)
            if dense is not None
        ]
        return TemporalEngine(
            self.temporal,
            from_type_view(view.view, exclusive, view.stable_relation_id),
            self.resolver.store,
            FixedClock().now,
        )

    async def ref(self, episode_id: str, local_id: str) -> str:
        """Return the stable entity id that a local ref resolved to."""
        found = await self.resolver.store.entity_id_for_ref(episode_id, local_id)
        assert found is not None
        return found


async def temporal_stack(store, handle, keyed: bool) -> TemporalStack:
    """Wire the components on one opened tenant."""
    runner = store.runner(handle)
    vectors = VectorStore(InMemoryBackend(), EmbeddingGateway(MockEmbedder()))
    await vectors.ensure_ready()
    resolver = EntityResolver(
        GraphResolverStore(runner, tenant_keyed=keyed),
        vectors,
        ResolutionPolicy(types={"Person": TypePolicy(mode=ResolutionMode.AUTO_KEY)}),
        ResolverSettings(inline_verification=False),
        FixedClock(),
    )
    manager = OntologyManager(
        MemoryOntologyStore(set[str]),
        ScriptedLLM(),
        OntologySettings(min_support=3),
        FixedClock(),
    )
    await manager.load_catalog(base())
    await manager.apply(
        cs(
            await manager.catalog(),
            AddRelation(
                label="employs",
                patterns=[PatternSpec(src="Organization", tgt="Person")],
            ),
        )
    )
    return TemporalStack(
        store,
        handle,
        keyed,
        GraphTemporalStore(store, handle),
        graph_harness(runner, tenant_keyed=keyed).store,
        resolver,
        manager,
    )


def people_and_orgs(view) -> tuple[int, int]:
    """Return the dense ids of Person and Organization in the view."""
    person = view.catalog.entity_by_label("Person")
    org = view.catalog.entity_by_label("Organization")
    assert person is not None and org is not None
    return person.id, org.id


def ana_acme_globex(view) -> tuple[list, list]:
    """Entities of the two earlier episodes, with dense type ids from the view."""
    person, org = people_and_orgs(view)
    first = [
        entity("e0", "Ana", person, "Person", "ep-2024"),
        entity("e1", "Acme", org, "Organization", "ep-2024"),
    ]
    second = [
        entity("e0", "Ana", person, "Person", "ep-2025"),
        entity("e1", "Globex", org, "Organization", "ep-2025"),
    ]
    return first, second
