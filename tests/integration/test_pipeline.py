"""End-to-end: ingestion, extraction, temporal and consolidation on each store."""

import re
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta

import pytest

from ontogr_mem.consolidation import (
    ConsolidationRunner,
    ConsolidationSettings,
    RunStatus,
    StageStatus,
    default_stages,
)
from ontogr_mem.consolidation.config import PruneSettings
from ontogr_mem.consolidation.models import Proposal, ProposalStatus, Score
from ontogr_mem.consolidation.signals import (
    record_access,
    record_feedback,
    record_session_entry,
)
from ontogr_mem.data_models import Episode, Kind, Role
from ontogr_mem.entity_extraction import (
    EntityExtraction,
    EntityExtractionSettings,
    ExtractionReport,
)
from ontogr_mem.entity_extraction.extractors import LLMExtractor, RecordMapper
from ontogr_mem.graphstore.cons_store import GraphConsStore
from ontogr_mem.graphstore.memory import MemoryEpisodeStore
from ontogr_mem.graphstore.schemas import (
    CONSOLIDATION,
    EPISODES,
    RESOLUTION,
    TEMPORAL,
)
from ontogr_mem.ingestion import (
    ChatMessage,
    DocumentInput,
    EpisodeStore,
    IngestionPipeline,
    default_registry,
)
from ontogr_mem.ontology.models import DeprecateType
from ontogr_mem.ontology.models import Kind as OntologyKind
from ontogr_mem.temporal.ledger import TENANT_CHAIN, slot_chain
from ontogr_mem.temporal.query import TemporalQuery
from tests.support.clock import FixedClock
from tests.support.episodes import T0, graph_harness
from tests.support.episodes import make_episode as episode_at
from tests.support.extraction import catalog, gate
from tests.support.llm import ScriptedLLM
from tests.support.ontology import cs
from tests.support.pipeline_stack import (
    NOW,
    TEMPORAL_TEXTS,
    ana_acme_globex,
    entity,
    people_and_orgs,
    relation_id,
    temporal_stack,
)
from tests.support.tenants import new_tenant


CHAT = [
    ChatMessage(
        f"m{i}",
        "s1",
        "u1" if i % 2 == 0 else "bot",
        Role.USER if i % 2 == 0 else Role.ASSISTANT,
        text,
        T0 + timedelta(minutes=i),
        "Ana" if i % 2 == 0 else "",
    )
    for i, text in enumerate(
        [
            "I moved to Berlin last week.",
            "Welcome to Berlin! Which district?",
            "Kreuzberg. I work at Acme now.",
            "Great, Acme is close to there.",
        ]
    )
]
DOCUMENT = DocumentInput(
    "file:///notes.md",
    ["Acme was founded in 2019.", "Its office is in Kreuzberg."],
    T0 + timedelta(days=1),
    "Company notes",
)


NAME_TYPES = {"Berlin": 1, "Kreuzberg": 1, "Acme": 2}
EPISODE_TEXT = re.compile(r"<episode_text>\n(.*?)\n</episode_text>", re.DOTALL)


def scripted_entities(request) -> dict:
    """Reply with the known names that appear in the episode text."""
    text = EPISODE_TEXT.search(request.prompt).group(1)
    return {
        "entities": [
            {"name": name, "type_id": type_id, "confidence": 0.9}
            for name, type_id in NAME_TYPES.items()
            if name in text
        ]
    }


async def extract_entities(
    store: EpisodeStore, episodes: list[Episode]
) -> tuple[ExtractionReport, ScriptedLLM]:
    """Extract entities with a scripted model and the two latest earlier episodes.

    Returns the report and the model client with its recorded requests.
    """
    llm = ScriptedLLM(answer=scripted_entities)
    extraction = EntityExtraction(
        [LLMExtractor(gate(llm))],
        RecordMapper({}),
        catalog(),
        store,
        EntityExtractionSettings(max_previous=2),
    )
    report = await extraction.extract(episodes)
    return report, llm


def assert_entities_extracted(
    report: ExtractionReport, episodes: list[Episode]
) -> None:
    """Assert each episode yields the names in its text and its named actor."""
    by_id = {episode.id: episode for episode in episodes}

    assert {result.episode_id for result in report.results} == set(by_id)
    for result in report.results:
        episode = by_id[result.episode_id]
        content = episode.content
        if not isinstance(content, str):
            raise AssertionError("expected a text episode")
        named = [item for item in result.entities if item.local_id != "actor"]
        assert [item.name for item in named] == [
            name for name in NAME_TYPES if name in content
        ]
        for item in named:
            assert item.type_id == NAME_TYPES[item.name]
            assert [content[start:end] for start, end in item.spans] == [item.name]

        actors = [item for item in result.entities if item.local_id == "actor"]
        if episode.kind is Kind.MESSAGE:
            assert [(item.name, list(item.spans)) for item in actors] == [
                (episode.actor.display_name or episode.actor.id, [])
            ]
            assert [item.attributes["actor_id"] for item in actors] == [
                episode.actor.id
            ]
        else:
            assert actors == []


async def assert_ingested(
    store: EpisodeStore,
    handed_off: list[list[str]],
    node_ids: Callable[[str], Awaitable[set[str]]] | None = None,
) -> None:
    """Assert every episode, source and actor was written once and reads back."""
    chat_ids, document_ids = handed_off
    ids = [*chat_ids, *document_ids]
    stored = await store.get_episodes(ids)
    by_id = {episode.id: episode for episode in stored}

    assert set(by_id) == set(ids)
    assert [by_id[episode_id].content for episode_id in chat_ids] == [
        message.text for message in CHAT
    ]
    assert [by_id[episode_id].content for episode_id in document_ids] == DOCUMENT.parts
    assert {by_id[episode_id].kind for episode_id in chat_ids} == {Kind.MESSAGE}
    assert {by_id[episode_id].kind for episode_id in document_ids} == {Kind.DOCUMENT}
    assert {by_id[episode_id].source.system for episode_id in chat_ids} == {"chat"}
    assert {by_id[episode_id].source.system for episode_id in document_ids} == {
        "document"
    }
    assert {by_id[episode_id].actor.id for episode_id in chat_ids} == {"u1", "bot"}
    assert {by_id[episode_id].actor.id for episode_id in document_ids} == {"document"}
    for episode in stored:
        assert episode.content_hash is not None
        assert await store.get_content_hash(episode.id) == episode.content_hash

    if node_ids is not None:
        assert await node_ids("Episode") == set(ids)
        assert await node_ids("Source") == {episode.source.id for episode in stored}
        assert await node_ids("Actor") == {episode.actor.id for episode in stored}


async def ingest_all(store, collected: list[Episode] | None = None) -> list[list[str]]:
    """Ingest the chat and the document, returning what the hand-off saw."""
    handed_off: list[list[str]] = []

    async def handoff(episodes):
        handed_off.append([e.id for e in episodes])
        if collected is not None:
            collected.extend(episodes)

    pipeline = IngestionPipeline(store, default_registry(), FixedClock(), handoff)
    await pipeline.ingest("message", CHAT)
    await pipeline.ingest("document", [DOCUMENT])
    await pipeline.ingest("message", CHAT)
    return handed_off


class TestPipeline:
    """Ingestion writes the same episodes on every store."""

    async def test_memory_twin_writes_each_episode_once(self):
        """The in-memory twin keeps one episode per message and document part."""
        store = MemoryEpisodeStore()

        handed_off = await ingest_all(store)

        assert [len(batch) for batch in handed_off] == [4, 2]
        await assert_ingested(store, handed_off)
        assert len(store.all_episodes()) == 6

    @pytest.mark.parametrize("backend", ["falkordb", "neo4j"])
    async def test_graph_backend_writes_the_same_episodes(
        self, backend, falkordb_store, neo4j_store
    ):
        """A real backend reads back the same episodes, sources and actors."""
        store = falkordb_store if backend == "falkordb" else neo4j_store
        handle = await store.open(new_tenant(), [EPISODES])
        runner = store.runner(handle)
        keyed = backend == "neo4j"
        try:
            harness = graph_harness(runner, tenant_keyed=keyed)

            handed_off = await ingest_all(harness.store)

            assert [len(batch) for batch in handed_off] == [4, 2]
            await assert_ingested(harness.store, handed_off, harness.node_ids)
        finally:
            await store.erase_tenant(handle.tenant)


class TestEntityExtraction:
    """Entities extracted from ingested episodes are the same on every store."""

    async def run_on(
        self, store: EpisodeStore
    ) -> tuple[list[Episode], ExtractionReport, ScriptedLLM]:
        """Ingest into `store`, then extract from the handed-off episodes."""
        episodes: list[Episode] = []
        await ingest_all(store, episodes)
        report, llm = await extract_entities(store, episodes)
        return episodes, report, llm

    def assert_context_was_given(self, llm: ScriptedLLM) -> None:
        """Check that a later message saw the earlier one as context only."""
        (request,) = [
            r
            for r in llm.requests
            if "Kreuzberg. I work" in r.prompt.split("<episode_text>")[-1]
        ]
        context = request.prompt.split("<episode_text>")[0]
        assert "I moved to Berlin last week." in context
        assert "Welcome to Berlin!" in context

    async def test_memory_twin_extracts_expected_entities(self):
        """The twin supplies earlier episodes and extracts the names in each one."""
        episodes, report, llm = await self.run_on(MemoryEpisodeStore())

        self.assert_context_was_given(llm)
        assert_entities_extracted(report, episodes)

    @pytest.mark.parametrize("backend", ["falkordb", "neo4j"])
    async def test_graph_backend_extracts_the_same_entities(
        self, backend, falkordb_store, neo4j_store
    ):
        """A real backend gives the same extraction as the twin."""
        store = falkordb_store if backend == "falkordb" else neo4j_store
        handle = await store.open(new_tenant(), [EPISODES])
        try:
            harness = graph_harness(
                store.runner(handle), tenant_keyed=backend == "neo4j"
            )

            episodes, report, llm = await self.run_on(harness.store)

            self.assert_context_was_given(llm)
            assert_entities_extracted(report, episodes)
        finally:
            await store.erase_tenant(handle.tenant)


class TestTemporal:
    """Facts from extraction become versioned FACT edges on each graph backend."""

    @pytest.fixture(params=["falkordb", "neo4j"])
    async def backend(self, request, falkordb_store, neo4j_store):
        """Yield a tenant opened with every graph part the path writes."""
        keyed = request.param == "neo4j"
        store = neo4j_store if keyed else falkordb_store
        handle = await store.open(new_tenant(), [EPISODES, RESOLUTION, TEMPORAL])
        try:
            yield await temporal_stack(store, handle, keyed)
        finally:
            await store.erase_tenant(handle.tenant)

    async def test_later_fact_closes_the_earlier_one(self, backend):
        """Two employers in turn: Acme closes at 2025, Globex is open."""
        view = (await backend.manager.extraction_view()).view
        first, second = ana_acme_globex(view)
        await backend.ingest(
            episode_at("ep-2024", minutes=0, content=TEMPORAL_TEXTS["ep-2024"]), first
        )
        await backend.ingest(
            episode_at("ep-2025", minutes=1, content=TEMPORAL_TEXTS["ep-2025"]), second
        )

        ana = await backend.ref("ep-2024", "e0")
        globex = await backend.ref("ep-2025", "e1")
        acme = await backend.ref("ep-2024", "e1")
        current = await backend.temporal.query(TemporalQuery(now=NOW, subject_id=ana))
        acme_before = await backend.temporal.query(
            TemporalQuery(
                now=NOW,
                subject_id=ana,
                object_id=acme,
                valid_at=datetime(2024, 6, 1, tzinfo=UTC),
            )
        )

        assert [v.object_id for v in current] == [globex]
        assert [(v.object_id, v.valid_to) for v in acme_before] == [
            (acme, datetime(2025, 1, 1, tzinfo=UTC))
        ]

    async def test_deprecating_a_relation_keeps_the_stored_relation(self, backend):
        """Stored facts keep their relation; the next episode uses the other one."""
        first_projection = await backend.manager.extraction_view()
        works_before = relation_id(first_projection.view, "works_at")
        first, _ = ana_acme_globex(first_projection.view)
        await backend.ingest(
            episode_at("ep-2024", content=TEMPORAL_TEXTS["ep-2024"]), first
        )
        await backend.manager.apply(
            cs(
                await backend.manager.catalog(),
                DeprecateType(kind=OntologyKind.RELATION, target="works_at"),
            )
        )
        view = await backend.manager.extraction_view()
        person, org = people_and_orgs(view.view)
        await backend.ingest(
            episode_at("ep-employs", minutes=2, content=TEMPORAL_TEXTS["ep-employs"]),
            [
                entity("e0", "Acme", org, "Organization", "ep-employs"),
                entity("e1", "Ana", person, "Person", "ep-employs"),
            ],
        )

        first_facts = await backend.temporal.facts_of_episode("ep-2024")
        later_facts = await backend.temporal.facts_of_episode("ep-employs")
        employs = view.stable_relation_id(relation_id(view.view, "employs"))

        assert {f.relation_type_id for f in first_facts} == {
            first_projection.stable_relation_id(works_before)
        }
        assert {f.relation_type_id for f in later_facts} == {employs}


STAGE_STATUSES = {
    "ingest_cleanup": "skipped",
    "deduplication": "ok",
    "feedback_weights": "ok",
    "descriptions": "ok",
    "communities": "ok",
    "reflection": "ok",
    "scoring": "ok",
    "prune_archive": "ok",
    "report": "ok",
}
CURATOR_MARK = "propose lessons"
WRITER_MARK = "decide whether to add one lesson"


def consolidation_answer(entry_id: str):
    """Answer every consolidation model call with a fixed, valid reply."""

    def answer(request) -> dict:
        if CURATOR_MARK in request.system:
            return {
                "lessons": [
                    {"text": "Ana prefers short answers", "entry_ids": [entry_id]}
                ]
            }
        if WRITER_MARK in request.system:
            return {"verdict": "accept"}
        return {"summary": "Ana and Globex.", "description": "A person and a firm."}

    return answer


class TestFullCycle:
    """Ingest, resolve, extract facts, settle them and consolidate, on each backend."""

    @pytest.fixture(params=["falkordb", "neo4j"])
    async def cycle(self, request, falkordb_store, neo4j_store):
        """Yield the temporal stack and a consolidation store on one tenant."""
        keyed = request.param == "neo4j"
        store = neo4j_store if keyed else falkordb_store
        handle = await store.open(
            new_tenant(), [EPISODES, RESOLUTION, TEMPORAL, CONSOLIDATION]
        )
        try:
            stack = await temporal_stack(store, handle, keyed)
            yield stack, GraphConsStore(store, handle)
        finally:
            await store.erase_tenant(handle.tenant)

    async def test_every_stage_runs_then_a_second_run_changes_nothing(self, cycle):
        """Every stage writes its result once and the second run leaves it alone."""
        stack, cons = cycle
        view = (await stack.manager.extraction_view()).view
        first, second = ana_acme_globex(view)
        await stack.ingest(
            episode_at("ep-2024", minutes=0, content=TEMPORAL_TEXTS["ep-2024"]), first
        )
        await stack.ingest(
            episode_at("ep-2025", minutes=1, content=TEMPORAL_TEXTS["ep-2025"]), second
        )
        ana = await stack.ref("ep-2024", "e0")
        stored = (await stack.resolver.get_entities([ana]))[0]
        await stack.resolver.save_entity(
            stored.model_copy(update={"summary_stale": True}), bump_revision=False
        )
        now = FixedClock().now()
        await record_feedback(cons, ana, 1.0, now)
        await record_access(cons, ana, now)
        entry = await record_session_entry(
            cons, "s1", "Ana prefers short answers", now, helpfulness=1
        )
        llm = ScriptedLLM(answer=consolidation_answer(entry.entry_id))
        run_ids = iter(f"run-{i}" for i in range(1, 10))
        runner = ConsolidationRunner(
            default_stages(),
            cons,
            FixedClock().now,
            ConsolidationSettings(_env_file=None, prune=PruneSettings(dry_run=False)),
            resolution=stack.resolver,
            temporal=await stack.engine(),
            llm=llm,
            run_ids=lambda: next(run_ids),
        )

        first_run = await runner.run()
        first_scores = await cons.list(Score)
        first_proposals = await cons.list(Proposal)
        first_entity = (await stack.resolver.get_entities([ana]))[0]
        calls = len(llm.requests)
        second_run = await runner.run()

        assert first_run.status is RunStatus.OK
        assert {s.stage: s.status.value for s in first_run.stages} == STAGE_STATUSES
        assert first_run.stage("descriptions").counts["rewritten_cheap"] >= 1
        assert first_run.stage("communities").counts["communities"] >= 1
        assert first_run.stage("reflection").counts["proposed"] == 1
        assert first_run.stage("report").counts["pending_proposals"] == 1
        assert first_entity.summary_stale is False
        assert first_scores
        assert calls > 0

        assert second_run.status is RunStatus.OK
        second_descriptions = second_run.stage("descriptions")
        second_reflection = second_run.stage("reflection")
        assert second_descriptions.status is StageStatus.SKIPPED
        assert second_descriptions.notes == ["no stale summaries"]
        assert second_reflection.status is StageStatus.SKIPPED
        assert second_reflection.notes == ["no new eligible session entries"]
        assert len(llm.requests) == calls
        assert await cons.list(Score) == first_scores
        assert (await stack.resolver.get_entities([ana]))[0] == first_entity
        assert await cons.list(Proposal) == first_proposals
        assert [p.status for p in first_proposals] == [ProposalStatus.PENDING]

    async def test_resetting_an_episode_then_consolidating_leaves_no_retraction(
        self, cycle
    ):
        """A reset episode is consolidated again and no retraction entry appears."""
        stack, cons = cycle
        view = (await stack.manager.extraction_view()).view
        first, second = ana_acme_globex(view)
        await stack.ingest(
            episode_at("ep-2024", minutes=0, content=TEMPORAL_TEXTS["ep-2024"]), first
        )
        await stack.ingest(
            episode_at("ep-2025", minutes=1, content=TEMPORAL_TEXTS["ep-2025"]), second
        )
        llm = ScriptedLLM(answer=consolidation_answer("none"))
        engine = await stack.engine()
        runner = ConsolidationRunner(
            default_stages(),
            cons,
            FixedClock().now,
            ConsolidationSettings(_env_file=None, prune=PruneSettings(dry_run=False)),
            resolution=stack.resolver,
            temporal=engine,
            llm=llm,
        )
        await runner.run()

        await engine.reset_episode("ep-2025", stack.resolver)
        result = await runner.run()

        assert result.status is RunStatus.OK
        tenant_ops = [
            entry.op.value
            for entry in await stack.temporal.ledger_entries(TENANT_CHAIN)
        ]
        slot_ops = [
            entry.op.value
            for slot_key in await stack.temporal.slot_keys()
            for entry in await stack.temporal.ledger_entries(slot_chain(slot_key))
        ]
        assert "fact_withdrawn" in slot_ops
        assert "episode_retracted" not in tenant_ops
