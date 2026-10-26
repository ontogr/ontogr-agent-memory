"""Prove episode and fact vectors remain reachable through temporal updates."""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import timedelta

import pytest

from ontogr_mem.data_models import Episode, Kind, Role
from ontogr_mem.entity_extraction import EntityExtraction, EntityExtractionSettings
from ontogr_mem.entity_extraction.extractors import LLMExtractor, RecordMapper
from ontogr_mem.fact_extraction import (
    FactExtraction,
    InMemoryFactStore,
    LLMFactExtractor,
)
from ontogr_mem.graphstore import GraphHandle, GraphStore, QuerySpec, ScopedRunner
from ontogr_mem.graphstore.episode_store import GraphEpisodeStore
from ontogr_mem.graphstore.memory import (
    MemoryEpisodeStore,
    MemoryOntologyStore,
    MemoryResolverStore,
    MemoryTemporalStore,
)
from ontogr_mem.graphstore.resolver_store import GraphResolverStore
from ontogr_mem.graphstore.schemas import EPISODES, RESOLUTION, TEMPORAL, VECTOR
from ontogr_mem.graphstore.temporal.store import GraphTemporalStore
from ontogr_mem.ingestion import ChatMessage, IngestionPipeline, default_registry
from ontogr_mem.ontology.extraction_items import stabilize_facts
from ontogr_mem.ontology.manager import OntologyManager
from ontogr_mem.ontology.settings import OntologySettings
from ontogr_mem.resolution import EntityResolver, ResolutionPolicy
from ontogr_mem.resolution.config import ResolverSettings
from ontogr_mem.resolution.policy import ResolutionMode, TypePolicy
from ontogr_mem.temporal import TemporalEngine, from_type_view
from ontogr_mem.vectorstore import (
    FactVectorObserver,
    FailReason,
    SearchFilters,
    TemporalFilter,
    VectorKind,
    VectorStore,
    VectorWriter,
    write_episode_vectors,
)
from ontogr_mem.vectorstore.errors import PermanentEmbeddingError
from ontogr_mem.vectorstore.graph.falkordb import FalkorDBVectorBackend
from ontogr_mem.vectorstore.graph.neo4j import Neo4jVectorBackend
from ontogr_mem.vectorstore.memory import InMemoryBackend
from ontogr_mem.vectorstore.text import edge_text
from tests.support.clock import FixedClock
from tests.support.embedder import MockEmbedder, gateway
from tests.support.episodes import T0
from tests.support.extraction import gate as extraction_gate
from tests.support.facts import fact, reply
from tests.support.facts import gate as fact_gate
from tests.support.llm import ScriptedLLM
from tests.support.ontology import base
from tests.support.tenants import new_tenant
from tests.support.vectors import HALF, OPPOSITE, ORTHOGONAL, A


FIRST_MESSAGE = "Ana works at Acme since March 2024."
SECOND_MESSAGE = "I like this neighborhood."
LATER_MESSAGE = "Ana works at Globex since January 2025."
FAILED_MESSAGE = "Ana works at Initech since October 2026."
FIRST_FACT_TEXT = "Ana works at Acme"
LATER_FACT_TEXT = "Ana works at Globex"
FAILED_FACT_TEXT = "Ana works at Initech"
VECTORS = {
    FIRST_MESSAGE: A,
    SECOND_MESSAGE: HALF,
    LATER_MESSAGE: ORTHOGONAL,
    FAILED_MESSAGE: OPPOSITE,
    FIRST_FACT_TEXT: A,
    LATER_FACT_TEXT: ORTHOGONAL,
    FAILED_FACT_TEXT: HALF,
}


class SelectiveFailureEmbedder(MockEmbedder):
    """A fixed embedder that can fail for one selected text."""

    def __init__(self) -> None:
        """Create the embedder with its failure disabled."""
        super().__init__(vectors=VECTORS)
        self.failed_text: str | None = None

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Fail only when a batch contains the selected text."""
        if self.failed_text in texts:
            raise PermanentEmbeddingError("selected text is unavailable")
        return await super().embed_batch(texts)


@dataclass
class WriterStack:
    """The components that run one ingest and temporal path."""

    graph: GraphStore | None
    handle: GraphHandle | None
    runner: ScopedRunner | None
    keyed: bool
    episodes: MemoryEpisodeStore | GraphEpisodeStore
    temporal: MemoryTemporalStore | GraphTemporalStore
    resolver: EntityResolver
    manager: OntologyManager
    vectors: VectorStore


def _messages(*texts: str, offset: int = 0) -> list[ChatMessage]:
    return [
        ChatMessage(
            message_id=f"message-{offset + index}",
            session_id="vector-writers",
            author_id="ana",
            role=Role.USER,
            text=text,
            sent_at=T0 + timedelta(minutes=offset + index),
            author_name="Ana",
        )
        for index, text in enumerate(texts)
    ]


async def _stack(variant: str, falkordb_store, neo4j_store) -> WriterStack:
    """Open the requested vector and graph stores."""
    fixed_clock = FixedClock()
    graph: GraphStore | None = None
    handle: GraphHandle | None = None
    runner: ScopedRunner | None = None
    keyed = variant == "neo4j"

    if variant == "twin":
        episodes = MemoryEpisodeStore()
        resolver_store = MemoryResolverStore()
        temporal = MemoryTemporalStore(episodes, resolver_store)
        vector_backend = InMemoryBackend()
    else:
        graph = neo4j_store if keyed else falkordb_store
        handle = await graph.open(
            new_tenant(), [EPISODES, RESOLUTION, TEMPORAL, VECTOR]
        )
        runner = graph.runner(handle)
        episodes = GraphEpisodeStore(runner, tenant_keyed=keyed)
        temporal = GraphTemporalStore(graph, handle, tenant_keyed=keyed)
        resolver_store = GraphResolverStore(runner, tenant_keyed=keyed)
        vector_backend = (
            Neo4jVectorBackend(graph, handle)
            if keyed
            else FalkorDBVectorBackend(graph, handle)
        )

    vector_embedder = SelectiveFailureEmbedder()
    vectors = VectorStore(vector_backend, gateway(vector_embedder))
    await vectors.ensure_ready()
    resolver_vectors = VectorStore(InMemoryBackend(), gateway(MockEmbedder()))
    await resolver_vectors.ensure_ready()
    resolver = EntityResolver(
        resolver_store,
        resolver_vectors,
        ResolutionPolicy(
            types={
                "Person": TypePolicy(mode=ResolutionMode.AUTO_KEY),
                "Organization": TypePolicy(mode=ResolutionMode.AUTO_KEY),
            }
        ),
        ResolverSettings(inline_verification=False),
        fixed_clock,
    )
    manager = OntologyManager(
        MemoryOntologyStore(set[str]),
        ScriptedLLM(),
        OntologySettings(min_support=3),
        fixed_clock,
    )
    await manager.load_catalog(base())
    return WriterStack(
        graph,
        handle,
        runner,
        keyed,
        episodes,
        temporal,
        resolver,
        manager,
        vectors,
    )


def _entity_answer(episode: Episode, type_ids: dict[str, int]) -> dict[str, object]:
    """Return the scripted entity reply for the names in an episode."""
    names = []
    if "Ana" in episode.text:
        names.append({"name": "Ana", "type_id": type_ids["Person"], "confidence": 0.9})
    for company in ("Acme", "Globex", "Initech"):
        if company in episode.text:
            names.append(
                {
                    "name": company,
                    "type_id": type_ids["Organization"],
                    "confidence": 0.9,
                }
            )
    return {"entities": names}


def _fact_answer(episode: Episode, relation_id: int) -> str:
    """Return a scripted state fact for the messages that assert one."""
    company, statement, valid_from, time_expression = {
        FIRST_MESSAGE: ("Acme", FIRST_FACT_TEXT, "2024-03", "March 2024"),
        LATER_MESSAGE: ("Globex", LATER_FACT_TEXT, "2025-01", "January 2025"),
        FAILED_MESSAGE: ("Initech", FAILED_FACT_TEXT, "2026-10", "October 2026"),
    }.get(episode.text, ("", "", "", ""))
    if not company:
        return reply()
    return reply(
        fact(
            text=statement,
            fact_kind="state",
            entity_refs=["e0", "e1"],
            relation_type_id=relation_id,
            evidence_quote=statement,
            valid_from=valid_from,
            precision="month",
            time_expression=time_expression,
        )
    )


async def _process(stack: WriterStack, episodes: Sequence[Episode], observer):
    """Extract, resolve, stabilize and temporally commit a batch."""
    projection = await stack.manager.extraction_view()
    type_ids = {
        label: projection.view.catalog.entity_by_label(label).id
        for label in ("Person", "Organization")
    }
    entity_llm = ScriptedLLM(
        answer=lambda request: _entity_answer(
            next(e for e in episodes if e.text in request.prompt), type_ids
        )
    )
    entities_report = await EntityExtraction(
        [LLMExtractor(extraction_gate(entity_llm))],
        RecordMapper({}),
        projection.view.catalog,
        stack.episodes,
        EntityExtractionSettings(max_previous=0),
    ).extract(episodes)

    by_episode = {}
    for result in entities_report.results:
        enriched = [
            replace(
                entity,
                type_label=projection.view.catalog.entity_label(entity.type_id),
            )
            for entity in result.entities
        ]
        by_episode[result.episode_id] = enriched
        await stack.resolver.resolve(enriched)

    relation = projection.view.catalog.relation_by_label("works_at")
    assert relation is not None
    fact_llm = ScriptedLLM(
        answer=lambda request: _fact_answer(
            next(e for e in episodes if e.text in request.prompt), relation.id
        )
    )
    extraction = FactExtraction(
        LLMFactExtractor(fact_gate(fact_llm)),
        InMemoryFactStore(),
        projection.view,
    )
    facts_report = await extraction.extract(episodes, by_episode)
    dense_relation = relation.id
    engine = TemporalEngine(
        stack.temporal,
        from_type_view(
            projection.view,
            [dense_relation],
            projection.stable_relation_id,
        ),
        stack.resolver.store,
        FixedClock().now,
        hooks=[observer],
    )
    reports = []
    for episode, result in zip(episodes, facts_report.results, strict=True):
        stabilized = await stabilize_facts(stack.manager, result.facts, projection)
        reports.append(await engine.ingest(stabilized.typed, episode))
    return reports


async def _edge_versions(stack: WriterStack, fact_id: str) -> list[dict[str, object]]:
    """Read the stored versions of one fact edge on graph-native backends."""
    if stack.runner is None or stack.handle is None:
        return []
    tenant_clause = " AND f._tenant = $tenant" if stack.keyed else ""
    result = await stack.runner.run(
        QuerySpec(
            "MATCH ()-[f:FACT]->() WHERE f.fact_id = $id"
            f"{tenant_clause} "
            "RETURN f.version AS version, f.embedding IS NOT NULL AS vectorized "
            "ORDER BY version",
            {
                "id": fact_id,
                **({"tenant": stack.handle.tenant} if stack.keyed else {}),
            },
            read_only=True,
            idempotent=True,
            tag="vector_writer_versions",
        )
    )
    return result.rows


@dataclass
class WriterRun:
    """The writer components shared across one end-to-end scenario."""

    writer: VectorWriter
    observer: FactVectorObserver
    embedder: SelectiveFailureEmbedder
    pipeline: IngestionPipeline
    handed_off: list[Episode]


def _pipeline(
    stack: WriterStack,
) -> WriterRun:
    """Build the writer hooks and the ingestion hand-off."""
    writer = VectorWriter(stack.vectors)
    observer = FactVectorObserver(writer)
    handed_off: list[Episode] = []
    embedder = stack.vectors.gateway.embedder
    assert isinstance(embedder, SelectiveFailureEmbedder)

    async def handoff(episodes: Sequence[Episode]) -> None:
        handed_off.extend(episodes)
        await write_episode_vectors(writer, episodes)

    return WriterRun(
        writer,
        observer,
        embedder,
        IngestionPipeline(stack.episodes, default_registry(), FixedClock(), handoff),
        handed_off,
    )


async def _ingest_and_commit(
    stack: WriterStack,
    pipeline: IngestionPipeline,
    handed_off: list[Episode],
    observer: FactVectorObserver,
    texts: Sequence[str],
    *,
    offset: int = 0,
) -> list[Episode]:
    """Ingest messages, then run their entity and temporal path."""
    await pipeline.ingest(Kind.MESSAGE, _messages(*texts, offset=offset))
    episodes = handed_off[-len(texts) :]
    assert len(episodes) == len(texts)
    reports = await _process(stack, episodes, observer)
    assert all(report.ok for report in reports)
    return episodes


async def _assert_graph_vectors(
    stack: WriterStack, episodes: Sequence[Episode], fact_id: str
) -> None:
    """Check graph object properties and the vectorless newer fact version."""
    if stack.graph is None or stack.handle is None:
        return
    rows = await _edge_versions(stack, fact_id)
    assert len(rows) >= 2
    assert rows[0]["vectorized"]
    assert not rows[-1]["vectorized"]
    stored_episodes = await stack.graph.get_nodes(
        stack.handle, "Episode", [episode.id for episode in episodes]
    )
    assert {node.id for node in stored_episodes} == {episode.id for episode in episodes}
    assert all("embedding" in node.props for node in stored_episodes)
    assert await stack.vectors.exists(
        VectorKind.EPISODE, [episode.id for episode in episodes]
    ) == {episode.id for episode in episodes}
    assert fact_id in await stack.vectors.exists(VectorKind.EDGE, [fact_id])


async def _assert_post_commit_failure(
    stack: WriterStack,
    run: WriterRun,
    existing_fact_ids: set[str],
    history_filter: SearchFilters,
) -> str:
    """Prove a committed fact is retained for retry after its vector fails."""
    run.embedder.failed_text = edge_text(FAILED_FACT_TEXT, "", "", "")
    (episode,) = await _ingest_and_commit(
        stack,
        run.pipeline,
        run.handed_off,
        run.observer,
        [FAILED_MESSAGE],
        offset=10,
    )
    facts = await stack.temporal.facts_of_episode(episode.id)
    fact_ids = {item.fact_id for item in facts} - existing_fact_ids
    assert len(fact_ids) == 1
    fact_id = fact_ids.pop()
    assert any(item.fact_id == fact_id for item in facts)
    assert await stack.vectors.exists(VectorKind.EDGE, [fact_id]) == set()
    assert len(run.writer.failed) == 1
    assert run.writer.failed[0].item.id == fact_id
    assert run.writer.failed[0].reason is FailReason.EMBED_FAILED
    assert "selected text is unavailable" in run.writer.failed[0].detail
    hits = await stack.vectors.search(
        VectorKind.EDGE, HALF, k=10, filters=history_filter
    )
    assert fact_id not in {hit.id for hit in hits}
    if stack.graph is not None:
        rows = await _edge_versions(stack, fact_id)
        assert rows
        assert all(not row["vectorized"] for row in rows)
    return fact_id


class TestVectorWriters:
    """Exercise episode and fact writers across twin and graph-native stores."""

    @pytest.fixture(params=["twin", "falkordb", "neo4j"])
    async def stack(self, request, falkordb_store, neo4j_store):
        """Build a fresh stack for each storage implementation."""
        built = await _stack(request.param, falkordb_store, neo4j_store)
        try:
            yield built
        finally:
            if built.graph is not None and built.handle is not None:
                await built.graph.erase_tenant(built.handle.tenant)

    async def test_vectors_remain_searchable_after_temporal_commits(
        self, stack: WriterStack
    ):
        """Search history after versioning and repair a post-commit embed failure."""
        run = _pipeline(stack)
        first_pair = await _ingest_and_commit(
            stack,
            run.pipeline,
            run.handed_off,
            run.observer,
            [FIRST_MESSAGE, SECOND_MESSAGE],
        )
        episode_hits = await stack.vectors.search(VectorKind.EPISODE, A, k=10)
        assert [hit.id for hit in episode_hits[:2]] == [
            first_pair[0].id,
            first_pair[1].id,
        ]
        assert await stack.vectors.exists(
            VectorKind.EPISODE, [episode.id for episode in first_pair]
        ) == {episode.id for episode in first_pair}

        first_ref = await stack.resolver.store.entity_id_for_ref(first_pair[0].id, "e0")
        assert first_ref is not None
        first_facts = await stack.temporal.facts_of_episode(first_pair[0].id)
        assert len(first_facts) == 1
        first_fact_id = first_facts[0].fact_id
        history_filter = SearchFilters(temporal=TemporalFilter(history=True))
        first_hits = await stack.vectors.search(
            VectorKind.EDGE, A, k=10, filters=history_filter
        )
        assert [hit.id for hit in first_hits] == [first_fact_id]

        (later_episode,) = await _ingest_and_commit(
            stack,
            run.pipeline,
            run.handed_off,
            run.observer,
            [LATER_MESSAGE],
            offset=5,
        )
        versioned_facts = await stack.temporal.facts_of_episode(later_episode.id)
        fact_ids = {item.fact_id for item in versioned_facts}
        assert len(fact_ids) == 2
        later_fact_id = (fact_ids - {first_fact_id}).pop()
        history_hits = await stack.vectors.search(
            VectorKind.EDGE, A, k=10, filters=history_filter
        )
        assert {hit.id for hit in history_hits} == {first_fact_id, later_fact_id}
        # Only graph-native stores read the updated validity on the FACT edge.
        if stack.graph is not None:
            current_hits = await stack.vectors.search(VectorKind.EDGE, A, k=10)
            assert first_fact_id not in {hit.id for hit in current_hits}
        await _assert_graph_vectors(stack, first_pair, first_fact_id)
        failed_fact_id = await _assert_post_commit_failure(
            stack,
            run,
            fact_ids,
            history_filter,
        )

        run.embedder.failed_text = None
        assert await run.writer.retry() == 0
        assert run.writer.failed == ()
        assert await stack.vectors.exists(VectorKind.EDGE, [failed_fact_id]) == {
            failed_fact_id
        }
        retried_hits = await stack.vectors.search(
            VectorKind.EDGE, HALF, k=10, filters=history_filter
        )
        assert failed_fact_id in {hit.id for hit in retried_hits}
