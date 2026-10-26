"""End-to-end proof of reset: a second run over an episode replaces its facts.

Two episodes make claims about the same entities. The first episode is reset
and ingested again with a different claim. The facts of the second episode
stay live, the ledger keeps the withdrawal, and no retraction entry is written.
The resolver scenarios run the same reset with entities linked by the real
entity resolver, including a crash between the mention write and the
observation write. The scenarios run on FalkorDB and on Neo4j.
"""

from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Any

import pytest

from ontogr_mem.data_models.episode import TemporalStatus
from ontogr_mem.data_models.extraction import ExtractedEntity
from ontogr_mem.data_models.fact_versions import FactStatus
from ontogr_mem.data_models.facts import (
    ExtractedFact,
    FactKind,
    Method,
    Modality,
    Polarity,
    Precision,
    TimeBasis,
    TypedFact,
)
from ontogr_mem.data_models.observation import Observation
from ontogr_mem.graphstore.falkordb import FalkorDBGraphStore
from ontogr_mem.graphstore.neo4j import Neo4jGraphStore
from ontogr_mem.graphstore.resolver_store import GraphResolverStore
from ontogr_mem.graphstore.schemas import EPISODES, RESOLUTION, TEMPORAL
from ontogr_mem.graphstore.temporal.store import GraphTemporalStore
from ontogr_mem.resolution import EntityResolver, ResolutionPolicy
from ontogr_mem.resolution.config import ResolverSettings
from ontogr_mem.resolution.errors import EpisodeAlreadyResolvedError
from ontogr_mem.resolution.policy import ResolutionMode, TypePolicy
from ontogr_mem.resolution.resolver_store import ResolverStore
from ontogr_mem.temporal.audit import audit
from ontogr_mem.temporal.cardinality import (
    Cardinality,
    DictRelationPolicy,
    RelationRules,
)
from ontogr_mem.temporal.engine import TemporalEngine
from ontogr_mem.temporal.ledger import TENANT_CHAIN, slot_chain, verify
from ontogr_mem.temporal.query import TemporalQuery
from ontogr_mem.temporal.temporal_store import EpisodeState, TemporalStore
from ontogr_mem.vectorstore import EmbeddingGateway, VectorStore
from ontogr_mem.vectorstore.memory import InMemoryBackend
from tests.support.clock import FixedClock
from tests.support.embedder import MockEmbedder
from tests.support.episodes import make_episode
from tests.support.temporal import TemporalHarness, graph_temporal_harness
from tests.support.tenants import new_tenant
from tests.unit.temporal.helpers import LocalRefResolver, at


ONE = DictRelationPolicy({1: RelationRules(Cardinality.ONE)})
WORKS_AT = 1


class FakeMentions:
    """Stands in for resolution: records the episodes whose mentions go."""

    def __init__(self) -> None:
        """Start with no removals."""
        self.removed: list[str] = []

    async def remove_episode_mentions(self, episode_id: str) -> list[str]:
        """Record the episode and delete no entity."""
        self.removed.append(episode_id)
        return []


def works_at(person: str, company: str, episode_id: str, quote: str) -> TypedFact:
    """Build a stable-typed fact that `person` works at `company`."""
    fact = ExtractedFact(
        fact_id=f"{episode_id}-{person}",
        episode_id=episode_id,
        text=quote,
        entity_refs=(person, company),
        relation_type_id=WORKS_AT,
        relation_label=None,
        fact_kind=FactKind.STATE,
        polarity=Polarity.AFFIRMED,
        modality=Modality.ACTUAL,
        asserted_by=None,
        valid_from=at(2020),
        valid_to=None,
        precision=Precision.YEAR,
        time_expression="in 2020",
        time_basis=TimeBasis.STATED,
        method=Method.LLM,
        evidence_quote=quote,
        evidence_span=None,
        confidence=1.0,
    )
    return TypedFact(fact=fact, relation_type=WORKS_AT, catalog_version="v1")


def engine_for(store) -> TemporalEngine:
    """Build an engine on the single-valued policy with a fixed clock."""
    return TemporalEngine(store, ONE, LocalRefResolver(), FixedClock().now)


async def _current_pairs(store) -> list[tuple[str, str]]:
    return sorted((f.subject_id, f.object_id) for f in await store.current_facts())


class ResetScenario:
    """Reset then re-ingest one episode while another episode keeps its fact."""

    async def _seeded(self, make_harness) -> TemporalHarness:
        harness = await make_harness()
        await harness.seed_entities(["alice", "bob", "acme", "beta"])
        await harness.seed_episode("e1")
        await harness.seed_episode("e2")
        return harness

    async def test_reset_replaces_one_episode_and_keeps_the_other(self, make_harness):
        """The rerun shows the new claim; the other episode's fact survives."""
        harness = await self._seeded(make_harness)
        store = harness.store
        engine = engine_for(store)
        await engine.ingest(
            [works_at("alice", "acme", "e1", "Alice works at Acme")],
            make_episode("e1"),
        )
        await engine.ingest(
            [works_at("bob", "acme", "e2", "Bob works at Acme")],
            make_episode("e2"),
        )
        assert await _current_pairs(store) == [("alice", "acme"), ("bob", "acme")]

        mentions = FakeMentions()
        await engine.reset_episode("e1", mentions)
        assert await _current_pairs(store) == [("bob", "acme")]
        assert mentions.removed == ["e1"]

        await engine.ingest(
            [works_at("alice", "beta", "e1", "Alice works at Beta")],
            make_episode("e1"),
        )

        assert await _current_pairs(store) == [("alice", "beta"), ("bob", "acme")]
        state = await store.episode_state("e1")
        assert state == EpisodeState(
            TemporalStatus.COMMITTED, make_episode("e1").content_hash
        )
        tenant_ops = [e.op.value for e in await store.ledger_entries(TENANT_CHAIN)]
        assert "episode_retracted" not in tenant_ops
        history = await store.query(
            TemporalQuery(now=FixedClock().now(), include_history=True)
        )
        assert FactStatus.WITHDRAWN in [v.status for v in history]

    async def test_ledger_and_audit_hold_after_reset(self, make_harness):
        """Every chain verifies and the audit reports no violation."""
        harness = await self._seeded(make_harness)
        store = harness.store
        engine = engine_for(store)
        await engine.ingest(
            [works_at("alice", "acme", "e1", "Alice works at Acme")],
            make_episode("e1"),
        )
        await engine.reset_episode("e1", FakeMentions())

        assert verify(await store.ledger_entries(TENANT_CHAIN)) == []
        for slot_key in await store.slot_keys():
            assert verify(await store.ledger_entries(slot_chain(slot_key))) == []
        assert await audit(store) == []


class TestFalkorDBResetEpisode(ResetScenario):
    """The reset scenario on FalkorDB."""

    @pytest.fixture
    async def make_harness(self, falkordb_store: FalkorDBGraphStore):
        """Return a factory of harnesses on fresh tenants, erased afterwards."""
        tenants: list[str] = []

        async def make() -> TemporalHarness:
            handle = await falkordb_store.open(
                new_tenant(), [EPISODES, RESOLUTION, TEMPORAL]
            )
            tenants.append(handle.tenant)
            return graph_temporal_harness(
                GraphTemporalStore(falkordb_store, handle),
                falkordb_store,
                handle,
                tenant_keyed=False,
            )

        yield make
        for tenant in tenants:
            await falkordb_store.erase_tenant(tenant)


class TestNeo4jResetEpisode(ResetScenario):
    """The reset scenario on Neo4j."""

    @pytest.fixture
    async def make_harness(self, neo4j_store: Neo4jGraphStore):
        """Return a factory of harnesses on fresh tenants, erased afterwards."""
        tenants: list[str] = []

        async def make() -> TemporalHarness:
            handle = await neo4j_store.open(
                new_tenant(), [EPISODES, RESOLUTION, TEMPORAL]
            )
            tenants.append(handle.tenant)
            return graph_temporal_harness(
                GraphTemporalStore(neo4j_store, handle),
                neo4j_store,
                handle,
                tenant_keyed=True,
            )

        yield make
        for tenant in tenants:
            await neo4j_store.erase_tenant(tenant)


def _person(local_id: str, name: str, episode_id: str) -> ExtractedEntity:
    """Build a person as extraction hands it to resolution."""
    return ExtractedEntity(
        local_id=local_id,
        name=name,
        type_id=1,
        type_label="Person",
        confidence=0.9,
        method=Method.LLM,
        episode_id=episode_id,
    )


def _org(local_id: str, name: str, episode_id: str) -> ExtractedEntity:
    """Build an organization as extraction hands it to resolution."""
    return ExtractedEntity(
        local_id=local_id,
        name=name,
        type_id=2,
        type_label="Organization",
        confidence=0.9,
        method=Method.LLM,
        episode_id=episode_id,
    )


def resolution_policy() -> ResolutionPolicy:
    """Match people by name alone, since the scenario scripts no verifier."""
    return ResolutionPolicy(types={"Person": TypePolicy(mode=ResolutionMode.AUTO_KEY)})


async def build_resolver(resolver_store: ResolverStore) -> EntityResolver:
    """Build the entity resolver over one tenant's resolver store.

    Vectors are kept in memory on every backend, so only the store differs.
    """
    vectors = VectorStore(InMemoryBackend(), EmbeddingGateway(MockEmbedder()))
    await vectors.ensure_ready()
    return EntityResolver(
        resolver_store,
        vectors,
        resolution_policy(),
        ResolverSettings(inline_verification=False),
        FixedClock(),
    )


@dataclass
class ResolverStack:
    """A temporal harness and the entity resolver on the same tenant.

    Attributes:
        temporal: Harness whose store receives the facts.
        resolver: Resolver that links the entities of each episode.
        admit_resolved: Makes resolved entity ids known to the temporal store
            before a commit names them.
    """

    temporal: TemporalHarness
    resolver: EntityResolver
    admit_resolved: Callable[[Iterable[str]], Awaitable[None]]


async def _graph_holds_entities(entity_ids: Iterable[str]) -> None:
    """Skip seeding: resolved entities are already nodes of the graph."""


class _AppendFails:
    """Delegate every store call except `append_observations`, which raises."""

    def __init__(self, inner: TemporalStore) -> None:
        """Wrap the store whose observation writes fail."""
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        """Forward every other attribute to the wrapped store."""
        return getattr(self._inner, name)

    async def append_observations(self, observations: list[Observation]) -> int:
        """Fail the way a crash between the two writes would."""
        msg = "simulated crash after the mention write"
        raise RuntimeError(msg)


def _engine(store: TemporalStore, resolver: EntityResolver) -> TemporalEngine:
    """Build the engine on the single-valued policy, refs from the resolver."""
    return TemporalEngine(store, ONE, resolver.store, FixedClock().now)


async def _ref(resolver: EntityResolver, episode_id: str, local_id: str) -> str:
    """Return the stable entity id that an episode-local ref resolved to."""
    found = await resolver.store.entity_id_for_ref(episode_id, local_id)
    assert found is not None
    return found


async def _assert_chains_hold(store: TemporalStore) -> None:
    """Check that every ledger chain verifies and the audit is clean."""
    assert verify(await store.ledger_entries(TENANT_CHAIN)) == []
    for slot_key in await store.slot_keys():
        assert verify(await store.ledger_entries(slot_chain(slot_key))) == []
    assert await audit(store) == []


class ResolverResetScenario:
    """Reset an episode whose entities are linked by the real resolver."""

    async def _stack(self, make_stack: Callable[[], Awaitable[ResolverStack]]):
        stack = await make_stack()
        for episode_id in ("e1", "e2", "e3"):
            await stack.temporal.seed_episode(episode_id)
        return stack

    async def _ingest(
        self,
        stack: ResolverStack,
        store: TemporalStore,
        episode_id: str,
        facts: list[TypedFact],
    ) -> None:
        entity_ids = [e.entity_id for e in await stack.resolver.store.active_entities()]
        await stack.admit_resolved(entity_ids)
        await _engine(store, stack.resolver).ingest(facts, make_episode(episode_id))

    async def test_rerun_replaces_claim_and_keeps_other_episode(self, make_stack):
        """The rerun shows the new claim; the other episode's entities survive."""
        stack = await self._stack(make_stack)
        store, resolver = stack.temporal.store, stack.resolver
        await resolver.resolve(
            [_person("alice", "Alice", "e1"), _org("acme", "Acme", "e1")]
        )
        await self._ingest(
            stack, store, "e1", [works_at("alice", "acme", "e1", "Alice works at Acme")]
        )
        await resolver.resolve(
            [_person("bob", "Bob", "e2"), _org("acme", "Acme", "e2")]
        )
        await self._ingest(
            stack, store, "e2", [works_at("bob", "acme", "e2", "Bob works at Acme")]
        )
        alice = await _ref(resolver, "e1", "alice")
        acme = await _ref(resolver, "e1", "acme")
        bob = await _ref(resolver, "e2", "bob")
        assert await _current_pairs(store) == sorted([(alice, acme), (bob, acme)])

        with pytest.raises(EpisodeAlreadyResolvedError):
            await resolver.resolve(
                [_person("alice", "Alice", "e1"), _org("acme", "Acme", "e1")]
            )

        await _engine(store, resolver).reset_episode("e1", resolver)
        assert await resolver.store.entity_id_for_ref("e1", "alice") is None
        assert await _ref(resolver, "e2", "acme") == acme
        assert await _current_pairs(store) == [(bob, acme)]

        await resolver.resolve(
            [_person("alice", "Alice", "e1"), _org("beta", "Beta", "e1")]
        )
        await self._ingest(
            stack, store, "e1", [works_at("alice", "beta", "e1", "Alice works at Beta")]
        )
        alice = await _ref(resolver, "e1", "alice")
        beta = await _ref(resolver, "e1", "beta")
        assert await _current_pairs(store) == sorted([(alice, beta), (bob, acme)])
        state = await store.episode_state("e1")
        assert state == EpisodeState(
            TemporalStatus.COMMITTED, make_episode("e1").content_hash
        )
        tenant_ops = [e.op.value for e in await store.ledger_entries(TENANT_CHAIN)]
        assert "episode_retracted" not in tenant_ops
        await _assert_chains_hold(store)

    async def test_relabelled_alice_gets_a_new_entity(self, make_stack):
        """A new local id for the same person deletes the old entity."""
        stack = await self._stack(make_stack)
        store, resolver = stack.temporal.store, stack.resolver
        await resolver.resolve(
            [_person("alice", "Alice", "e1"), _org("acme", "Acme", "e1")]
        )
        await self._ingest(
            stack, store, "e1", [works_at("alice", "acme", "e1", "Alice works at Acme")]
        )
        old_alice = await _ref(resolver, "e1", "alice")

        await _engine(store, resolver).reset_episode("e1", resolver)
        await resolver.resolve(
            [_person("alice-2", "Alice", "e1"), _org("acme", "Acme", "e1")]
        )
        await self._ingest(
            stack,
            store,
            "e1",
            [works_at("alice-2", "acme", "e1", "Alice works at Acme")],
        )
        new_alice = await _ref(resolver, "e1", "alice-2")
        acme = await _ref(resolver, "e1", "acme")

        assert new_alice != old_alice
        assert await resolver.store.get_entities([old_alice]) == []
        assert await resolver.store.entity_id_for_ref("e1", "alice") is None
        assert await _current_pairs(store) == [(new_alice, acme)]
        await _assert_chains_hold(store)

    async def test_crash_after_mentions_is_cleared_by_reset(self, make_stack):
        """A reset clears mentions left by a crash before the observations."""
        stack = await self._stack(make_stack)
        store, resolver = stack.temporal.store, stack.resolver
        carol = [_person("carol", "Carol", "e3"), _org("acme", "Acme", "e3")]
        claim = [works_at("carol", "acme", "e3", "Carol works at Acme")]
        await resolver.resolve(carol)
        with pytest.raises(RuntimeError):
            await self._ingest(stack, _AppendFails(store), "e3", claim)

        with pytest.raises(EpisodeAlreadyResolvedError):
            await resolver.resolve(carol)
        await _engine(store, resolver).reset_episode("e3", resolver)
        assert await resolver.store.entity_id_for_ref("e3", "carol") is None

        await resolver.resolve(carol)
        await self._ingest(stack, store, "e3", claim)
        carol_id = await _ref(resolver, "e3", "carol")
        acme = await _ref(resolver, "e3", "acme")
        assert await _current_pairs(store) == [(carol_id, acme)]
        await _assert_chains_hold(store)


class TestFalkorDBResolverReset(ResolverResetScenario):
    """The resolver reset scenario on FalkorDB."""

    @pytest.fixture
    async def make_stack(self, falkordb_store: FalkorDBGraphStore):
        """Return a factory of resolver stacks on fresh tenants, erased after."""
        tenants: list[str] = []

        async def make() -> ResolverStack:
            handle = await falkordb_store.open(
                new_tenant(), [EPISODES, RESOLUTION, TEMPORAL]
            )
            tenants.append(handle.tenant)
            resolver = await build_resolver(
                GraphResolverStore(falkordb_store.runner(handle), tenant_keyed=False)
            )
            temporal = graph_temporal_harness(
                GraphTemporalStore(falkordb_store, handle),
                falkordb_store,
                handle,
                tenant_keyed=False,
            )
            return ResolverStack(temporal, resolver, _graph_holds_entities)

        yield make
        for tenant in tenants:
            await falkordb_store.erase_tenant(tenant)


class TestNeo4jResolverReset(ResolverResetScenario):
    """The resolver reset scenario on Neo4j."""

    @pytest.fixture
    async def make_stack(self, neo4j_store: Neo4jGraphStore):
        """Return a factory of resolver stacks on fresh tenants, erased after."""
        tenants: list[str] = []

        async def make() -> ResolverStack:
            handle = await neo4j_store.open(
                new_tenant(), [EPISODES, RESOLUTION, TEMPORAL]
            )
            tenants.append(handle.tenant)
            resolver = await build_resolver(
                GraphResolverStore(neo4j_store.runner(handle), tenant_keyed=True)
            )
            temporal = graph_temporal_harness(
                GraphTemporalStore(neo4j_store, handle),
                neo4j_store,
                handle,
                tenant_keyed=True,
            )
            return ResolverStack(temporal, resolver, _graph_holds_entities)

        yield make
        for tenant in tenants:
            await neo4j_store.erase_tenant(tenant)
