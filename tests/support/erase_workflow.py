"""Build a memory-backed Erase workflow scenario."""

from dataclasses import dataclass, replace
from datetime import UTC, datetime

from ontogr_mem.data_models import (
    Alias,
    AliasSource,
    DecisionMethod,
    Entity,
    Mention,
    Outcome,
)
from ontogr_mem.graphstore.memory.tenant_stacks import memory_stores
from ontogr_mem.llm.cache import MemoryResponseCache
from ontogr_mem.relational_store.memory import MemoryJobStore
from ontogr_mem.resolution.config import ResolverSettings
from ontogr_mem.resolution.ids import alias_id
from ontogr_mem.resolution.keys import loose_key, strict_key
from ontogr_mem.resolution.policy import ResolutionPolicy
from ontogr_mem.retrieval.settings import RetrievalSettings
from ontogr_mem.service.forget.plan import ForgetPlan, build_plan
from ontogr_mem.service.forget.selector import ForgetSelector
from ontogr_mem.service.relations import load_defaults
from ontogr_mem.service.runtime import TenantRuntime, build_tenant_runtime
from ontogr_mem.temporal import TemporalEngine
from ontogr_mem.vectorstore import Embedder
from ontogr_mem.vectorstore.episode_vectors import write_episode_vectors
from tests.support.clock import FixedClock
from tests.support.embedder import MockEmbedder
from tests.support.episodes import make_episode
from tests.unit.temporal.helpers import at, obs


NEEDLE = "Robert Smith works at Acme"


@dataclass
class EraseWorld:
    """The runtime, plan, journal and cache for one erase scenario."""

    runtime: TenantRuntime
    plan: ForgetPlan
    journal: MemoryJobStore
    cache: MemoryResponseCache
    needle: str


async def make_erase_world(*, cache: MemoryResponseCache | None = None) -> EraseWorld:
    """Create episodes, mentions, one Fact and their search vectors."""
    clock = FixedClock(datetime(2026, 10, 22, tzinfo=UTC))
    embedder: Embedder = MockEmbedder()
    stores = await memory_stores("erase-test", embedder)
    runtime = build_tenant_runtime(
        stores,
        llm=None,
        embedder=embedder,
        clock=clock,
        policy=ResolutionPolicy(),
        resolver_settings=ResolverSettings(),
        retrieval_settings=RetrievalSettings(),
    )
    ontology = await load_defaults(runtime.ontology)
    relation = ontology.relation_by_label("WORKS_AT")
    if relation is None:
        raise RuntimeError("default catalog does not define WORKS_AT")

    episodes = [
        make_episode("e1", content=NEEDLE),
        make_episode("e2", content="Bob moved to a new office"),
    ]
    await stores.episodes.write_episodes(episodes)
    for episode in episodes:
        if episode.content_hash is not None:
            await stores.episodes.record_content_hash(episode.id, episode.content_hash)

    person = _entity("person", "Robert Smith")
    organization = _entity("organization", "Acme", type_label="Organization")
    aliases = [
        _alias(person.entity_id, "Robert Smith", "e1"),
        _alias(person.entity_id, "Bob", "e2"),
        _alias(organization.entity_id, "Acme", "e1"),
    ]
    await stores.resolver_store.write_entity_batch([person, organization], aliases)
    await stores.resolver_store.write_mentions(
        "e1",
        [
            _mention("m-e1-person", person.entity_id, "e1", "Robert Smith"),
            _mention("m-e1-org", organization.entity_id, "e1", "Acme"),
        ],
    )
    await stores.resolver_store.write_mentions(
        "e2", [_mention("m-e2-person", person.entity_id, "e2", "Bob")]
    )

    observation = obs(
        "obs-e1",
        organization.entity_id,
        at(2020),
        subject=person.entity_id,
        slot="work",
        episode="e1",
        rel=relation.type_id,
    )
    observation = replace(observation, statement=NEEDLE, quote=NEEDLE)
    engine: TemporalEngine = runtime.engine(ontology, runtime.resolver.store)
    await engine.ingest_observations([observation])
    await write_episode_vectors(runtime.writer, episodes)
    for entity in (person, organization):
        await runtime.resolver.embed(entity)

    selected_cache = cache if cache is not None else MemoryResponseCache()
    plan = await build_plan(
        ForgetSelector(episode_ids=("e1",)),
        episodes=stores.episodes,
        temporal=stores.temporal_store,
        resolver=stores.resolver_store,
    )
    return EraseWorld(
        runtime,
        plan,
        MemoryJobStore(),
        selected_cache,
        NEEDLE,
    )


def _entity(entity_id: str, name: str, *, type_label: str = "Person") -> Entity:
    """Create an entity with text fields that Erase must clear."""
    return Entity(
        entity_id=entity_id,
        type_label=type_label,
        canonical_name=name,
        strict_key=strict_key(name),
        loose_key=loose_key(name),
        summary=f"Summary mentioning {name}",
        descriptions=[f"Description mentioning {name}"],
        attributes={"note": f"Note mentioning {name}"},
        created_at=datetime(2026, 10, 22, tzinfo=UTC),
        updated_at=datetime(2026, 10, 22, tzinfo=UTC),
    )


def _alias(entity_id: str, text: str, episode_id: str) -> Alias:
    """Create a name alias first seen in one Episode."""
    key = strict_key(text)
    return Alias(
        alias_id=alias_id(entity_id, key),
        entity_id=entity_id,
        text=text,
        strict_key=key,
        loose_key=loose_key(text),
        source=AliasSource.EXTRACTED,
        first_episode_id=episode_id,
    )


def _mention(
    mention_id: str, entity_id: str, episode_id: str, surface_form: str
) -> Mention:
    """Create a stored mention used by resolver cleanup."""
    return Mention(
        mention_id=mention_id,
        entity_id=entity_id,
        episode_id=episode_id,
        local_id=mention_id,
        surface_form=surface_form,
        type_label_at_extraction="Person",
        resolver_version="test",
        decision_method=DecisionMethod.KEY,
        decision_outcome=Outcome.CREATED,
        quote=surface_form,
    )
