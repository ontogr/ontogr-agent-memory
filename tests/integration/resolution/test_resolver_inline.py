"""Behavior tests for the resolver on a live graph tenant."""

import asyncio
import math

import pytest

from ontogr_mem.data_models import SystemClock
from ontogr_mem.data_models.entities import (
    Alias,
    AliasSource,
    DecisionMethod,
    Entity,
    Outcome,
    ResolutionState,
)
from ontogr_mem.resolution.candidates import candidate_pass
from ontogr_mem.resolution.config import ResolverSettings
from ontogr_mem.resolution.errors import EpisodeAlreadyResolvedError
from ontogr_mem.resolution.grouping import build_groups
from ontogr_mem.resolution.ids import entity_id
from ontogr_mem.resolution.keys import entity_text
from ontogr_mem.resolution.policy import ResolutionPolicy, TypePolicy
from ontogr_mem.resolution.resolver import EntityResolver
from ontogr_mem.resolution.resolver_store import ResolverStore
from ontogr_mem.vectorstore import (
    EmbeddingGateway,
    EmbedItem,
    VectorKind,
    VectorStore,
    VectorStoreError,
)
from ontogr_mem.vectorstore.memory import InMemoryBackend
from tests.integration.resolution.resolver_helpers import count_active, ex
from tests.support.clock import FixedClock
from tests.support.embedder import MockEmbedder
from tests.support.graph_backend import GraphBackend
from tests.unit.resolution.helpers import OracleLLM


NAMES = (
    "Quasar",
    "Nimbus",
    "Vortex",
    "Zephyr",
    "Onyx",
    "Cobalt",
    "Delta",
    "Fjord",
    "Glyph",
    "Harbor",
    "Ivory",
    "Jasper",
)
QUERY = "Manganese"


async def make(
    graph_backend: GraphBackend,
    store: ResolverStore | None = None,
    *,
    llm: object | None = None,
    inline: bool = False,
    clock: object | None = None,
    policy: ResolutionPolicy | None = None,
    vectors: VectorStore | None = None,
    **settings: object,
) -> EntityResolver:
    """Build a resolver with a deterministic embedder."""
    chosen = store or (await graph_backend.open_stores()).resolver_store
    if vectors is None:
        vectors = VectorStore(InMemoryBackend(), EmbeddingGateway(MockEmbedder()))
        await vectors.ensure_ready()
    return EntityResolver(
        chosen,
        vectors,
        policy or ResolutionPolicy(),
        ResolverSettings(inline_verification=inline, **settings),  # type: ignore[arg-type]
        clock or SystemClock(),  # type: ignore[arg-type]
        llm=llm,  # type: ignore[arg-type]
    )


def _rendered(text: str) -> str:
    return " ".join(text.split())


async def _ranked_store(
    graph_backend: GraphBackend,
) -> tuple[ResolverStore, VectorStore]:
    """Seed twelve entities whose vectors clear the ann floor in id order."""
    clock = FixedClock()
    store = (await graph_backend.open_stores()).resolver_store
    entities: list[Entity] = []
    vectors: dict[str, list[float]] = {
        _rendered(f"{QUERY} | Company | "): [1.0, 0.0, 0.0]
    }
    for index, name in enumerate(NAMES):
        entity = Entity(
            entity_id=f"c{index:02}",
            type_label="Company",
            canonical_name=name,
            strict_key=name.casefold(),
            loose_key=name.casefold(),
            created_at=clock.now(),
            updated_at=clock.now(),
        )
        entities.append(entity)
        cosine = 0.99 - index * 0.005
        vectors[_rendered(entity_text(entity))] = [
            cosine,
            math.sqrt(max(0.0, 1.0 - cosine * cosine)),
            0.0,
        ]
    await store.write_entity_batch(entities, [])
    gateway = EmbeddingGateway(MockEmbedder(vectors=vectors))
    vector_store = VectorStore(InMemoryBackend(), gateway)
    await vector_store.ensure_ready()
    await vector_store.upsert(
        VectorKind.NODE,
        [EmbedItem(entity.entity_id, entity_text(entity)) for entity in entities],
    )
    return store, vector_store


async def test_same_name_twice_gives_one_entity_two_mentions_and_a_key_decision(
    graph_backend: GraphBackend,
):
    """A later mention with the same key links to the existing entity."""
    store = (await graph_backend.open_stores()).resolver_store
    resolver = await make(graph_backend, store)

    first = await resolver.resolve([ex("1", "Acme Corp")])
    second = await resolver.resolve([ex("1", "acme  corp", episode_id="ep2")])

    assert first.entities[0].entity_id == second.entities[0].entity_id
    assert second.decisions[0].method is DecisionMethod.KEY
    assert second.decisions[0].outcome is Outcome.MATCHED
    [entity] = await store.active_entities()
    assert entity.mention_count == 2


async def test_two_mentions_in_one_batch_collapse_into_one_entity(
    graph_backend: GraphBackend,
):
    """Same-key mentions in one batch become one entity with two mentions."""
    store = (await graph_backend.open_stores()).resolver_store
    resolver = await make(graph_backend, store)

    result = await resolver.resolve(
        [ex("1", "Acme"), ex("2", "acme", episode_id="ep2")]
    )

    assert len(result.entities) == 1
    assert len(result.mentions) == 2
    assert await count_active(store) == 1


async def test_incompatible_types_with_one_name_stay_separate(
    graph_backend: GraphBackend,
):
    """Two groups with unrelated types never collapse."""
    store = (await graph_backend.open_stores()).resolver_store
    resolver = await make(graph_backend, store)

    result = await resolver.resolve(
        [ex("1", "Java", "Language"), ex("2", "Java", "City")]
    )

    assert result.entities[0].entity_id != result.entities[1].entity_id
    assert await count_active(store) == 2


async def test_java_language_never_matches_java_place_across_batches(
    graph_backend: GraphBackend,
):
    """A place does not match a language with the same name."""
    store = (await graph_backend.open_stores()).resolver_store
    resolver = await make(graph_backend, store)

    first = await resolver.resolve([ex("1", "Java", "City")])
    second = await resolver.resolve([ex("1", "Java", "Language", episode_id="ep2")])

    assert first.entities[0].entity_id != second.entities[0].entity_id
    assert second.decisions[0].outcome is Outcome.CREATED


async def test_person_family_key_hit_is_created_provisional_not_merged(
    graph_backend: GraphBackend,
):
    """A person key hit needs a verdict, so it stays provisional."""
    store = (await graph_backend.open_stores()).resolver_store
    resolver = await make(graph_backend, store)

    first = await resolver.resolve([ex("1", "John Smith", "Person")])
    second = await resolver.resolve([ex("1", "John Smith", "Person", episode_id="ep2")])

    assert first.entities[0].entity_id != second.entities[0].entity_id
    decision = second.decisions[0]
    assert decision.outcome is Outcome.CREATED_PROVISIONAL
    assert decision.reason == "verify_always"
    assert decision.candidates == (first.entities[0].entity_id,)
    created = next(
        entity
        for entity in await store.active_entities()
        if entity.entity_id == second.entities[0].entity_id
    )
    assert created.resolution_state is ResolutionState.PROVISIONAL


async def test_equal_identity_attribute_matches_a_person_automatically(
    graph_backend: GraphBackend,
):
    """An equal identity attribute settles a key hit."""
    policy = ResolutionPolicy(types={"Person": TypePolicy(identity_attrs=("email",))})
    store = (await graph_backend.open_stores()).resolver_store
    resolver = await make(graph_backend, store, policy=policy)

    first = await resolver.resolve([ex("1", "John Smith", "Person", email="J@x.org")])
    other = await resolver.resolve(
        [ex("1", "Johnny S", "Person", episode_id="ep2", email="j@x.org")]
    )
    third = await resolver.resolve(
        [ex("1", "John Smith", "Person", episode_id="ep3", email="j@x.org")]
    )

    assert first.entities[0].entity_id != other.entities[0].entity_id
    assert third.entities[0].entity_id == first.entities[0].entity_id
    assert third.decisions[0].reason == "identity_attr"


async def test_conflicting_identity_attribute_vetoes_a_key_hit(
    graph_backend: GraphBackend,
):
    """A conflicting identity attribute turns a key hit into a new entity."""
    policy = ResolutionPolicy(types={"Person": TypePolicy(identity_attrs=("email",))})
    store = (await graph_backend.open_stores()).resolver_store
    resolver = await make(graph_backend, store, policy=policy)

    first = await resolver.resolve([ex("1", "John Smith", "Person", email="a@x.org")])
    second = await resolver.resolve(
        [ex("1", "John Smith", "Person", episode_id="ep2", email="b@x.org")]
    )

    assert first.entities[0].entity_id != second.entities[0].entity_id
    assert second.decisions[0].outcome is Outcome.CREATED
    assert second.decisions[0].reason == "no_candidates"


async def test_digit_only_difference_never_auto_merges(graph_backend: GraphBackend):
    """Two names that differ only in a number stay apart."""
    store = (await graph_backend.open_stores()).resolver_store
    resolver = await make(graph_backend, store)

    first = await resolver.resolve([ex("1", "Quarterly Report Q3 2023", "Work")])
    second = await resolver.resolve(
        [ex("1", "Quarterly Report Q3 2024", "Work", episode_id="ep2")]
    )

    assert first.entities[0].entity_id != second.entities[0].entity_id
    assert second.decisions[0].outcome is Outcome.CREATED


async def test_fuzzy_hyphen_variant_matches_with_a_jaccard_score(
    graph_backend: GraphBackend,
):
    """A punctuation variant keeps the same blocking key and matches."""
    store = (await graph_backend.open_stores()).resolver_store
    resolver = await make(graph_backend, store)

    first = await resolver.resolve([ex("1", "Joe Michaels", "Work")])
    second = await resolver.resolve([ex("1", "Joe-Michaels", "Work", episode_id="ep2")])

    assert first.entities[0].entity_id == second.entities[0].entity_id
    decision = second.decisions[0]
    assert decision.method is DecisionMethod.FUZZY
    assert decision.score == 1.0


async def test_near_name_below_jaccard_threshold_is_not_merged(
    graph_backend: GraphBackend,
):
    """A Jaccard value below the fuzzy bar leaves the entity separate."""
    store = (await graph_backend.open_stores()).resolver_store
    resolver = await make(graph_backend, store)

    first = await resolver.resolve([ex("1", "Joe Michaels", "Work")])
    second = await resolver.resolve([ex("1", "Joe Michael", "Work", episode_id="ep2")])

    assert first.entities[0].entity_id != second.entities[0].entity_id


async def test_short_low_entropy_names_are_exact_only(graph_backend: GraphBackend):
    """A short name never takes part in fuzzy matching."""
    store = (await graph_backend.open_stores()).resolver_store
    resolver = await make(graph_backend, store)

    first = await resolver.resolve([ex("1", "Sam", "Work")])
    second = await resolver.resolve([ex("1", "Sammy", "Work", episode_id="ep2")])

    assert first.entities[0].entity_id != second.entities[0].entity_id


async def test_unknown_candidate_type_is_promoted_to_the_specific_type(
    graph_backend: GraphBackend,
):
    """A matching mention with a specific type promotes the entity."""
    store = (await graph_backend.open_stores()).resolver_store
    resolver = await make(graph_backend, store)

    first = await resolver.resolve([ex("1", "Berlin", "Unknown")])
    second = await resolver.resolve([ex("1", "Berlin", "City", episode_id="ep2")])

    assert first.entities[0].entity_id == second.entities[0].entity_id
    [entity] = await store.active_entities()
    assert entity.type_label == "City"


async def test_ontology_uri_match_beats_name(graph_backend: GraphBackend):
    """A shared uri wins before any name lookup."""
    store = (await graph_backend.open_stores()).resolver_store
    resolver = await make(graph_backend, store)

    first = await resolver.resolve([ex("1", "Big Apple", "City", uri="http://x/nyc")])
    second = await resolver.resolve(
        [
            ex(
                "1",
                "New York City",
                "City",
                episode_id="ep2",
                uri="http://x/nyc",
            )
        ]
    )

    assert first.entities[0].entity_id == second.entities[0].entity_id
    assert second.decisions[0].method is DecisionMethod.URI


async def test_surface_forms_become_aliases_and_resolve_next_time(
    graph_backend: GraphBackend,
):
    """An alias written for an entity resolves a later mention."""
    store = (await graph_backend.open_stores()).resolver_store
    resolver = await make(graph_backend, store)
    first = await resolver.resolve([ex("1", "New York City", "City")])
    target = first.entities[0].entity_id
    await store.write_entity_batch(
        [],
        [
            Alias(
                alias_id="x",
                entity_id=target,
                text="NYC",
                strict_key="nyc",
                loose_key="nyc",
                source=AliasSource.EXTRACTED,
            )
        ],
    )

    second = await resolver.resolve([ex("1", "NYC", "City", episode_id="ep3")])

    assert second.entities[0].entity_id == target
    assert second.decisions[0].method is DecisionMethod.ALIAS


async def test_concurrent_resolves_of_one_new_name_end_with_one_entity(
    graph_backend: GraphBackend,
):
    """Two resolvers on one store serialize and leave one entity."""
    store = (await graph_backend.open_stores()).resolver_store
    first = await make(graph_backend, store)
    second = await make(graph_backend, store)

    await asyncio.gather(
        first.resolve([ex("1", "Acme")]),
        second.resolve([ex("1", "Acme", episode_id="ep2")]),
    )

    assert await count_active(store) == 1


async def test_empty_name_is_skipped_and_empty_batch_is_a_noop(
    graph_backend: GraphBackend,
):
    """Empty names and empty batches write nothing."""
    store = (await graph_backend.open_stores()).resolver_store
    resolver = await make(graph_backend, store)

    assert (await resolver.resolve([])).entities == []
    assert (await resolver.resolve([ex("1", "   ")])).entities == []
    assert await count_active(store) == 0


async def test_same_local_id_from_different_episodes_stay_apart(
    graph_backend: GraphBackend,
):
    """A local id is only unique inside its episode."""
    store = (await graph_backend.open_stores()).resolver_store
    resolver = await make(graph_backend, store)

    result = await resolver.resolve(
        [
            ex("1", "Acme", episode_id="a"),
            ex("1", "Globex", episode_id="b"),
        ]
    )

    assert result.entities[0].entity_id != result.entities[1].entity_id
    assert len(result.decisions) == 2
    assert await count_active(store) == 2


async def test_two_fresh_tenants_write_byte_identical_ids_with_a_fixed_clock(
    graph_backend: GraphBackend,
):
    """Deterministic ids and a fixed clock make two tenants write the same rows."""
    clock = FixedClock()
    first = await make(
        graph_backend, (await graph_backend.open_stores()).resolver_store, clock=clock
    )
    second = await make(
        graph_backend, (await graph_backend.open_stores()).resolver_store, clock=clock
    )

    one = await first.resolve([ex("1", "Acme", "Company", episode_id="ep1")])
    two = await second.resolve([ex("1", "Acme", "Company", episode_id="ep1")])

    assert one.entities == two.entities
    assert one.aliases == two.aliases
    assert one.mentions == two.mentions


async def test_entity_id_is_the_literal_uuid5_of_the_episode_and_local_id(
    graph_backend: GraphBackend,
):
    """The created entity id is pinned to the literal UUID."""
    resolver = await make(
        graph_backend,
    )

    result = await resolver.resolve([ex("e0", "Acme", "Company", episode_id="ep1")])

    assert result.entities[0].entity_id == entity_id("ep1", "e0")
    assert result.entities[0].entity_id == "408424a3-4698-5fec-b7d9-2d221ebd4221"


async def test_second_resolve_over_an_episode_raises_and_writes_nothing(
    graph_backend: GraphBackend,
):
    """Mentions are the commit marker, so a second run is refused."""
    store = (await graph_backend.open_stores()).resolver_store
    resolver = await make(graph_backend, store)
    await resolver.resolve([ex("1", "Acme")])

    with pytest.raises(EpisodeAlreadyResolvedError) as raised:
        await resolver.resolve([ex("2", "Globex")])

    assert raised.value.episode_id == "ep1"
    assert await count_active(store) == 1
    assert await store.episode_resolution_state("ep1")


async def test_vector_search_failure_aborts_the_episode_untouched(
    graph_backend: GraphBackend, monkeypatch
):
    """A vector error is not treated as an empty candidate list."""
    store = (await graph_backend.open_stores()).resolver_store
    resolver = await make(graph_backend, store)
    await resolver.resolve([ex("1", "Acme")])

    async def boom(*args: object, **kwargs: object) -> list[object]:
        raise VectorStoreError("down")

    monkeypatch.setattr(resolver.vectors, "search", boom)

    with pytest.raises(VectorStoreError):
        await resolver.resolve([ex("1", "Globex", episode_id="ep2")])

    assert await count_active(store) == 1
    assert not await store.episode_resolution_state("ep2")


async def test_candidate_cap_and_verifier_order_follow_rank_key(
    graph_backend: GraphBackend,
):
    """Ten candidates survive the cap and the best three reach the verifier."""
    store, vector_store = await _ranked_store(graph_backend)
    seen: list[str] = []

    def record(name_a: str, name_b: str) -> bool:
        seen.append(name_b)
        return False

    resolver = await make(
        graph_backend, store, vectors=vector_store, inline=True, llm=OracleLLM(record)
    )
    groups = build_groups(
        [ex("1", QUERY, "Company")], resolver.policy, resolver.relations
    )
    candidates, _ = await candidate_pass(
        groups,
        store=store,
        vectors=vector_store,
        policy=resolver.policy,
        settings=resolver.settings,
    )

    assert len(candidates[0]) == 10
    assert [candidate.entity_id for candidate in candidates[0][:3]] == [
        "c00",
        "c01",
        "c02",
    ]

    result = await resolver.resolve([ex("1", QUERY, "Company")])

    assert seen == ["Quasar", "Nimbus", "Vortex"]
    assert result.decisions[0].outcome is Outcome.CREATED_PROVISIONAL
    assert result.decisions[0].reason == "unverified"
    assert result.decisions[0].candidates == (
        "c03",
        "c04",
        "c05",
        "c06",
        "c07",
        "c08",
        "c09",
    )
