"""Builders and per-backend hooks for temporal store tests."""

from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Any

from ontogr_mem.data_models.entities import Entity
from ontogr_mem.graphstore import GraphHandle, QuerySpec
from ontogr_mem.graphstore.episode_store import GraphEpisodeStore
from ontogr_mem.graphstore.protocol import GraphStore
from ontogr_mem.graphstore.resolver_store import GraphResolverStore
from ontogr_mem.graphstore.temporal.store import GraphTemporalStore
from ontogr_mem.ingestion.store import EpisodeStore
from ontogr_mem.temporal.temporal_store import TemporalStore
from tests.support.episodes import graph_harness, make_episode
from tests.support.graph_backend import GraphBackend


@dataclass
class TemporalHarness:
    """A temporal store with the hooks the contract needs.

    Attributes:
        store: The store under test, bound to one tenant.
        seed_entities: Make entity nodes exist so facts may name them.
        seed_episode: Make an episode exist so its temporal state can be set.
        rewrite_episode: Write the episode again, as ingestion does on re-read.
    """

    store: TemporalStore
    seed_entities: Callable[[Iterable[str]], Awaitable[None]]
    seed_episode: Callable[[str], Awaitable[None]]
    rewrite_episode: Callable[[str], Awaitable[None]]
    raw_slot_properties: Callable[[str], Awaitable[dict[str, Any]]] | None = None


def graph_temporal_harness(
    store: TemporalStore,
    graph: GraphStore,
    handle: GraphHandle,
    *,
    tenant_keyed: bool,
) -> TemporalHarness:
    """Build a harness for a graph backend around a tenant-bound store."""
    episodes = graph_harness(graph.runner(handle), tenant_keyed=tenant_keyed).store
    resolver = GraphResolverStore(graph.runner(handle), tenant_keyed=tenant_keyed)

    async def seed_entities(entity_ids: Iterable[str]) -> None:
        for entity_id in entity_ids:
            await resolver.save_entity(
                Entity(
                    entity_id=entity_id,
                    type_label="Thing",
                    canonical_name=entity_id,
                    strict_key=entity_id,
                    loose_key=entity_id,
                    created_at=make_episode("entity-time").occurred_at,
                    updated_at=make_episode("entity-time").occurred_at,
                )
            )

    async def seed_episode(episode_id: str) -> None:
        await episodes.write_episodes([make_episode(episode_id)])

    async def rewrite_episode(episode_id: str) -> None:
        await episodes.write_episodes([make_episode(episode_id)])

    async def raw_slot_properties(slot_key: str) -> dict[str, Any]:
        pattern = "{_tenant: $tenant, id: $slot}" if tenant_keyed else "{id: $slot}"
        rows = await graph.runner(handle).run(
            QuerySpec(
                f"MATCH (s:Slot {pattern}) RETURN properties(s) AS props",
                {"slot": slot_key},
                read_only=True,
                idempotent=True,
                tag="test_raw_slot_properties",
            )
        )
        return dict(rows.rows[0]["props"]) if rows.rows else {}

    return TemporalHarness(
        store, seed_entities, seed_episode, rewrite_episode, raw_slot_properties
    )


async def open_temporal(
    backend: GraphBackend,
    entity_ids: Iterable[str] = (),
    store_type: type[GraphTemporalStore] = GraphTemporalStore,
    **kwargs: Any,
) -> tuple[EpisodeStore, GraphTemporalStore]:
    """Open a tenant and return its episode store and a temporal store over it.

    Args:
        backend: The live graph backend to open the tenant on.
        entity_ids: Entities that exist before the test writes facts.
        store_type: The temporal store class, for tests that inject faults.
        **kwargs: Extra constructor arguments for `store_type`.
    """
    handle = await backend.open_tenant()
    runner = backend.store.runner(handle)
    episodes = GraphEpisodeStore(runner, tenant_keyed=backend.tenant_keyed)
    resolver = GraphResolverStore(runner, tenant_keyed=backend.tenant_keyed)
    for entity_id in entity_ids:
        await resolver.save_entity(
            Entity(
                entity_id=entity_id,
                type_label="Thing",
                canonical_name=entity_id,
                strict_key=entity_id,
                loose_key=entity_id,
                created_at=make_episode("entity-time").occurred_at,
                updated_at=make_episode("entity-time").occurred_at,
            )
        )
    store = store_type(
        backend.store, handle, tenant_keyed=backend.tenant_keyed, **kwargs
    )
    return episodes, store
