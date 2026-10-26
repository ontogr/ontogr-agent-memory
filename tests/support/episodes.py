"""Builders and per-backend hooks for episode store tests."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from ontogr_mem.data_models import (
    Actor,
    Episode,
    Kind,
    Role,
    Source,
    TemporalStatus,
)
from ontogr_mem.graphstore import QuerySpec, ScopedRunner
from ontogr_mem.graphstore.episode_store import GraphEpisodeStore
from ontogr_mem.ingestion import EpisodeStore


T0 = datetime(2026, 10, 8, 12, 0, tzinfo=UTC)


def make_episode(
    episode_id: str,
    *,
    minutes: int = 0,
    ref: str | None = None,
    actor: str = "u1",
    content: Any = "hello",
    ingested_minutes: int = 0,
    metadata: dict[str, Any] | None = None,
) -> Episode:
    """Build an episode that occurred `minutes` after `T0`."""
    return Episode(
        id=episode_id,
        kind=Kind.MESSAGE if isinstance(content, str) else Kind.EVENT,
        occurred_at=T0 + timedelta(minutes=minutes, microseconds=7),
        ingested_at=T0 + timedelta(minutes=ingested_minutes, microseconds=3),
        source=Source("chat", ref or f"ref-{episode_id}", "chat", "session s1"),
        actor=Actor(actor, Role.USER, "Ana"),
        content=content,
        content_hash=f"hash-{episode_id}",
        metadata=metadata if metadata is not None else {"session_id": "s1"},
    )


@dataclass
class Harness:
    """An episode store with the hooks the contract needs."""

    store: EpisodeStore
    set_temporal_state: Callable[[str, TemporalStatus, str], Awaitable[None]]
    node_ids: Callable[[str], Awaitable[set[str]]]


def graph_harness(runner: ScopedRunner, *, tenant_keyed: bool) -> Harness:
    """Build a harness for a graph backend from its tenant-bound runner."""
    key = "_tenant: $tenant, " if tenant_keyed else ""

    async def set_temporal_state(
        episode_id: str, status: TemporalStatus, committed_hash: str
    ) -> None:
        await runner.run(
            QuerySpec(
                f"MATCH (e:Episode {{{key}id: $id}}) "
                "SET e.temporal_status = $status, "
                "e.temporal_committed_hash = $hash",
                {"id": episode_id, "status": status.value, "hash": committed_hash},
            )
        )

    async def node_ids(label: str) -> set[str]:
        scope = "{_tenant: $tenant}" if tenant_keyed else ""
        rows = await runner.run(
            QuerySpec(f"MATCH (n:{label} {scope}) RETURN n.id AS id", read_only=True)
        )
        return {row["id"] for row in rows.rows}

    return Harness(
        GraphEpisodeStore(runner, tenant_keyed=tenant_keyed),
        set_temporal_state,
        node_ids,
    )
