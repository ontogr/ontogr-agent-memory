"""Tenant-owned entities, fact versions and episodes as JSON lines."""

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, TypedDict, overload

from ontogr_mem.data_models import Episode
from ontogr_mem.ingestion.store import EpisodeStore
from ontogr_mem.resolution.resolver_store import ResolverStore
from ontogr_mem.temporal.query import TemporalQuery
from ontogr_mem.temporal.reader import FactReader
from ontogr_mem.temporal.temporal_store import TemporalStore


SCHEMA_VERSION = 1
EPISODE_PAGE_SIZE = 100


class MetaRow(TypedDict):
    """Export identity and schema version."""

    type: Literal["meta"]
    schema: int
    tenant_id: str
    exported_at: str


class EntityRow(TypedDict):
    """One entity and its aliases."""

    type: Literal["entity"]
    id: str
    name: str
    type_label: str
    summary: str
    status: str
    merged_into: str | None
    aliases: list[str]


class FactRow(TypedDict):
    """One temporal fact version and its evidence references."""

    type: Literal["fact"]
    id: str
    version: int
    subject_id: str
    object_id: str
    relation_type_id: int
    statement: str
    source: str
    valid_from: str
    valid_to: str | None
    recorded_at: str
    expired_at: str | None
    status: str
    episode_ids: list[str]
    closed_by: str | None
    close_kind: str | None


class EpisodeSource(TypedDict):
    """Source fields carried by a stored episode."""

    system: str
    ref: str
    connector: str
    description: str


class EpisodeActor(TypedDict):
    """Actor fields carried by a stored episode."""

    id: str
    role: str
    display_name: str


class EpisodeRow(TypedDict):
    """One full stored episode, including its structured content."""

    type: Literal["episode"]
    id: str
    kind: str
    occurred_at: str
    ingested_at: str
    source: EpisodeSource
    actor: EpisodeActor
    content: object
    content_hash: str | None
    metadata: dict[str, object]


@dataclass(frozen=True, slots=True)
class TenantExport:
    """A deterministic JSONL document and counts for each row type.

    Attributes:
        jsonl: Newline-terminated JSON objects, one per export row.
        counts: Number of rows keyed by row type, including ``meta``.
    """

    jsonl: str
    counts: dict[str, int]


@overload
def _iso(value: datetime) -> str: ...


@overload
def _iso(value: datetime | None) -> str | None: ...


def _iso(value: datetime | None) -> str | None:
    """Format an optional timestamp as UTC ISO 8601 with microseconds."""
    return (
        value.astimezone(UTC).isoformat(timespec="microseconds")
        if value is not None
        else None
    )


async def _all_episodes(episodes: EpisodeStore) -> list[Episode]:
    """Read every episode through the store's newest-first keyset pages."""
    rows = []
    before = None
    while True:
        page = await episodes.list_episodes(limit=EPISODE_PAGE_SIZE, before=before)
        rows.extend(page)
        if len(page) < EPISODE_PAGE_SIZE:
            return rows
        last = page[-1]
        before = (last.ingested_at, last.id)


async def export_tenant(
    store: TemporalStore,
    resolver: ResolverStore,
    episodes: EpisodeStore,
    reader: FactReader,
    *,
    tenant_id: str,
    now: datetime,
) -> TenantExport:
    """Return every entity, fact version and episode belonging to a tenant.

    Args:
        store: Tenant-bound temporal store.
        resolver: Tenant-bound entity and alias store.
        episodes: Tenant-bound, paginated episode store.
        reader: Reader bound to the temporal store.
        tenant_id: Tenant id written in the metadata row.
        now: Export timestamp and present time for the fact query.

    Returns:
        Sorted JSON lines and counts for every row type.
    """
    aliases_by_entity: dict[str, list[str]] = defaultdict(list)
    for alias in await resolver.all_aliases():
        aliases_by_entity[alias.entity_id].append(alias.text)

    entities: list[EntityRow] = [
        {
            "type": "entity",
            "id": entity.entity_id,
            "name": entity.canonical_name,
            "type_label": entity.type_label,
            "summary": entity.summary,
            "status": entity.status.value,
            "merged_into": entity.merged_into,
            "aliases": sorted(aliases_by_entity[entity.entity_id]),
        }
        for entity in await resolver.all_entities()
    ]

    query = TemporalQuery(now=now, include_history=True)
    views = await reader.read(query)
    facts = await store.facts_by_ids(sorted({view.fact_id for view in views}))
    facts_by_key = {(fact.fact_id, fact.version): fact for fact in facts}
    observations = await reader.observation_texts(
        sorted({obs_id for view in views for obs_id in view.evidence})
    )
    views_by_key = {(view.fact_id, view.version): view for view in views}
    fact_rows: list[FactRow] = []
    for key in sorted(views_by_key):
        view = views_by_key[key]
        fact = facts_by_key[key]
        fact_rows.append(
            {
                "type": "fact",
                "id": view.fact_id,
                "version": view.version,
                "subject_id": view.subject_id,
                "object_id": view.object_id,
                "relation_type_id": view.relation_type_id,
                "statement": view.statement,
                "source": view.source,
                "valid_from": _iso(view.valid_from),
                "valid_to": _iso(view.valid_to),
                "recorded_at": _iso(view.recorded_at),
                "expired_at": _iso(view.expired_at),
                "status": view.status.value,
                "episode_ids": sorted(
                    {
                        observations[obs_id].episode_id
                        for obs_id in view.evidence
                        if obs_id in observations
                    }
                ),
                "closed_by": fact.closed_by,
                "close_kind": fact.close_kind.value if fact.close_kind else None,
            }
        )

    episode_rows = [_episode_row(episode) for episode in await _all_episodes(episodes)]
    episode_rows.sort(key=lambda row: row["id"])
    rows: list[MetaRow | EntityRow | FactRow | EpisodeRow] = [
        {
            "type": "meta",
            "schema": SCHEMA_VERSION,
            "tenant_id": tenant_id,
            "exported_at": _iso(now),
        },
        *sorted(entities, key=lambda row: row["id"]),
        *fact_rows,
        *episode_rows,
    ]
    counts = Counter(row["type"] for row in rows)
    jsonl = (
        "\n".join(json.dumps(row, sort_keys=True, ensure_ascii=False) for row in rows)
        + "\n"
    )
    return TenantExport(jsonl, dict(sorted(counts.items())))


def _episode_row(episode: Episode) -> EpisodeRow:
    """Copy one episode to its full public export row."""
    return {
        "type": "episode",
        "id": episode.id,
        "kind": episode.kind.value,
        "occurred_at": _iso(episode.occurred_at),
        "ingested_at": _iso(episode.ingested_at),
        "source": {
            "system": episode.source.system,
            "ref": episode.source.ref,
            "connector": episode.source.connector,
            "description": episode.source.description,
        },
        "actor": {
            "id": episode.actor.id,
            "role": episode.actor.role.value,
            "display_name": episode.actor.display_name,
        },
        "content": episode.content,
        "content_hash": episode.content_hash,
        "metadata": episode.metadata,
    }
