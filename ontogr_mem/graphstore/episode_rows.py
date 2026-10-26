"""Conversion between `Episode` and the properties stored for it."""

import json
from collections.abc import Mapping
from typing import Any

from ontogr_mem.data_models import (
    Actor,
    Episode,
    Kind,
    Role,
    Source,
    TemporalStatus,
    from_micros,
    to_micros,
)
from ontogr_mem.data_models.episode import EpisodeMeta


def episode_meta_params(meta: EpisodeMeta) -> dict[str, Any]:
    """Return metadata parameters with fact ids stored as canonical JSON."""
    return {
        "id": meta.episode_id,
        "session_id": meta.session_id,
        "thread": meta.thread,
        "subject_ref": meta.subject_ref,
        "fact_ids": _dumps(list(meta.fact_ids)),
    }


def episode_meta_from_row(row: Mapping[str, Any]) -> EpisodeMeta:
    """Rebuild episode metadata, preserving the order of fact ids."""
    return EpisodeMeta(
        episode_id=row["id"],
        session_id=row["session_id"],
        thread=row["thread"],
        subject_ref=row["subject_ref"],
        fact_ids=tuple(json.loads(row["fact_ids"])),
    )


def episode_params(episode: Episode) -> dict[str, Any]:
    """Return the statement parameters that store `episode`.

    Times are integer microseconds. The content is the text of the episode and
    the metadata is canonical JSON.
    """
    is_text = isinstance(episode.content, str)
    return {
        "id": episode.id,
        "kind": episode.kind.value,
        "occurred_at": to_micros(episode.occurred_at),
        "ingested_at": to_micros(episode.ingested_at),
        "content": episode.text,
        "content_format": "text" if is_text else "json",
        "metadata": _dumps(episode.metadata),
        "source_id": episode.source.id,
        "source_system": episode.source.system,
        "source_ref": episode.source.ref,
        "source_connector": episode.source.connector,
        "source_description": episode.source.description,
        "actor_id": episode.actor.id,
        "actor_role": episode.actor.role.value,
        "actor_name": episode.actor.display_name,
    }


def episode_from_row(row: Mapping[str, Any]) -> Episode:
    """Rebuild an `Episode` from a row of `EPISODE_COLUMNS`."""
    occurred_at = from_micros(row["occurred_at"])
    ingested_at = from_micros(row["ingested_at"])
    if occurred_at is None or ingested_at is None:
        msg = "stored episode has an open time"
        raise ValueError(msg)
    content = row["content"]
    status = row["temporal_status"]
    return Episode(
        id=row["id"],
        kind=Kind(row["kind"]),
        occurred_at=occurred_at,
        ingested_at=ingested_at,
        source=Source(
            system=row["source_system"],
            ref=row["source_ref"],
            connector=row["source_connector"],
            description=row["source_description"],
        ),
        actor=Actor(
            id=row["actor_id"],
            role=Role(row["actor_role"]),
            display_name=row["actor_name"],
        ),
        content=content if row["content_format"] == "text" else json.loads(content),
        content_hash=row["content_hash"],
        metadata=json.loads(row["metadata"]),
        temporal_status=TemporalStatus(status) if status else None,
        temporal_committed_hash=row["temporal_committed_hash"],
    )


def _dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
