"""EpisodeStateStore operations for graph-backed temporal storage."""

import json
from collections.abc import Mapping
from typing import Any

from ontogr_mem.data_models.episode import TemporalStatus
from ontogr_mem.graphstore.cypher import TenantScope
from ontogr_mem.graphstore.executor import QuerySpec
from ontogr_mem.graphstore.protocol import ScopedRunner
from ontogr_mem.temporal.errors import MissingEpisodeError
from ontogr_mem.temporal.temporal_store import (
    EpisodeState,
    IncompleteEpisode,
    SourceSelector,
)

from .rows import _required
from .scope import (
    EPISODE,
    SOURCE,
)


class EpisodeStateStore:
    """Tenant-bound episode state persistence operations."""

    def __init__(self, runner: ScopedRunner, scope: TenantScope) -> None:
        """Bind storage operations to one tenant and runner."""
        self._runner = runner
        self._scope = scope

    def _map(self, body: str = "") -> str:
        return self._scope.pattern(body)

    async def episode_state(self, episode_id: str) -> EpisodeState | None:
        """Return the temporal state of the episode, or ``None`` if it has none."""
        rows = await self._runner.run(
            QuerySpec(
                f"MATCH (e:{EPISODE} {self._map('id: $id')}) "
                "RETURN e.temporal_status AS status, "
                "e.temporal_committed_hash AS committed_hash, "
                "e.temporal_ontology_refused AS ontology_refused, "
                "e.temporal_skipped AS skipped",
                {"id": episode_id},
                read_only=True,
                idempotent=True,
                tag="temporal_episode_state",
            )
        )
        if not rows.rows or rows.rows[0]["status"] is None:
            return None
        row = rows.rows[0]
        skipped = json.loads(row["skipped"]) if row["skipped"] is not None else None
        return EpisodeState(
            status=TemporalStatus(row["status"]),
            committed_hash=row["committed_hash"],
            ontology_refused=row["ontology_refused"],
            skipped=skipped,
        )

    async def set_episode_pending(
        self,
        episode_id: str,
        *,
        ontology_refused: int | None = None,
        skipped: Mapping[str, int] | None = None,
    ) -> None:
        """Mark the episode as pending and store any known item outcomes."""
        assignments = ["e.temporal_status = $status"]
        params: dict[str, Any] = {"status": TemporalStatus.PENDING.value}
        if ontology_refused is not None:
            assignments.append("e.temporal_ontology_refused = $ontology_refused")
            params["ontology_refused"] = ontology_refused
        if skipped is not None:
            assignments.append("e.temporal_skipped = $skipped")
            params["skipped"] = json.dumps(
                skipped, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            )
        await self._set_episode(
            episode_id,
            f"SET {', '.join(assignments)}",
            params,
        )

    async def mark_committed(self, episode_id: str, content_hash: str) -> None:
        """Mark the episode committed with the content hash that was read."""
        await self._set_episode(
            episode_id,
            "SET e.temporal_status = $status, e.temporal_committed_hash = $hash",
            {"status": TemporalStatus.COMMITTED.value, "hash": content_hash},
        )

    async def mark_retracted(self, episode_id: str) -> None:
        """Mark the episode retracted; its node stays.

        Raises:
            MissingEpisodeError: The episode does not exist.
        """
        await self._set_episode(
            episode_id,
            "SET e.temporal_status = $status",
            {"status": TemporalStatus.RETRACTED.value},
        )

    async def clear_episode_state(self, episode_id: str) -> None:
        """Clear the episode's temporal status and committed hash.

        The episode node and its content hash stay as they are.

        Raises:
            MissingEpisodeError: The episode does not exist.
        """
        await self._set_episode(
            episode_id,
            "REMOVE e.temporal_status, e.temporal_committed_hash, "
            "e.temporal_ontology_refused, e.temporal_skipped",
            {},
        )

    async def incomplete_episodes(self) -> list[IncompleteEpisode]:
        """Return every episode in ``PENDING`` state, sorted by episode id."""
        rows = await self._runner.run(
            QuerySpec(
                f"MATCH (e:{EPISODE} {self._map()}) "
                "WHERE e.temporal_status = $status "
                "RETURN e.id AS id, e.ingested_at AS ingested_at, "
                "e.content_hash AS content_hash ORDER BY e.id",
                {"status": TemporalStatus.PENDING.value},
                read_only=True,
                idempotent=True,
                tag="temporal_incomplete_episodes",
            )
        )
        return [
            IncompleteEpisode(
                episode_id=str(row["id"]),
                ingested_at=_required(row["ingested_at"]),
                content_hash=row["content_hash"],
            )
            for row in rows.rows
        ]

    async def episodes_matching(self, selector: SourceSelector) -> list[str]:
        """Return the ids of episodes whose source matches, sorted."""
        clauses = ["s.system = $system"]
        params: dict[str, Any] = {"system": selector.system}
        if selector.ref is not None:
            clauses.append("s.ref = $ref")
            params["ref"] = selector.ref
        if selector.connector is not None:
            clauses.append("s.connector = $connector")
            params["connector"] = selector.connector
        rows = await self._runner.run(
            QuerySpec(
                f"MATCH (e:{EPISODE} {self._map()})"
                f"-[:FROM_SOURCE {self._scope.edge()}]->(s:{SOURCE} {self._map()}) "
                f"WHERE {' AND '.join(clauses)} "
                "RETURN e.id AS id ORDER BY e.id",
                params,
                read_only=True,
                idempotent=True,
                tag="temporal_episodes_matching",
            )
        )
        return [str(row["id"]) for row in rows.rows]

    async def _set_episode(
        self, episode_id: str, assignments: str, params: dict[str, Any]
    ) -> None:
        rows = await self._runner.run(
            QuerySpec(
                f"MATCH (e:{EPISODE} {self._map('id: $id')}) {assignments} "
                "RETURN count(e) AS n",
                {"id": episode_id, **params},
                tag="temporal_set_episode",
            )
        )
        if int(rows.scalar("n", 0)) == 0:
            raise MissingEpisodeError(episode_id)
