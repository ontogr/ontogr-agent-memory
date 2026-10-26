"""Read complete temporal snapshots through tenant-bound collaborators."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from ontogr_mem.data_models.fact_versions import Fact, FactStatus, Supersession
from ontogr_mem.data_models.observation import Observation
from ontogr_mem.data_models.timestamps import OPEN_MICROS
from ontogr_mem.graphstore.cypher import TenantScope
from ontogr_mem.graphstore.executor import QuerySpec
from ontogr_mem.graphstore.protocol import ScopedRunner
from ontogr_mem.temporal.query import FactView, TemporalQuery, matches, view_of
from ontogr_mem.temporal.reconcile import Conflict
from ontogr_mem.temporal.temporal_store import (
    TemporalSnapshot,
)

from .rows import (
    _CONFLICT_COLUMNS,
    _FACT_COLUMNS,
    _OBSERVATION_COLUMNS,
    _SUPERSESSION_COLUMNS,
    _columns,
    _conflict_from,
    _fact_from,
    _observation_from,
    _supersession_from,
)
from .scope import (
    CONFLICT,
    ENTITY,
    FACT,
    OBSERVATION,
    SLOT,
    SUPERSESSION,
)
from .verdict_store import VerdictStore


async def snapshot(slots: SnapshotReader, verdicts: VerdictStore) -> TemporalSnapshot:
    """Read every temporal row concurrently, retaining each collection's order."""
    facts_task = asyncio.create_task(slots.all_facts())
    observations_task = asyncio.create_task(slots.all_observations())
    supersessions_task = asyncio.create_task(slots.all_supersessions())
    verdicts_task = asyncio.create_task(verdicts.all_verdicts())
    decisions_task = asyncio.create_task(verdicts.all_decisions())
    conflicts_task = asyncio.create_task(slots.all_conflicts())
    slot_keys_task = asyncio.create_task(slots.slot_keys())
    await asyncio.gather(
        facts_task,
        observations_task,
        supersessions_task,
        verdicts_task,
        decisions_task,
        conflicts_task,
        slot_keys_task,
    )
    return TemporalSnapshot(
        facts=tuple(facts_task.result()),
        observations=tuple(observations_task.result()),
        supersessions=tuple(supersessions_task.result()),
        verdicts=tuple(verdicts_task.result()),
        decisions=tuple(decisions_task.result()),
        conflicts=tuple(conflicts_task.result()),
        slot_keys=tuple(slot_keys_task.result()),
    )


class SnapshotReader:
    """Tenant-bound temporal reads used by queries, slots and audit snapshots."""

    def __init__(self, runner: ScopedRunner, scope: TenantScope) -> None:
        """Bind temporal reads to one tenant and runner."""
        self._runner = runner
        self._scope = scope

    def _map(self, body: str = "") -> str:
        return self._scope.pattern(body)

    async def query(self, query: TemporalQuery) -> list[FactView]:
        """Return the fact versions that satisfy `query`, ordered by fact id."""
        clauses: list[str] = []
        params: dict[str, Any] = {}
        if query.subject_id is not None:
            clauses.append("a.id = $subject")
            params["subject"] = query.subject_id
        if query.object_id is not None:
            clauses.append("b.id = $object")
            params["object"] = query.object_id
        if query.relation_type_id is not None:
            clauses.append("f.relation_type_id = $relation")
            params["relation"] = query.relation_type_id
        prefix = self._fact_pattern()
        if clauses:
            prefix += " WHERE " + " AND ".join(clauses)
        facts = await self._facts(prefix, params)
        return [view_of(fact) for fact in facts if matches(query, fact)]

    async def facts_of_episode(self, episode_id: str) -> list[Fact]:
        """Return every fact version in a slot that the episode touched."""
        prefix = (
            f"MATCH (o:{OBSERVATION} {self._map('episode_id: $episode')})"
            f"-[:IN_SLOT {self._scope.edge()}]->(s:{SLOT} {self._map()}) "
            "WITH DISTINCT s.id AS slot "
            f"MATCH (a:{ENTITY} {self._map()})-"
            f"[f:{FACT} {self._map('slot_key: slot')}]->"
            f"(b:{ENTITY} {self._map()})"
        )
        return await self._facts(prefix, {"episode": episode_id})

    async def observations_of_episode(self, episode_id: str) -> list[Observation]:
        """Return the episode's observations, ordered by observation id."""
        rows = await self._runner.run(
            QuerySpec(
                f"MATCH (o:{OBSERVATION} {self._map('episode_id: $episode')}) "
                f"RETURN {_columns('o', _OBSERVATION_COLUMNS)} ORDER BY o.id",
                {"episode": episode_id},
                read_only=True,
                idempotent=True,
                tag="temporal_observations",
            )
        )
        return [_observation_from(row) for row in rows.rows]

    async def observations_of_slot(self, slot_key: str) -> list[Observation]:
        """Return the slot's observations, ordered by observation id."""
        rows = await self._runner.run(
            QuerySpec(
                f"MATCH (o:{OBSERVATION} {self._map('slot_key: $slot')}) "
                f"RETURN {_columns('o', _OBSERVATION_COLUMNS)} ORDER BY o.id",
                {"slot": slot_key},
                read_only=True,
                idempotent=True,
                tag="temporal_slot_observations",
            )
        )
        return [_observation_from(row) for row in rows.rows]

    async def supersessions_of_slot(self, slot_key: str) -> list[Supersession]:
        """Return the audit rows of a slot, ordered by id."""
        rows = await self._runner.run(
            QuerySpec(
                f"MATCH (n:{SUPERSESSION} {self._map('slot_key: $slot')}) "
                f"RETURN {_columns('n', _SUPERSESSION_COLUMNS)} ORDER BY n.id",
                {"slot": slot_key},
                read_only=True,
                idempotent=True,
                tag="temporal_supersessions",
            )
        )
        return [_supersession_from(row) for row in rows.rows]

    async def slot_keys(self) -> list[str]:
        """Return every slot key, sorted."""
        rows = await self._runner.run(
            QuerySpec(
                f"MATCH (s:{SLOT} {self._map()}) RETURN s.id AS id ORDER BY s.id",
                read_only=True,
                idempotent=True,
                tag="temporal_slot_keys",
            )
        )
        return [str(row["id"]) for row in rows.rows]

    async def episode_has_observations(self, episode_id: str) -> bool:
        """True when the episode already has observations."""
        rows = await self._runner.run(
            QuerySpec(
                f"MATCH (o:{OBSERVATION} {self._map('episode_id: $id')}) "
                "RETURN count(o) AS n",
                {"id": episode_id},
                read_only=True,
                idempotent=True,
                tag="temporal_episode_has_observations",
            )
        )
        return int(rows.scalar("n", 0)) > 0

    async def conflicts_for(
        self, slot_keys: Sequence[str], fact_ids: Sequence[str]
    ) -> list[Conflict]:
        """Return the stored conflicts of the slots or that name the facts."""
        return await self._conflicts(
            f"MATCH (c:{CONFLICT} {self._map()}) "
            "WHERE c.slot_key IN $slots OR any(x IN c.fact_ids WHERE x IN $facts)",
            {"slots": list(slot_keys), "facts": list(fact_ids)},
        )

    async def current_facts(self) -> list[Fact]:
        """Return the current version of every fact in the tenant."""
        return await self._facts(
            self._fact_pattern()
            + " WHERE f.expired_at = $open AND f.status <> $withdrawn",
            {"open": OPEN_MICROS, "withdrawn": FactStatus.WITHDRAWN.value},
        )

    async def facts_touching(self, entity_ids: Sequence[str]) -> list[Fact]:
        """Return the current versions with an entity in ``entity_ids`` on a side."""
        ids = list(dict.fromkeys(entity_ids))
        if not ids:
            return []
        return await self._facts(
            self._fact_pattern()
            + " WHERE f.expired_at = $open AND f.status <> $withdrawn"
            + " AND (a.id IN $ids OR b.id IN $ids)",
            {
                "ids": ids,
                "open": OPEN_MICROS,
                "withdrawn": FactStatus.WITHDRAWN.value,
            },
        )

    async def facts_by_ids(self, fact_ids: Sequence[str]) -> list[Fact]:
        """Return every version of each fact, ordered by fact id and version."""
        ids = list(dict.fromkeys(fact_ids))
        if not ids:
            return []
        return await self._facts(
            self._fact_pattern() + " WHERE f.fact_id IN $ids",
            {"ids": ids},
            tag="temporal_facts_by_ids",
        )

    async def facts_of_entities(self, entity_ids: Sequence[str]) -> list[Fact]:
        """Return every version of every fact with an entity on a side."""
        ids = list(dict.fromkeys(entity_ids))
        if not ids:
            return []
        return await self._facts(
            self._fact_pattern() + " WHERE (a.id IN $ids OR b.id IN $ids)",
            {"ids": ids},
            tag="temporal_facts_of_entities",
        )

    async def observations_by_ids(self, obs_ids: Sequence[str]) -> list[Observation]:
        """Return the observations, ordered by observation id."""
        ids = list(dict.fromkeys(obs_ids))
        if not ids:
            return []
        rows = await self._runner.run(
            QuerySpec(
                f"MATCH (o:{OBSERVATION} {self._map()}) WHERE o.id IN $ids "
                f"RETURN {_columns('o', _OBSERVATION_COLUMNS)} ORDER BY o.id",
                {"ids": ids},
                read_only=True,
                idempotent=True,
                tag="temporal_observations_by_ids",
            )
        )
        return [_observation_from(row) for row in rows.rows]

    async def _observations(self, match: str) -> list[Observation]:
        rows = await self._runner.run(
            QuerySpec(
                f"{match} RETURN {_columns('o', _OBSERVATION_COLUMNS)} ORDER BY o.id",
                read_only=True,
                idempotent=True,
                tag="temporal_snapshot_observations",
            )
        )
        return [_observation_from(row) for row in rows.rows]

    async def _supersessions(self, match: str) -> list[Supersession]:
        rows = await self._runner.run(
            QuerySpec(
                f"{match} RETURN {_columns('n', _SUPERSESSION_COLUMNS)} ORDER BY n.id",
                read_only=True,
                idempotent=True,
                tag="temporal_snapshot_supersessions",
            )
        )
        return [_supersession_from(row) for row in rows.rows]

    async def _conflicts(self, match: str, params: dict[str, Any]) -> list[Conflict]:
        rows = await self._runner.run(
            QuerySpec(
                f"{match} RETURN {_columns('c', _CONFLICT_COLUMNS)} ORDER BY c.id",
                params,
                read_only=True,
                idempotent=True,
                tag="temporal_conflicts",
            )
        )
        return [_conflict_from(row) for row in rows.rows]

    def _fact_pattern(self, edge_body: str = "") -> str:
        return (
            f"MATCH (a:{ENTITY} {self._map()})-"
            f"[f:{FACT} {self._scope.edge(edge_body)}]->"
            f"(b:{ENTITY} {self._map()})"
        )

    async def _facts(
        self, prefix: str, params: dict[str, Any], *, tag: str = "temporal_facts"
    ) -> list[Fact]:
        columns = ", ".join(
            ["a.id AS subject_id", "b.id AS object_id", _columns("f", _FACT_COLUMNS)]
        )
        rows = await self._runner.run(
            QuerySpec(
                f"{prefix} RETURN {columns} ORDER BY f.fact_id, f.version",
                params,
                read_only=True,
                idempotent=True,
                tag=tag,
            )
        )
        return [_fact_from(row) for row in rows.rows]

    async def all_facts(self) -> list[Fact]:
        """Return every fact version, ordered by fact id and version."""
        return await self._facts(self._fact_pattern(), {})

    async def all_observations(self) -> list[Observation]:
        """Return every observation, ordered by id."""
        return await self._observations(f"MATCH (o:{OBSERVATION} {self._map()})")

    async def all_supersessions(self) -> list[Supersession]:
        """Return every supersession, ordered by id."""
        return await self._supersessions(f"MATCH (n:{SUPERSESSION} {self._map()})")

    async def all_conflicts(self) -> list[Conflict]:
        """Return every stored conflict, ordered by id."""
        return await self._conflicts(f"MATCH (c:{CONFLICT} {self._map()})", {})

    async def facts_of_slot(self, slot_key: str) -> list[Fact]:
        """Return every fact version of a slot, ordered by fact id and version."""
        return await self._facts(
            self._fact_pattern("slot_key: $slot"), {"slot": slot_key}
        )

    async def conflicts_of_slot(self, slot_key: str) -> list[Conflict]:
        """Return every stored conflict of a slot, ordered by id."""
        return await self._conflicts(
            f"MATCH (c:{CONFLICT} {self._map('slot_key: $slot')})",
            {"slot": slot_key},
        )
