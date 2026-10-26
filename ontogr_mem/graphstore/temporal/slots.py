"""SlotStore operations for graph-backed temporal storage."""

from collections.abc import Sequence
from typing import Any

from ontogr_mem.data_models.observation import Observation
from ontogr_mem.graphstore.cypher import TenantScope
from ontogr_mem.graphstore.executor import QuerySpec
from ontogr_mem.graphstore.protocol import ScopedRunner
from ontogr_mem.temporal.errors import (
    MissingEntityError,
    MissingFactVersionError,
    RevisionConflict,
    TemporalError,
)
from ontogr_mem.temporal.ledger import GENESIS
from ontogr_mem.temporal.temporal_store import (
    SlotCommit,
    SlotState,
)

from .commit import build_commit_spec
from .rows import (
    _observation_props,
)
from .scope import (
    ENTITY,
    EXPIRED_MATCH,
    FACT,
    OBSERVATION,
    SLOT,
    SUPERSESSION,
)
from .snapshot import SnapshotReader


class SlotStore:
    """Tenant-bound slots persistence operations."""

    def __init__(
        self,
        runner: ScopedRunner,
        scope: TenantScope,
        tenant: str,
        reads: SnapshotReader,
    ) -> None:
        """Bind storage operations to one tenant and runner."""
        self._runner = runner
        self._scope = scope
        self._tenant = tenant
        self._reads = reads

    def _map(self, body: str = "") -> str:
        return self._scope.pattern(body)

    def _stamp(self, props: dict[str, Any]) -> dict[str, Any]:
        return self._scope.stamp(props, self._tenant)

    async def load_slot(self, slot_key: str) -> SlotState:
        """Return the slot's revision, ledger head and every fact version."""
        rows = await self._runner.run(
            QuerySpec(
                f"MATCH (s:{SLOT} {self._map('id: $slot')}) "
                "RETURN s.revision AS revision, s.ledger_seq AS ledger_seq, "
                "s.ledger_head AS ledger_head",
                {"slot": slot_key},
                read_only=True,
                idempotent=True,
                tag="temporal_load_slot",
            )
        )
        if not rows.rows:
            return SlotState(slot_key)
        row = rows.rows[0]
        facts = await self._reads.facts_of_slot(slot_key)
        conflicts = await self._reads.conflicts_of_slot(slot_key)
        return SlotState(
            slot_key=slot_key,
            revision=int(row["revision"]),
            ledger_seq=int(row["ledger_seq"]),
            ledger_head=str(row["ledger_head"]),
            facts=tuple(facts),
            conflicts=tuple(conflicts),
        )

    async def append_observations(self, observations: Sequence[Observation]) -> int:
        """Write new observations and create the slots they mention."""
        unique: dict[str, Observation] = {}
        for observation in observations:
            unique.setdefault(observation.obs_id, observation)
        if not unique:
            return 0
        stored = await self._runner.run(
            QuerySpec(
                f"UNWIND $ids AS i MATCH (o:{OBSERVATION} {self._map('id: i')}) "
                "RETURN o.id AS id",
                {"ids": list(unique)},
                read_only=True,
                idempotent=True,
                tag="temporal_observation_ids",
            )
        )
        present = {row["id"] for row in stored.rows}
        new = [item for key, item in unique.items() if key not in present]
        if not new:
            return 0
        rows = [
            {
                "id": item.obs_id,
                "slot_key": item.slot_key,
                "props": self._stamp(_observation_props(item)),
            }
            for item in new
        ]
        await self._runner.run(
            QuerySpec(
                "\n".join(
                    [
                        "UNWIND $rows AS r",
                        f"MERGE (o:{OBSERVATION} {self._map('id: r.id')})",
                        "SET o += r.props",
                        "WITH o, r",
                        f"MERGE (s:{SLOT} {self._map('id: r.slot_key')})",
                        "ON CREATE SET s.revision = 0, s.ledger_seq = 0, "
                        "s.ledger_head = $genesis",
                        f"MERGE (o)-[edge:IN_SLOT {self._scope.edge()}]->(s)",
                        "SET edge += $edge_props",
                    ]
                ),
                {
                    "rows": rows,
                    "genesis": GENESIS,
                    "edge_props": self._scope.stamp({}, self._tenant),
                },
                tag="temporal_append_observations",
            )
        )
        return len(new)

    async def commit_slot(self, commit: SlotCommit) -> int:
        """Write a slot commit atomically and return the new revision."""
        if commit.empty:
            return commit.base_revision
        rows = await self._runner.run(
            build_commit_spec(commit, self._scope, self._tenant)
        )
        if rows.rows:
            return int(rows.scalar("revision"))
        raise await self._refusal(commit)

    async def remove_episode_observations(self, episode_id: str) -> set[str]:
        """Delete the episode's observations and the audit rows they caused.

        Returns:
            The slot keys the removed observations were in.
        """
        # Slot keys are collected first because the observations are gone after
        # the delete, and the slots they pointed at are needed for re-judging.
        rows = await self._runner.run(
            QuerySpec(
                f"MATCH (o:{OBSERVATION} {self._map('episode_id: $episode')})"
                f"-[:IN_SLOT {self._scope.edge()}]->(s:{SLOT} {self._map()}) "
                "WITH collect(DISTINCT s.id) AS slots "
                f"OPTIONAL MATCH (n:{SUPERSESSION} "
                f"{self._map('source_episode_id: $episode')}) "
                "DETACH DELETE n "
                "WITH DISTINCT slots "
                f"OPTIONAL MATCH (o:{OBSERVATION} {self._map('episode_id: $episode')}) "
                "DETACH DELETE o "
                "RETURN DISTINCT slots",
                {"episode": episode_id},
                tag="temporal_remove_episode_observations",
            )
        )
        return {str(slot) for slot in rows.scalar("slots", [])}

    async def _refusal(self, commit: SlotCommit) -> TemporalError:
        rows = await self._runner.run(
            QuerySpec(
                f"MATCH (s:{SLOT} {self._map('id: $slot')}) "
                "RETURN s.revision AS revision",
                {"slot": commit.slot_key},
                read_only=True,
                idempotent=True,
                tag="temporal_refusal_revision",
            )
        )
        revision = int(rows.scalar("revision", 0))
        if revision != commit.base_revision:
            return RevisionConflict(commit.slot_key, commit.base_revision)
        entity_ids = list(commit.entity_ids)
        if entity_ids:
            missing = await self._runner.run(
                QuerySpec(
                    "UNWIND $ids AS eid "
                    f"OPTIONAL MATCH (e:{ENTITY} {self._map('id: eid')}) "
                    "WITH eid, count(e) AS found WHERE found = 0 "
                    "RETURN eid AS id ORDER BY eid",
                    {"ids": entity_ids},
                    read_only=True,
                    idempotent=True,
                    tag="temporal_refusal_entities",
                )
            )
            if missing.rows:
                return MissingEntityError(str(missing.rows[0]["id"]))
        if commit.expired_facts:
            absent = await self._runner.run(
                QuerySpec(
                    "UNWIND $expired AS x "
                    f"OPTIONAL MATCH ()-[f:{FACT} {self._map(EXPIRED_MATCH)}]->() "
                    "WITH x, count(f) AS found WHERE found = 0 "
                    "RETURN x.fact_id AS fact_id, x.version AS version "
                    "ORDER BY fact_id, version",
                    {
                        "slot": commit.slot_key,
                        "expired": [
                            {"fact_id": fact.fact_id, "version": fact.version}
                            for fact in commit.expired_facts
                        ],
                    },
                    read_only=True,
                    idempotent=True,
                    tag="temporal_refusal_versions",
                )
            )
            if absent.rows:
                row = absent.rows[0]
                return MissingFactVersionError(str(row["fact_id"]), int(row["version"]))
        return RevisionConflict(commit.slot_key, commit.base_revision)
