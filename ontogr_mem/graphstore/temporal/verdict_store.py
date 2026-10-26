"""VerdictStore operations for graph-backed temporal storage."""

from collections.abc import Sequence
from typing import Any

from ontogr_mem.graphstore.cypher import TenantScope
from ontogr_mem.graphstore.executor import QuerySpec
from ontogr_mem.graphstore.protocol import ScopedRunner
from ontogr_mem.temporal.verdicts import ConflictDecision, FactVerdict

from .rows import (
    _DECISION_COLUMNS,
    _VERDICT_COLUMNS,
    _columns,
    _decision_from,
    _decision_props,
    _verdict_from,
    _verdict_props,
)
from .scope import (
    FACT_DECISION,
    FACT_VERDICT,
)


class VerdictStore:
    """Tenant-bound verdict store persistence operations."""

    def __init__(self, runner: ScopedRunner, scope: TenantScope, tenant: str) -> None:
        """Bind storage operations to one tenant and runner."""
        self._runner = runner
        self._scope = scope
        self._tenant = tenant

    def _map(self, body: str = "") -> str:
        return self._scope.pattern(body)

    def _stamp(self, props: dict[str, Any]) -> dict[str, Any]:
        return self._scope.stamp(props, self._tenant)

    async def scrub_episode(self, episode_id: str) -> int:
        """Blank the text copied from the episode into verdicts.

        Returns:
            The number of verdicts changed.
        """
        rows = await self._runner.run(
            QuerySpec(
                f"MATCH (v:{FACT_VERDICT} {self._map()}) "
                "WHERE $episode IN v.source_episode_ids "
                "SET v.reason_text = '', v.quote_a = '', v.quote_b = '' "
                "RETURN count(v) AS n",
                {"episode": episode_id},
                tag="temporal_scrub_episode",
            )
        )
        return int(rows.scalar("n", 0))

    async def put_verdicts(self, verdicts: Sequence[FactVerdict]) -> None:
        """Store verdicts, replacing any row with the same verdict id."""
        unique = {verdict.verdict_id: verdict for verdict in verdicts}
        if not unique:
            return
        rows = [
            {"id": key, "props": self._stamp(_verdict_props(verdict))}
            for key, verdict in unique.items()
        ]
        await self._runner.run(
            QuerySpec(
                "UNWIND $rows AS r "
                f"MERGE (v:{FACT_VERDICT} {self._map('id: r.id')}) "
                "SET v += r.props",
                {"rows": rows},
                tag="temporal_put_verdicts",
            )
        )

    async def verdicts_for(self, fact_ids: Sequence[str]) -> list[FactVerdict]:
        """Return the verdicts that name any of the facts on either side."""
        ids = list(dict.fromkeys(fact_ids))
        if not ids:
            return []
        columns = _columns("v", _VERDICT_COLUMNS)
        rows = await self._runner.run(
            QuerySpec(
                "UNWIND $ids AS i "
                f"MATCH (v:{FACT_VERDICT} {self._map('fact_a_id: i')}) "
                f"RETURN {columns} "
                "UNION "
                "UNWIND $ids AS i "
                f"MATCH (v:{FACT_VERDICT} {self._map('fact_b_id: i')}) "
                f"RETURN {columns}",
                {"ids": ids},
                read_only=True,
                idempotent=True,
                tag="temporal_verdicts_for",
            )
        )
        found = [_verdict_from(row) for row in rows.rows]
        return sorted(found, key=lambda verdict: verdict.verdict_id)

    async def put_decision(self, decision: ConflictDecision) -> None:
        """Store a decision, replacing any row with the same decision id."""
        await self._runner.run(
            QuerySpec(
                f"MERGE (d:{FACT_DECISION} {self._map('id: $id')}) SET d += $props",
                {
                    "id": decision.decision_id,
                    "props": self._stamp(_decision_props(decision)),
                },
                tag="temporal_put_decision",
            )
        )

    async def decisions_for(self, fact_ids: Sequence[str]) -> list[ConflictDecision]:
        """Return the decisions that name any of the given facts."""
        ids = list(dict.fromkeys(fact_ids))
        if not ids:
            return []
        rows = await self._runner.run(
            QuerySpec(
                f"MATCH (d:{FACT_DECISION} {self._map()}) "
                "WHERE any(x IN d.fact_ids WHERE x IN $ids) "
                f"RETURN {_columns('d', _DECISION_COLUMNS)} ORDER BY d.id",
                {"ids": ids},
                read_only=True,
                idempotent=True,
                tag="temporal_decisions_for",
            )
        )
        return [_decision_from(row) for row in rows.rows]

    async def all_verdicts(self) -> list[FactVerdict]:
        """Return every stored verdict, ordered by id."""
        match = f"MATCH (v:{FACT_VERDICT} {self._map()})"
        rows = await self._runner.run(
            QuerySpec(
                f"{match} RETURN {_columns('v', _VERDICT_COLUMNS)} ORDER BY v.id",
                read_only=True,
                idempotent=True,
                tag="temporal_snapshot_verdicts",
            )
        )
        return [_verdict_from(row) for row in rows.rows]

    async def all_decisions(self) -> list[ConflictDecision]:
        """Return every stored decision, ordered by id."""
        match = f"MATCH (d:{FACT_DECISION} {self._map()})"
        rows = await self._runner.run(
            QuerySpec(
                f"{match} RETURN {_columns('d', _DECISION_COLUMNS)} ORDER BY d.id",
                read_only=True,
                idempotent=True,
                tag="temporal_snapshot_decisions",
            )
        )
        return [_decision_from(row) for row in rows.rows]
