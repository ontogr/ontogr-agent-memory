"""Atomic merge writes and recovery of interrupted graph merges."""

import asyncio
import logging
from collections.abc import Sequence
from typing import Any
from uuid import uuid4

from ontogr_mem.data_models.entities import (
    Alias,
    EntityStatus,
    Mention,
    MergeRecord,
    MergeState,
)
from ontogr_mem.graphstore.cypher import TenantScope
from ontogr_mem.graphstore.executor import QuerySpec, Rows
from ontogr_mem.graphstore.protocol import ScopedRunner
from ontogr_mem.graphstore.reserved import LOCK
from ontogr_mem.graphstore.resolver.entities import ResolverEntities
from ontogr_mem.graphstore.resolver.rows import (
    alias_row,
    entity_row,
    mention_row,
    merge_row,
    verdict_row,
)
from ontogr_mem.resolution.errors import MergeRefusedError
from ontogr_mem.resolution.resolver_store import (
    MergePlan,
    MergeRows,
    UnmergePlan,
    merge_intent,
    merge_rows_moved,
    merge_source_entity,
    snapshot_with_intent,
)


ENTITY = "Entity"
ALIAS = "Alias"
MENTION = "Mention"
VERDICT = "EntityVerdict"
MERGE = "MergeRecord"


log = logging.getLogger(__name__)


class ResolverMerges:
    """Apply and recover merges using the same entity storage and lock."""

    def __init__(
        self,
        runner: ScopedRunner,
        scope: TenantScope,
        entities: ResolverEntities,
        resolve_lock: asyncio.Lock,
    ) -> None:
        """Bind merge operations to entity storage and the facade lock."""
        self._runner = runner
        self._scope = scope
        self._entities = entities
        self._resolve_lock = resolve_lock

    async def apply_merge(self, plan: MergePlan) -> MergeRecord:
        """Apply one merge atomically and return its record.

        Raises:
            MergeRefusedError: If either entity is no longer active, or the
                merge was already applied.
        """
        async with self._resolve_lock:
            record = plan.record.model_copy(
                update={"seq": await self._entities.next_revision()}
            )
            rows = MergeRows.from_merge_plan(plan)
            applied = record.model_copy(
                update={
                    "snapshot": snapshot_with_intent(record.snapshot, rows),
                }
            )
            result = await self._apply_merge_atomically(applied, rows)
            if not result.rows:
                if await self.merge_record(record.merge_id) is not None:
                    raise MergeRefusedError("already_applied", record.merge_id)
                await self._require_active(plan.survivor.entity_id)
                await self._require_active(plan.absorbed.entity_id)
                raise MergeRefusedError("rows_missing", record.merge_id)
            return applied.model_copy(deep=True)

    async def apply_unmerge(self, plan: UnmergePlan) -> MergeRecord:
        """Apply one unmerge atomically and return its record.

        Raises:
            MergeRefusedError: If the merge record is missing or is not applied.
        """
        async with self._resolve_lock:
            record = plan.record.model_copy(
                update={"seq": await self._entities.next_revision()}
            )
            rows = MergeRows.from_unmerge_plan(plan)
            reverted = record.model_copy(
                update={
                    "snapshot": snapshot_with_intent(record.snapshot, rows),
                }
            )
            result = await self._apply_merge_atomically(
                reverted, rows, expected_record_state=MergeState.APPLIED
            )
            if not result.rows:
                current = await self.merge_record(record.merge_id)
                if current is None:
                    raise MergeRefusedError("missing", record.merge_id)
                if current.state is not MergeState.APPLIED:
                    raise MergeRefusedError("not_applied", record.merge_id)
                raise MergeRefusedError("rows_missing", record.merge_id)
            return reverted.model_copy(deep=True)

    async def _apply_merge_atomically(
        self,
        record: MergeRecord,
        rows: MergeRows,
        *,
        expected_record_state: MergeState | None = None,
    ) -> Rows:
        """Write a merge's graph rows and final record in one statement."""
        record_row = merge_row(record)
        mention_rows = [mention_row(mention) for mention in rows.mentions]
        alias_rows = [alias_row(alias) for alias in rows.aliases]
        verdict_rows = [verdict_row(verdict) for verdict in rows.verdicts]
        entity_ids = sorted(
            {
                rows.survivor.entity_id,
                rows.absorbed.entity_id,
                *(entity.entity_id for entity in rows.redirected),
                *(mention.entity_id for mention in rows.mentions),
                *(alias.entity_id for alias in rows.aliases),
            }
        )
        params: dict[str, Any] = {
            "survivor_id": rows.survivor.entity_id,
            "absorbed_id": rows.absorbed.entity_id,
            "survivor_status": EntityStatus.ACTIVE.value,
            "absorbed_status": (
                EntityStatus.ACTIVE.value
                if expected_record_state is None
                else EntityStatus.MERGED.value
            ),
            "entity_rows": [
                entity_row(entity)
                for entity in (rows.survivor, rows.absorbed, *rows.redirected)
            ],
            "entity_ids": entity_ids,
            "mention_rows": mention_rows,
            "mention_ids": [row["id"] for row in mention_rows],
            "alias_rows": alias_rows,
            "alias_ids": [row["id"] for row in alias_rows],
            "verdict_rows": verdict_rows,
            "id": record_row["id"],
            "props": record_row["props"],
            "owner": uuid4().hex,
            "writer": uuid4().hex,
        }
        lines = [
            f"MATCH (survivor:{ENTITY} {self._scope.pattern('id: $survivor_id')})",
            f"MATCH (absorbed:{ENTITY} {self._scope.pattern('id: $absorbed_id')})",
            "WITH survivor, absorbed, "
            "CASE WHEN survivor.id < absorbed.id THEN survivor ELSE absorbed END "
            "AS first, "
            "CASE WHEN survivor.id < absorbed.id THEN absorbed ELSE survivor END "
            "AS second",
            f"WITH survivor, absorbed, first, second, first.{LOCK} AS first_lock, "
            f"second.{LOCK} AS second_lock",
            f"SET first.{LOCK} = $writer",
            f"SET second.{LOCK} = $writer",
            "WITH survivor, absorbed, first, second, first_lock, second_lock, "
            "(survivor.status = $survivor_status "
            "AND absorbed.status = $absorbed_status "
            "AND (absorbed.merged_into = $survivor_id "
            "OR $absorbed_status = 'active')) AS eligible",
            f"SET first.{LOCK} = first_lock, second.{LOCK} = second_lock",
            "WITH survivor, absorbed, eligible WHERE eligible",
            f"OPTIONAL MATCH (target_entity:{ENTITY} "
            f"{self._scope.pattern()}) "
            "WHERE target_entity.id IN $entity_ids",
            "WITH survivor, absorbed, "
            "collect(DISTINCT target_entity) AS target_entities "
            "WHERE size(target_entities) = size($entity_ids)",
            f"OPTIONAL MATCH (mention:{MENTION} "
            f"{self._scope.pattern()}) WHERE mention.id IN $mention_ids",
            "WITH survivor, absorbed, collect(DISTINCT mention) AS mentions "
            "WHERE size(mentions) = size($mention_ids)",
            f"OPTIONAL MATCH (alias:{ALIAS} "
            f"{self._scope.pattern()}) WHERE alias.id IN $alias_ids",
            "WITH survivor, absorbed, collect(DISTINCT alias) AS aliases "
            "WHERE size(aliases) = size($alias_ids)",
        ]

        if expected_record_state is None:
            lines.extend(
                [
                    f"MERGE (record:{MERGE} {self._scope.pattern('id: $id')})",
                    "ON CREATE SET record.owner = $owner",
                    "WITH survivor, absorbed, record WHERE record.owner = $owner",
                ]
            )
        else:
            params["expected_record_state"] = expected_record_state.value
            params["final_record_state"] = record.state.value
            lines.extend(
                [
                    f"MATCH (record:{MERGE} {self._scope.pattern('id: $id')})",
                    f"WITH survivor, absorbed, record, record.{LOCK} AS previous_lock",
                    f"SET record.{LOCK} = $writer",
                    "WITH survivor, absorbed, record, previous_lock, "
                    "(record.state = $expected_record_state OR "
                    "(record.state = $final_record_state "
                    "AND record.owner = $owner)) AS record_is_current",
                    f"SET record.{LOCK} = previous_lock",
                    "WITH survivor, absorbed, record, record_is_current "
                    "WHERE record_is_current",
                ]
            )

        lines.extend(
            [
                "UNWIND $entity_rows AS item",
                f"MATCH (e:{ENTITY} {self._scope.pattern('id: item.id')})",
                f"SET e += item.props{self._scope.stamp_clause('e')}",
                "WITH DISTINCT record",
                "UNWIND CASE WHEN size($mention_rows) = 0 "
                "THEN [null] ELSE $mention_rows END AS item",
                f"OPTIONAL MATCH (m:{MENTION} {self._scope.pattern('id: item.id')})",
                f"OPTIONAL MATCH (e:{ENTITY} "
                f"{self._scope.pattern('id: item.entity_id')})",
                f"OPTIONAL MATCH (m)-[old:OF {self._scope.edge()}]->"
                f"(:{ENTITY} {self._scope.pattern()})",
                "FOREACH (_ IN CASE WHEN item IS NOT NULL AND m IS NOT NULL "
                "AND e IS NOT NULL THEN [1] ELSE [] END | DELETE old "
                "SET m += item.props "
                f"MERGE (m)-[r:OF {self._scope.edge()}]->(e) "
                f"{self._scope.stamp_assignment('r')})",
                "WITH DISTINCT record",
                "UNWIND CASE WHEN size($alias_rows) = 0 "
                "THEN [null] ELSE $alias_rows END AS item",
                f"OPTIONAL MATCH (a:{ALIAS} {self._scope.pattern('id: item.id')})",
                f"OPTIONAL MATCH (e:{ENTITY} "
                f"{self._scope.pattern('id: item.entity_id')})",
                f"OPTIONAL MATCH (a)-[old:ALIAS_OF {self._scope.edge()}]->"
                f"(:{ENTITY} {self._scope.pattern()})",
                "FOREACH (_ IN CASE WHEN item IS NOT NULL AND a IS NOT NULL "
                "AND e IS NOT NULL THEN [1] ELSE [] END | DELETE old "
                "SET a += item.props "
                f"MERGE (a)-[r:ALIAS_OF {self._scope.edge()}]->(e) "
                f"{self._scope.stamp_assignment('r')})",
                "WITH DISTINCT record",
                "UNWIND CASE WHEN size($verdict_rows) = 0 "
                "THEN [null] ELSE $verdict_rows END AS item",
                "FOREACH (_ IN CASE WHEN item IS NOT NULL THEN [1] ELSE [] END | "
                f"MERGE (v:{VERDICT} {self._scope.pattern('id: item.id')}) "
                f"SET v += item.props{self._scope.stamp_clause('v')})",
                "WITH DISTINCT record",
                f"SET record += $props{self._scope.stamp_clause('record')}",
                "SET record.owner = $owner",
                "RETURN record.json AS value",
            ]
        )
        return await self._runner.run(
            QuerySpec(
                "\n".join(lines),
                params,
                idempotent=False,
                tag="resolver_merge_apply",
            )
        )

    async def merge_record(self, merge_id: str) -> MergeRecord | None:
        """Return the merge record, or None."""
        rows = await self._runner.run(
            QuerySpec(
                f"MATCH (m:{MERGE} {self._scope.pattern('id: $id')}) "
                "RETURN m.json AS value",
                {"id": merge_id},
                read_only=True,
                idempotent=True,
                tag="resolver_merge_record",
            )
        )
        return (
            MergeRecord.model_validate_json(rows.rows[0]["value"])
            if rows.rows
            else None
        )

    async def recover_merges(self) -> list[MergeRecord]:
        """Finish every interrupted merge and return the finished records.

        A record left ``APPLYING`` is reapplied from the rows it keeps. Those
        rows are checked before the record is settled. A record whose rows did
        not move stays ``APPLYING`` and is not returned.
        """
        rows = await self._runner.run(
            QuerySpec(
                f"MATCH (m:{MERGE} {self._scope.pattern('state: $state')}) "
                "RETURN m.json AS value ORDER BY m.id",
                {"state": MergeState.APPLYING.value},
                read_only=True,
                idempotent=True,
                tag="resolver_merge_applying",
            )
        )
        finished: list[MergeRecord] = []
        for row in rows.rows:
            record = MergeRecord.model_validate_json(row["value"])
            settled = await self._settle_merge(record)
            if settled is not None:
                finished.append(settled)
        return finished

    async def _settle_merge(self, record: MergeRecord) -> MergeRecord | None:
        """Reapply one interrupted merge and settle its record."""
        rows = merge_intent(record.snapshot)
        if rows is None:
            log.warning(
                "merge %s keeps no rows and cannot be finished", record.merge_id
            )
            return None
        await self._apply_rows(rows)
        if not await self._rows_moved(rows):
            log.warning(
                "merge %s did not move its rows and stays applying", record.merge_id
            )
            return None
        settled = record.model_copy(update={"state": rows.state})
        await self.write_merge_record(settled)
        return settled

    async def _rows_moved(self, rows: MergeRows) -> bool:
        """Return whether the rows of a merge left the entity they must leave."""
        source = merge_source_entity(rows)
        mentions = {
            mention.mention_id: mention.entity_id
            for mention in await self._entities.mentions_of(source)
        }
        aliases = {
            alias.alias_id: alias.entity_id
            for alias in await self._entities.aliases_of(source)
        }
        return merge_rows_moved(
            rows,
            absorbed=await self._entities.get_one(rows.absorbed.entity_id),
            mention_entities=mentions,
            alias_entities=aliases,
        )

    async def write_merge_record(self, record: MergeRecord) -> None:
        """Write a merge record without changing its sequence."""
        row = merge_row(record)
        await self._runner.run(
            QuerySpec(
                f"MERGE (m:{MERGE} {self._scope.pattern('id: $id')}) "
                f"SET m += $props{self._scope.stamp_clause('m')}",
                {"id": row["id"], "props": row["props"]},
                idempotent=True,
                tag="resolver_merge_write",
            )
        )

    async def _apply_rows(self, rows: MergeRows) -> None:
        """Write the rows of one merge or unmerge."""
        await self._entities.write_entities(
            [rows.survivor, rows.absorbed, *rows.redirected]
        )
        await self._move_rows(rows.mentions, rows.aliases)
        await self._entities.write_verdicts(rows.verdicts)

    async def _move_rows(
        self,
        mentions: Sequence[Mention],
        aliases: Sequence[Alias],
    ) -> None:
        """Move mentions and aliases to the entity they now belong to."""
        if mentions:
            await self._runner.run(
                QuerySpec(
                    "UNWIND $rows AS item "
                    f"MATCH (m:{MENTION} {self._scope.pattern('id: item.id')}) "
                    f"MATCH (e:{ENTITY} {self._scope.pattern('id: item.entity_id')}) "
                    f"OPTIONAL MATCH (m)-[old:OF {self._scope.edge()}]->"
                    f"(:{ENTITY} {self._scope.pattern()}) DELETE old "
                    "SET m += item.props WITH m, e "
                    f"MERGE (m)-[r:OF {self._scope.edge()}]->(e)"
                    f" {self._scope.stamp_assignment('r')}",
                    {"rows": [mention_row(m) for m in mentions]},
                    idempotent=True,
                    tag="resolver_mention_move",
                )
            )
        if aliases:
            await self._runner.run(
                QuerySpec(
                    "UNWIND $rows AS item "
                    f"MATCH (a:{ALIAS} {self._scope.pattern('id: item.id')}) "
                    f"MATCH (e:{ENTITY} {self._scope.pattern('id: item.entity_id')}) "
                    f"OPTIONAL MATCH (a)-[old:ALIAS_OF {self._scope.edge()}]->"
                    f"(:{ENTITY} {self._scope.pattern()}) DELETE old "
                    "SET a += item.props WITH a, e "
                    f"MERGE (a)-[r:ALIAS_OF {self._scope.edge()}]->(e)"
                    f" {self._scope.stamp_assignment('r')}",
                    {"rows": [alias_row(a) for a in aliases]},
                    idempotent=True,
                    tag="resolver_alias_move",
                )
            )

    async def _require_active(self, entity_id: str) -> None:
        entity = await self._entities.get_one(entity_id)
        if entity is None or entity.status is not EntityStatus.ACTIVE:
            raise MergeRefusedError("not_active", entity_id)
