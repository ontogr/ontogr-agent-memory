"""Entity, alias, mention and verdict storage on a tenant graph."""

from collections.abc import Sequence
from typing import Any
from uuid import uuid4

from ontogr_mem.data_models.entities import (
    Alias,
    Entity,
    EntityStatus,
    EntityVerdict,
    Mention,
)
from ontogr_mem.data_models.timestamps import OPEN_MICROS
from ontogr_mem.graphstore.cypher import TenantScope
from ontogr_mem.graphstore.executor import QuerySpec
from ontogr_mem.graphstore.protocol import ScopedRunner
from ontogr_mem.graphstore.reserved import LOCK
from ontogr_mem.graphstore.resolver.rows import (
    alias_row,
    entity_row,
    mention_row,
    verdict_row,
)
from ontogr_mem.graphstore.resolver.selection import deduplication_entities
from ontogr_mem.resolution.blocking import BlockingRow
from ontogr_mem.resolution.errors import (
    ConcurrentEntityWriteError,
    EntityHasFactsError,
    EpisodeAlreadyResolvedError,
    NotFoundError,
)
from ontogr_mem.resolution.keys import stripped_key


COUNTER = "ResolverCounter"
ENTITY = "Entity"
ALIAS = "Alias"
MENTION = "Mention"
VERDICT = "EntityVerdict"
MERGE = "MergeRecord"


def _alias_rows_by_owner(aliases: Sequence[Alias]) -> dict[str, list[dict[str, Any]]]:
    """Group unique alias rows by owner and reject conflicting duplicate IDs."""
    rows_by_id: dict[str, dict[str, Any]] = {}
    for alias in aliases:
        row = alias_row(alias)
        existing = rows_by_id.get(row["id"])
        if existing is not None and existing != row:
            raise ValueError(f"conflicting alias rows for {row['id']!r}")
        rows_by_id[row["id"]] = row

    by_owner: dict[str, list[dict[str, Any]]] = {}
    for row in rows_by_id.values():
        by_owner.setdefault(row["entity_id"], []).append(row)
    for owner_rows in by_owner.values():
        owner_rows.sort(key=lambda row: row["id"])
    return by_owner


class ResolverEntities:
    """Read and write resolution rows through one scoped runner."""

    def __init__(self, runner: ScopedRunner, scope: TenantScope) -> None:
        """Bind entity operations to the runner and scope."""
        self._runner = runner
        self._scope = scope

    async def episode_resolution_state(self, episode_id: str) -> bool:
        """Return True when resolution already wrote mentions for the episode."""
        rows = await self._runner.run(
            QuerySpec(
                f"MATCH (m:{MENTION} "
                f"{self._scope.pattern('episode_id: $episode_id')}) "
                "RETURN count(m) AS n",
                {"episode_id": episode_id},
                read_only=True,
                idempotent=True,
                tag="resolver_episode_state",
            )
        )
        return bool(rows.scalar("n", 0))

    async def entity_id_for_ref(self, episode_id: str, local_id: str) -> str | None:
        """Return the entity named by an episode-local reference, or None."""
        pattern = self._scope.pattern("episode_id: $episode_id, local_id: $local_id")
        rows = await self._runner.run(
            QuerySpec(
                f"MATCH (m:{MENTION} {pattern}) "
                "RETURN m.entity_id AS entity_id ORDER BY m.id LIMIT 1",
                {"episode_id": episode_id, "local_id": local_id},
                read_only=True,
                idempotent=True,
                tag="resolver_entity_for_ref",
            )
        )
        return rows.scalar("entity_id")

    async def has_entity(self, entity_id: str) -> bool:
        """Return whether this tenant contains the entity id."""
        return bool(await self.get_entities([entity_id]))

    async def entities_by_uri(self, uri: str) -> list[Entity]:
        """Return the entities carrying `uri`, in any status, ordered by id."""
        rows = await self._runner.run(
            QuerySpec(
                f"MATCH (n:{ENTITY} {self._scope.pattern('uri: $uri')}) "
                "RETURN n.json AS value ORDER BY n.id",
                {"uri": uri},
                read_only=True,
                idempotent=True,
                tag="resolver_by_uri",
            )
        )
        return [Entity.model_validate_json(r["value"]) for r in rows.rows]

    async def entities_by_strict_key(self, strict_key: str) -> list[Entity]:
        """Return the entities an alias with `strict_key` resolves to."""
        rows = await self._runner.run(
            QuerySpec(
                f"MATCH (a:{ALIAS} {self._scope.pattern('strict_key: $key')})"
                f"-[:ALIAS_OF {self._scope.edge()}]->"
                f"(e:{ENTITY} {self._scope.pattern()}) "
                "RETURN DISTINCT e.id AS id ORDER BY e.id",
                {"key": strict_key},
                read_only=True,
                idempotent=True,
                tag="resolver_by_key",
            )
        )
        ids = [r["id"] for r in rows.rows]
        return await self.get_entities(ids, follow_merged=True)

    async def entities_by_loose_key(self, loose_key: str) -> list[Entity]:
        """Return candidate entities an alias's indexed loose key resolves to."""
        rows = await self._runner.run(
            QuerySpec(
                f"MATCH (a:{ALIAS} {self._scope.pattern('loose_key: $key')}) "
                "RETURN DISTINCT a.entity_id AS id ORDER BY id",
                {"key": loose_key},
                read_only=True,
                idempotent=True,
                tag="resolver_by_loose_key",
            )
        )
        return await self.get_entities(
            [row["id"] for row in rows.rows], follow_merged=True
        )

    async def get_entities(
        self, ids: Sequence[str], *, follow_merged: bool = False
    ) -> list[Entity]:
        """Return the entities with these ids, ordered by id."""
        if not ids:
            return []
        found = await self._fetch_entities(list(dict.fromkeys(ids)))
        if follow_merged:
            for _ in range(10):
                missing = {
                    e.merged_into
                    for e in found.values()
                    if e.merged_into and e.merged_into not in found
                }
                if not missing:
                    break
                found.update(await self._fetch_entities(sorted(missing)))
            resolved: dict[str, Entity] = {}
            for entity in found.values():
                target = self._follow(entity.entity_id, found)
                resolved[target.entity_id] = target
            found = resolved
        return [found[i] for i in sorted(found)]

    async def _fetch_entities(self, ids: Sequence[str]) -> dict[str, Entity]:
        if not ids:
            return {}
        rows = await self._runner.run(
            QuerySpec(
                f"UNWIND $ids AS id MATCH (e:{ENTITY} {self._scope.pattern('id: id')}) "
                "RETURN e.json AS value",
                {"ids": list(ids)},
                read_only=True,
                idempotent=True,
                tag="resolver_get_entities",
            )
        )
        return {
            entity.entity_id: entity
            for entity in (Entity.model_validate_json(r["value"]) for r in rows.rows)
        }

    async def blocking_rows(
        self, *, include_archived: bool = False
    ) -> list[BlockingRow]:
        """Return the blocking rows of the active entities, ordered by id."""
        statuses = [EntityStatus.ACTIVE]
        if include_archived:
            statuses.append(EntityStatus.ARCHIVED)
        entity_rows: list[dict[str, Any]] = []
        for status in statuses:
            found = await self._runner.run(
                QuerySpec(
                    f"MATCH (e:{ENTITY} {self._scope.pattern('status: $status')}) "
                    "RETURN e.json AS value, e.stripped_key AS stripped ORDER BY e.id",
                    {"status": status.value},
                    read_only=True,
                    idempotent=True,
                    tag="resolver_blocking",
                )
            )
            entity_rows.extend(found.rows)
        entity_rows.sort(key=lambda r: Entity.model_validate_json(r["value"]).entity_id)
        aliases = await self._runner.run(
            QuerySpec(
                f"MATCH (a:{ALIAS} {self._scope.pattern()}) "
                "RETURN a.entity_id AS entity_id, "
                "a.loose_key AS loose_key ORDER BY a.id",
                {},
                read_only=True,
                idempotent=True,
                tag="resolver_blocking_aliases",
            )
        )
        by_entity: dict[str, list[str]] = {}
        for row in aliases.rows:
            by_entity.setdefault(row["entity_id"], []).append(row["loose_key"])
        return [
            BlockingRow(
                entity_id=entity.entity_id,
                loose_key=entity.loose_key,
                stripped_key=row.get("stripped") or stripped_key(entity.canonical_name),
                alias_keys=tuple(by_entity.get(entity.entity_id, ())),
            )
            for row in entity_rows
            for entity in [Entity.model_validate_json(row["value"])]
        ]

    async def write_entity_batch(
        self, entities: Sequence[Entity], aliases: Sequence[Alias]
    ) -> list[Entity]:
        """Write entities and aliases, advancing affected entity revisions."""
        if not entities and not aliases:
            return []

        entities_by_id = {entity.entity_id: entity for entity in entities}
        aliases_by_owner = _alias_rows_by_owner(aliases)
        entity_ids = sorted(set(entities_by_id) | set(aliases_by_owner))

        for _ in range(3):
            snapshots = await self._runner.run(
                QuerySpec(
                    "UNWIND $ids AS id "
                    f"MATCH (e:{ENTITY} {self._scope.pattern('id: id')}) "
                    "RETURN e.id AS id, e.json AS value, e.revision AS revision, "
                    "e.last_checked_version AS checkpoint",
                    {"ids": entity_ids},
                    read_only=True,
                    idempotent=True,
                    tag="resolver_entity_write_snapshot",
                )
            )
            current: dict[str, tuple[Entity, str, int, int]] = {}
            for row in snapshots.rows:
                value = row["value"]
                revision = int(row["revision"] or 0)
                checkpoint = int(row["checkpoint"] or 0)
                entity = Entity.model_validate_json(value).model_copy(
                    update={
                        "revision": revision,
                        "last_checked_version": checkpoint,
                    }
                )
                current[row["id"]] = (entity, value, revision, checkpoint)

            missing_alias_owners = (
                set(aliases_by_owner) - set(current) - set(entities_by_id)
            )
            if missing_alias_owners:
                raise NotFoundError(f"entity {sorted(missing_alias_owners)[0]!r}")

            revision = await self.next_revision()
            write_rows: list[dict[str, Any]] = []
            stored: dict[str, Entity] = {}
            for entity_id in entity_ids:
                snapshot = current.get(entity_id)
                incoming = entities_by_id.get(entity_id)
                entity = snapshot[0] if incoming is None and snapshot else incoming
                if entity is None:
                    raise NotFoundError(f"entity {entity_id!r}")
                checkpoint = max(
                    entity.last_checked_version,
                    snapshot[3] if snapshot else 0,
                )
                stamped = entity.model_copy(
                    update={
                        "revision": revision,
                        "last_checked_version": checkpoint,
                    }
                )
                stored[entity_id] = stamped
                row = entity_row(stamped)
                row.update(
                    {
                        "expected_json": snapshot[1] if snapshot else None,
                        "expected_revision": snapshot[2] if snapshot else None,
                        "expected_checkpoint": snapshot[3] if snapshot else None,
                        "aliases": aliases_by_owner.get(entity_id, []),
                    }
                )
                write_rows.append(row)

            if await self._write_entity_rows(write_rows, strict_snapshot=True):
                return [
                    stored[entity.entity_id].model_copy(deep=True)
                    for entity in entities
                ]

        raise ConcurrentEntityWriteError(
            "entity rows changed during three write attempts"
        )

    async def delete_aliases(self, entity: Entity, alias_ids: Sequence[str]) -> Entity:
        """Delete aliases and advance their owner's revision atomically."""
        if not alias_ids:
            return entity.model_copy(deep=True)

        snapshots = await self._runner.run(
            QuerySpec(
                "UNWIND $ids AS id "
                f"MATCH (e:{ENTITY} {self._scope.pattern('id: id')}) "
                "RETURN e.id AS id, e.json AS value, e.revision AS revision, "
                "e.last_checked_version AS checkpoint",
                {"ids": [entity.entity_id]},
                read_only=True,
                idempotent=True,
                tag="resolver_entity_write_snapshot",
            )
        )
        if not snapshots.rows:
            raise NotFoundError(f"entity {entity.entity_id!r}")

        snapshot = snapshots.rows[0]
        revision = int(snapshot["revision"] or 0)
        checkpoint = int(snapshot["checkpoint"] or 0)
        if revision != entity.revision or checkpoint != entity.last_checked_version:
            raise ConcurrentEntityWriteError(
                "entity changed before its aliases could be deleted"
            )

        current = Entity.model_validate_json(snapshot["value"]).model_copy(
            update={"revision": revision, "last_checked_version": checkpoint}
        )
        next_revision = await self.next_revision()
        stamped = current.model_copy(update={"revision": next_revision})
        row = entity_row(stamped)
        row.update(
            {
                "expected_json": snapshot["value"],
                "expected_revision": revision,
                "expected_checkpoint": checkpoint,
                "aliases": [
                    {"id": alias_id, "entity_id": entity.entity_id, "remove": True}
                    for alias_id in sorted(set(alias_ids))
                ],
            }
        )
        if not await self._write_entity_rows([row], strict_snapshot=True):
            raise ConcurrentEntityWriteError(
                "entity changed before its aliases could be deleted"
            )
        return stamped.model_copy(deep=True)

    async def _write_entity_rows(
        self,
        rows: Sequence[dict[str, Any]],
        *,
        strict_snapshot: bool,
        allow_alias_moves: bool = False,
    ) -> bool:
        """Apply entity and alias rows only when every owner is safe to write."""
        aliases = [alias for row in rows for alias in row["aliases"]]
        counter_pattern = self._scope.pattern("id: 'revision'")
        cypher = (
            "WITH 1 AS seed "
            "FOREACH (_ IN CASE WHEN size($aliases) > 0 THEN [1] ELSE [] END | "
            f"MERGE (guard:{COUNTER} {counter_pattern}) "
            "ON CREATE SET guard.value = 0 "
            f"SET guard.{LOCK} = $writer REMOVE guard.{LOCK}) "
            "WITH seed "
            "UNWIND CASE WHEN size($aliases) = 0 "
            "THEN [null] ELSE $aliases END AS alias_item "
            f"OPTIONAL MATCH (existing_alias:{ALIAS} "
            f"{self._scope.pattern('id: alias_item.id')}) "
            f"OPTIONAL MATCH (existing_alias)-[:ALIAS_OF {self._scope.edge()}]->"
            f"(old_owner:{ENTITY} {self._scope.pattern()}) "
            "WITH alias_item, existing_alias, collect(old_owner.id) "
            "AS old_owner_ids "
            "WITH collect({item: alias_item, node: existing_alias, "
            "old_owner_ids: old_owner_ids}) AS alias_checks "
            "WITH alias_checks, coalesce(all(alias IN alias_checks "
            "WHERE alias.item IS NULL OR "
            "(alias.item.entity_id IN [row IN $rows | row.id] AND "
            "(alias.node IS NULL OR "
            "(alias.node.entity_id IN [row IN $rows | row.id] AND "
            "all(owner_id IN alias.old_owner_ids "
            "WHERE owner_id IN [row IN $rows | row.id]) AND "
            "($allow_alias_moves OR "
            "(alias.node.entity_id = alias.item.entity_id AND "
            "all(owner_id IN alias.old_owner_ids "
            "WHERE owner_id = alias.item.entity_id))))))), false) "
            "AS aliases_match "
            "WITH alias_checks, aliases_match, "
            "coalesce(all(alias IN alias_checks WHERE alias.item IS NULL OR "
            "(alias.node IS NOT NULL AND "
            "alias.node.entity_id = alias.item.entity_id AND "
            "size(alias.old_owner_ids) > 0 AND "
            "all(owner_id IN alias.old_owner_ids "
            "WHERE owner_id = alias.item.entity_id))), false) "
            "AS aliases_already_moved "
            "UNWIND $rows AS item WITH aliases_match, aliases_already_moved, item "
            "ORDER BY item.id "
            f"MERGE (e:{ENTITY} "
            f"{self._scope.pattern('id: item.id')}) "
            f"ON CREATE SET e.{LOCK} = $writer "
            f"WITH aliases_match, aliases_already_moved, e, item, "
            f"e.{LOCK} AS previous_lock ORDER BY e.id "
            "WITH aliases_match, aliases_already_moved, e, item, previous_lock, "
            "coalesce(previous_lock = $writer, false) AS created "
            f"FOREACH (_ IN CASE WHEN created THEN [] ELSE [1] END | "
            f"SET e.{LOCK} = $writer) "
            "WITH aliases_match, aliases_already_moved, "
            "collect({node: e, item: item, created: created, "
            "previous_lock: CASE WHEN created THEN null ELSE previous_lock END, "
            "eligible: coalesce(CASE WHEN created THEN item.expected_json IS NULL "
            "WHEN $strict_snapshot THEN e.json = item.expected_json "
            "AND coalesce(e.revision, 0) = item.expected_revision "
            "AND item.props.revision > coalesce(e.revision, 0) "
            "AND coalesce(e.last_checked_version, 0) = item.expected_checkpoint "
            "ELSE (coalesce(e.revision, 0) < item.props.revision OR "
            "(coalesce(e.revision, 0) = item.props.revision AND "
            "(NOT $allow_alias_moves OR aliases_already_moved))) "
            "AND coalesce(e.last_checked_version, 0) "
            "<= item.props.last_checked_version END, false)}) AS owners "
            "WITH owners, aliases_match, aliases_already_moved, "
            "size(owners) = size($rows) "
            "AND all(owner IN owners WHERE owner.eligible = true) AS applied "
            "WITH owners, applied AND aliases_match AS applied "
            "FOREACH (owner IN owners | "
            "FOREACH (e IN CASE WHEN applied THEN [owner.node] ELSE [] END | "
            f"SET e += owner.item.props, e.{LOCK} = owner.previous_lock "
            ") "
            "FOREACH (e IN CASE WHEN NOT applied AND owner.created "
            "THEN [owner.node] ELSE [] END | DETACH DELETE e) "
            "FOREACH (e IN CASE WHEN NOT applied AND NOT owner.created "
            "THEN [owner.node] ELSE [] END | "
            f"SET e.{LOCK} = owner.previous_lock)) "
            "WITH applied "
            "UNWIND CASE WHEN size($aliases) = 0 "
            "THEN [null] ELSE $aliases END AS alias_item "
            f"OPTIONAL MATCH (owner:{ENTITY} "
            f"{self._scope.pattern('id: alias_item.entity_id')}) "
            f"OPTIONAL MATCH (existing_alias:{ALIAS} "
            f"{self._scope.pattern('id: alias_item.id')}) "
            "FOREACH (a IN CASE WHEN applied AND existing_alias IS NOT NULL "
            "AND ($allow_alias_moves OR coalesce(alias_item.remove, false)) "
            "THEN [existing_alias] ELSE [] END | DETACH DELETE a) "
            "FOREACH (_ IN CASE WHEN applied AND alias_item IS NOT NULL "
            "AND owner IS NOT NULL AND NOT coalesce(alias_item.remove, false) "
            "THEN [1] ELSE [] END | "
            f"MERGE (a:{ALIAS} {self._scope.pattern('id: alias_item.id')}) "
            f"SET a += alias_item.props{self._scope.stamp_clause('a')} "
            f"MERGE (a)-[r:ALIAS_OF {self._scope.edge()}]->(owner)"
            f" {self._scope.stamp_assignment('r')}) "
            "RETURN applied AS applied"
        )
        result = await self._runner.run(
            QuerySpec(
                cypher,
                {
                    "rows": list(rows),
                    "aliases": aliases,
                    "strict_snapshot": strict_snapshot,
                    "allow_alias_moves": allow_alias_moves,
                    "writer": uuid4().hex,
                },
                idempotent=False,
                tag="resolver_entities_write",
            )
        )
        return bool(result.scalar("applied", False))

    async def write_intent_rows(
        self, entities: Sequence[Entity], aliases: Sequence[Alias]
    ) -> None:
        """Replay stamped merge rows with their aliases in one graph write."""
        entities_by_id = {entity.entity_id: entity for entity in entities}
        aliases_by_owner = _alias_rows_by_owner(aliases)
        for alias in aliases:
            if alias.entity_id not in entities_by_id:
                raise ValueError("merge intent is missing an alias owner row")

        rows = []
        for entity_id in sorted(entities_by_id):
            row = entity_row(entities_by_id[entity_id])
            row.update(
                {
                    "expected_json": "must_exist",
                    "expected_revision": None,
                    "expected_checkpoint": None,
                    "aliases": aliases_by_owner.get(entity_id, []),
                }
            )
            rows.append(row)
        if not await self._write_entity_rows(
            rows, strict_snapshot=False, allow_alias_moves=True
        ):
            raise ConcurrentEntityWriteError(
                "merge intent is older than an entity revision or checkpoint"
            )

    async def write_mentions(
        self, episode_id: str, mentions: Sequence[Mention]
    ) -> None:
        """Write an episode's mentions and recount the touched entities."""
        incoming = {mention.mention_id: mention for mention in mentions}
        rows = await self._runner.run(
            QuerySpec(
                f"MATCH (m:{MENTION} "
                f"{self._scope.pattern('episode_id: $episode_id')}) "
                "RETURN m.id AS id",
                {"episode_id": episode_id},
                read_only=True,
                idempotent=True,
                tag="resolver_mentions_existing",
            )
        )
        existing = {r["id"] for r in rows.rows}
        if existing - incoming.keys():
            raise EpisodeAlreadyResolvedError(episode_id)
        new = [m for mid, m in incoming.items() if mid not in existing]
        if new:
            seq = await self.next_revision()
            stamped = [
                m.model_copy(update={"episode_id": episode_id, "seq": seq}) for m in new
            ]
            await self._runner.run(
                QuerySpec(
                    f"UNWIND $rows AS item MERGE (m:{MENTION} "
                    f"{self._scope.pattern('id: item.id')}) "
                    f"SET m += item.props{self._scope.stamp_clause('m')} "
                    "WITH m, item "
                    f"MATCH (e:{ENTITY} {self._scope.pattern('id: item.entity_id')}) "
                    f"MERGE (m)-[r:OF {self._scope.edge()}]->(e)"
                    f" {self._scope.stamp_assignment('r')}",
                    {"rows": [mention_row(m) for m in stamped]},
                    idempotent=True,
                    tag="resolver_mentions_write",
                )
            )
        await self._recount({m.entity_id for m in incoming.values()})

    async def live_verdict(self, a_id: str, b_id: str) -> EntityVerdict | None:
        """Return the verdict for the pair unless a merge superseded it."""
        lo, hi = sorted((a_id, b_id))
        rows = await self._runner.run(
            QuerySpec(
                f"MATCH (v:{VERDICT} "
                f"{self._scope.pattern('a_id: $lo, b_id: $hi')}) "
                "RETURN v.json AS value ORDER BY v.id LIMIT 1",
                {"lo": lo, "hi": hi},
                read_only=True,
                idempotent=True,
                tag="resolver_verdict",
            )
        )
        if not rows.rows:
            return None
        verdict = EntityVerdict.model_validate_json(rows.rows[0]["value"])
        return None if verdict.superseded_by else verdict

    async def write_verdicts(self, verdicts: Sequence[EntityVerdict]) -> None:
        """Write verdicts, replacing rows with the same id."""
        if not verdicts:
            return
        await self._runner.run(
            QuerySpec(
                f"UNWIND $rows AS item MERGE (v:{VERDICT} "
                f"{self._scope.pattern('id: item.id')}) "
                f"SET v += item.props{self._scope.stamp_clause('v')}",
                {"rows": [verdict_row(v) for v in verdicts]},
                idempotent=True,
                tag="resolver_verdicts_write",
            )
        )

    async def list_for_deduplication(
        self, *, provisional: bool = False, sweep_limit: int = 0
    ) -> list[Entity]:
        """Return the active entities a deduplication run should consider."""
        rows = await self._runner.run(
            QuerySpec(
                f"MATCH (e:{ENTITY} {self._scope.pattern('status: $status')}) "
                "RETURN e.json AS value",
                {"status": EntityStatus.ACTIVE.value},
                read_only=True,
                idempotent=True,
                tag="resolver_dedup_list",
            )
        )
        return deduplication_entities(
            (Entity.model_validate_json(r["value"]) for r in rows.rows),
            provisional=provisional,
            sweep_limit=sweep_limit,
        )

    async def mark_deduplication_checked(self, entities: Sequence[Entity]) -> None:
        """Advance active rows that still match the selected snapshots."""
        if not entities:
            return
        rows = [
            {
                "id": entity.entity_id,
                "revision": entity.revision,
                "snapshot_json": entity.model_dump_json(),
                "checked_json": entity.model_copy(
                    update={"last_checked_version": entity.revision}
                ).model_dump_json(),
            }
            for entity in sorted(entities, key=lambda entity: entity.entity_id)
        ]
        await self._runner.run(
            QuerySpec(
                f"UNWIND $rows AS item MATCH (e:{ENTITY} "
                f"{self._scope.pattern('id: item.id')}) "
                f"WITH e, item, e.{LOCK} AS previous_lock "
                f"SET e.{LOCK} = $writer "
                "WITH e, item, previous_lock, "
                "(e.json = item.snapshot_json "
                "AND e.status = $active_status "
                "AND e.revision = item.revision "
                "AND e.last_checked_version <= item.revision) AS eligible "
                f"SET e.{LOCK} = previous_lock "
                "WITH e, item WHERE eligible "
                "SET e.json = item.checked_json, "
                "e.last_checked_version = item.revision",
                {
                    "rows": rows,
                    "writer": uuid4().hex,
                    "active_status": EntityStatus.ACTIVE.value,
                },
                idempotent=True,
                tag="resolver_dedup_checkpoint",
            )
        )

    async def remove_episode_mentions(self, episode_id: str) -> list[str]:
        """Remove an episode's mentions and recount their entities."""
        rows = await self._runner.run(
            QuerySpec(
                f"MATCH (m:{MENTION} "
                f"{self._scope.pattern('episode_id: $episode_id')}) "
                "RETURN m.id AS id, m.entity_id AS entity_id",
                {"episode_id": episode_id},
                read_only=True,
                idempotent=True,
                tag="resolver_mentions_of_episode",
            )
        )
        ids = [r["id"] for r in rows.rows]
        touched = sorted({r["entity_id"] for r in rows.rows})
        if ids:
            await self._runner.run(
                QuerySpec(
                    f"UNWIND $ids AS id MATCH (m:{MENTION} "
                    f"{self._scope.pattern('id: id')}) "
                    "DETACH DELETE m",
                    {"ids": ids},
                    idempotent=True,
                    tag="resolver_mentions_remove",
                )
            )
        await self._recount(set(touched))
        return touched

    async def mentions_of_episode(self, episode_id: str) -> list[Mention]:
        """Return one episode's mentions, ordered by mention id."""
        rows = await self._runner.run(
            QuerySpec(
                f"MATCH (m:{MENTION} "
                f"{self._scope.pattern('episode_id: $episode_id')}) "
                "RETURN m.json AS value ORDER BY m.id",
                {"episode_id": episode_id},
                read_only=True,
                idempotent=True,
                tag="resolver_mentions_by_episode",
            )
        )
        return [Mention.model_validate_json(row["value"]) for row in rows.rows]

    async def has_merge_record(self, entity_id: str) -> bool:
        """Return whether any merge record names this entity."""
        rows = await self._runner.run(
            QuerySpec(
                f"MATCH (mr:{MERGE} {self._scope.pattern()}) "
                "WHERE mr.survivor_id = $entity_id OR mr.absorbed_id = $entity_id "
                "RETURN count(mr) AS n",
                {"entity_id": entity_id},
                read_only=True,
                idempotent=True,
                tag="resolver_entity_merge_record",
            )
        )
        return bool(rows.scalar("n", 0))

    async def delete_orphans(self, entity_ids: Sequence[str]) -> list[str]:
        """Delete the given entities that are safe to delete."""
        if not entity_ids:
            return []
        rows = await self._runner.run(
            QuerySpec(
                "UNWIND $ids AS id "
                f"MATCH (e:{ENTITY} {self._scope.pattern('id: id')}) "
                f"OPTIONAL MATCH (m:{MENTION} "
                f"{self._scope.pattern('entity_id: id')})"
                f"-[r:OF {self._scope.edge()}]->(e) "
                "WITH e, count(m) AS mentions WHERE mentions = 0 "
                f"OPTIONAL MATCH (e)-[f:FACT {self._scope.edge()}]-() "
                "WHERE f.status <> 'withdrawn' "
                "AND (f.expired_at IS NULL OR f.expired_at >= $open) "
                "WITH e, count(f) AS facts WHERE facts = 0 "
                f"OPTIONAL MATCH (mr:{MERGE} {self._scope.pattern()}) "
                "WHERE mr.survivor_id = e.id OR mr.absorbed_id = e.id "
                "WITH e, count(mr) AS merges WHERE merges = 0 "
                "RETURN e.id AS id ORDER BY e.id",
                {"ids": list(dict.fromkeys(entity_ids)), "open": OPEN_MICROS},
                read_only=True,
                idempotent=True,
                tag="resolver_orphans",
            )
        )
        deleted = [r["id"] for r in rows.rows]
        for entity_id in deleted:
            await self._delete_rows(entity_id)
        return deleted

    async def set_entity_status(self, entity_id: str, status: EntityStatus) -> None:
        """Set the status without bumping the revision or changing `updated_at`."""
        for attempt in range(3):
            entity = await self.get_one(entity_id)
            if entity is None:
                raise NotFoundError(f"entity {entity_id!r}")
            updated = entity.model_copy(update={"status": status})
            try:
                await self.write_entities([updated])
            except ConcurrentEntityWriteError:
                if attempt == 2:
                    raise
            else:
                return

    async def delete_entity(self, entity_id: str) -> None:
        """Delete an entity with its aliases, mentions and tombstone edges."""
        entity = await self.get_one(entity_id)
        if entity is None:
            raise NotFoundError(f"entity {entity_id!r}")
        if await self._live_facts(entity_id):
            raise EntityHasFactsError(entity_id)
        await self._delete_rows(entity_id)

    async def next_revision(self) -> int:
        """Return the next value of the tenant's revision counter.

        The counter is read and written in one statement, so the value is
        never handed out twice. The statement is not idempotent: a replay
        would count the same revision twice, so the store refuses to replay
        it and raises `DatabaseUnavailableError` instead.
        """
        counter_pattern = self._scope.pattern("id: 'revision'")
        rows = await self._runner.run(
            QuerySpec(
                f"MERGE (c:{COUNTER} {counter_pattern}) "
                "ON CREATE SET c.value = 0 "
                "SET c.value = c.value + 1 "
                "RETURN c.value AS value",
                {},
                idempotent=False,
                tag="resolver_revision",
            )
        )
        return int(rows.scalar("value", 1))

    async def active_entities(self) -> list[Entity]:
        """Return the active entities, ordered by id."""
        rows = await self._runner.run(
            QuerySpec(
                f"MATCH (e:{ENTITY} {self._scope.pattern('status: $status')}) "
                "RETURN e.json AS value ORDER BY e.id",
                {"status": EntityStatus.ACTIVE.value},
                read_only=True,
                idempotent=True,
                tag="resolver_active_entities",
            )
        )
        return [Entity.model_validate_json(r["value"]) for r in rows.rows]

    async def list_active_entities(
        self, *, type_ids: frozenset[int] | None, after: str | None, limit: int
    ) -> list[Entity]:
        """Return active entities by id, after ``after``, of the given types."""
        if type_ids == frozenset():
            return []
        conditions = ["e.status = $status"]
        params: dict[str, Any] = {"status": EntityStatus.ACTIVE.value, "limit": limit}
        if type_ids is not None:
            conditions.append("e.ontology_type_id IN $types")
            params["types"] = sorted(type_ids)
        if after is not None:
            conditions.append("e.id > $after")
            params["after"] = after
        rows = await self._runner.run(
            QuerySpec(
                f"MATCH (e:{ENTITY} {self._scope.pattern()}) "
                f"WHERE {' AND '.join(conditions)} "
                "RETURN e.json AS value ORDER BY e.id LIMIT $limit",
                params,
                read_only=True,
                idempotent=True,
                tag="resolver_list_active",
            )
        )
        return [Entity.model_validate_json(row["value"]) for row in rows.rows]

    async def all_entities(self) -> list[Entity]:
        """Return every entity status, ordered by id."""
        rows = await self._runner.run(
            QuerySpec(
                f"MATCH (e:{ENTITY} {self._scope.pattern()}) "
                "RETURN e.json AS value ORDER BY e.id",
                {},
                read_only=True,
                idempotent=True,
                tag="resolver_all_entities",
            )
        )
        return [Entity.model_validate_json(row["value"]) for row in rows.rows]

    async def save_entity(
        self, entity: Entity, *, bump_revision: bool = True
    ) -> Entity:
        """Write one entity and return the stored copy."""
        row = entity.model_copy(deep=True)
        if bump_revision:
            row = row.model_copy(update={"revision": await self.next_revision()})
        await self.write_entities([row])
        return row.model_copy(deep=True)

    async def aliases_of(self, entity_id: str) -> list[Alias]:
        """Return the aliases of `entity_id`, ordered by id."""
        rows = await self._runner.run(
            QuerySpec(
                f"MATCH (a:{ALIAS} {self._scope.pattern('entity_id: $entity_id')}) "
                "RETURN a.json AS value ORDER BY a.id",
                {"entity_id": entity_id},
                read_only=True,
                idempotent=True,
                tag="resolver_aliases_of",
            )
        )
        return [Alias.model_validate_json(r["value"]) for r in rows.rows]

    async def all_aliases(self) -> list[Alias]:
        """Return every alias in this tenant, ordered by alias id."""
        rows = await self._runner.run(
            QuerySpec(
                f"MATCH (a:{ALIAS} {self._scope.pattern()}) "
                "RETURN a.json AS value ORDER BY a.id",
                {},
                read_only=True,
                idempotent=True,
                tag="resolver_all_aliases",
            )
        )
        return [Alias.model_validate_json(row["value"]) for row in rows.rows]

    async def mentions_of(self, entity_id: str) -> list[Mention]:
        """Return the mentions of `entity_id`, ordered by id."""
        rows = await self._runner.run(
            QuerySpec(
                f"MATCH (m:{MENTION} {self._scope.pattern('entity_id: $entity_id')}) "
                "RETURN m.json AS value ORDER BY m.id",
                {"entity_id": entity_id},
                read_only=True,
                idempotent=True,
                tag="resolver_mentions_of",
            )
        )
        return [Mention.model_validate_json(r["value"]) for r in rows.rows]

    async def write_entities(self, entities: Sequence[Entity]) -> None:
        """Write entity rows without allocating a revision."""
        if not entities:
            return
        entities_by_id = {item.entity_id: item for item in entities}
        rows = []
        for entity_id in sorted(entities_by_id):
            entity = entities_by_id[entity_id]
            row = entity_row(entity)
            row.update(
                {
                    "expected_json": None,
                    "expected_revision": None,
                    "expected_checkpoint": None,
                    "aliases": [],
                }
            )
            rows.append(row)
        if not await self._write_entity_rows(rows, strict_snapshot=False):
            raise ConcurrentEntityWriteError(
                "entity revision or checkpoint advanced before the write"
            )

    async def _recount(self, entity_ids: set[str]) -> None:
        if not entity_ids:
            return
        rows = await self._runner.run(
            QuerySpec(
                "UNWIND $ids AS id "
                f"MATCH (e:{ENTITY} {self._scope.pattern('id: id')}) "
                f"OPTIONAL MATCH (m:{MENTION} {self._scope.pattern('entity_id: id')}) "
                "RETURN id AS id, count(m) AS n",
                {"ids": sorted(entity_ids)},
                read_only=True,
                idempotent=True,
                tag="resolver_mentions_count",
            )
        )
        counts = {r["id"]: int(r["n"]) for r in rows.rows}
        for attempt in range(3):
            entities = await self.get_entities(sorted(entity_ids))
            try:
                await self.write_entities(
                    [
                        entity.model_copy(
                            update={"mention_count": counts.get(entity.entity_id, 0)}
                        )
                        for entity in entities
                    ]
                )
            except ConcurrentEntityWriteError:
                if attempt == 2:
                    raise
            else:
                return

    async def get_one(self, entity_id: str) -> Entity | None:
        """Return one entity, or None if its id is absent."""
        rows = await self.get_entities([entity_id])
        return rows[0] if rows else None

    async def _live_facts(self, entity_id: str) -> bool:
        rows = await self._runner.run(
            QuerySpec(
                f"MATCH (e:{ENTITY} {self._scope.pattern('id: $id')})"
                f"-[f:FACT {self._scope.edge()}]-() "
                "WHERE f.status <> 'withdrawn' "
                "AND (f.expired_at IS NULL OR f.expired_at >= $open) "
                "RETURN count(f) AS n",
                {"id": entity_id, "open": OPEN_MICROS},
                read_only=True,
                idempotent=True,
                tag="resolver_live_facts",
            )
        )
        return bool(rows.scalar("n", 0))

    async def _delete_rows(self, entity_id: str) -> None:
        await self._runner.run(
            QuerySpec(
                f"MATCH (e:{ENTITY} {self._scope.pattern('id: $id')}) DETACH DELETE e",
                {"id": entity_id},
                idempotent=True,
                tag="resolver_entity_delete",
            )
        )
        await self._runner.run(
            QuerySpec(
                f"MATCH (a:{ALIAS} "
                f"{self._scope.pattern('entity_id: $id')}) DETACH DELETE a",
                {"id": entity_id},
                idempotent=True,
                tag="resolver_alias_delete",
            )
        )
        await self._runner.run(
            QuerySpec(
                f"MATCH (m:{MENTION} "
                f"{self._scope.pattern('entity_id: $id')}) DETACH DELETE m",
                {"id": entity_id},
                idempotent=True,
                tag="resolver_mention_delete",
            )
        )

    def _follow(self, entity_id: str, found: dict[str, Entity]) -> Entity:
        seen: set[str] = set()
        current = found.get(entity_id)
        while current is not None and current.merged_into is not None:
            if current.entity_id in seen:
                return current
            seen.add(current.entity_id)
            next_entity = found.get(current.merged_into)
            if next_entity is None:
                return current
            current = next_entity
        if current is None:
            raise NotFoundError(f"entity {entity_id!r}")
        return current
