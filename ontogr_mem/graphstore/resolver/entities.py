"""Entity, alias, mention and verdict storage on a tenant graph."""

from collections.abc import Sequence
from typing import Any

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
from ontogr_mem.graphstore.resolver.rows import (
    alias_row,
    entity_row,
    mention_row,
    verdict_row,
)
from ontogr_mem.graphstore.resolver.selection import deduplication_entities
from ontogr_mem.resolution.blocking import BlockingRow
from ontogr_mem.resolution.errors import (
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
        """Write entities and aliases and stamp each entity with a new revision."""
        revision = await self.next_revision() if entities else None
        stored = [
            entity.model_copy(update={"revision": revision}) for entity in entities
        ]
        if stored:
            await self._runner.run(
                QuerySpec(
                    f"UNWIND $rows AS item MERGE (e:{ENTITY} "
                    f"{self._scope.pattern('id: item.id')}) "
                    f"SET e += item.props{self._scope.stamp_clause('e')}",
                    {"rows": [entity_row(e) for e in stored]},
                    idempotent=True,
                    tag="resolver_entities_write",
                )
            )
        if aliases:
            await self._runner.run(
                QuerySpec(
                    "UNWIND $rows AS item "
                    f"MATCH (e:{ENTITY} {self._scope.pattern('id: item.entity_id')}) "
                    f"MERGE (a:{ALIAS} {self._scope.pattern('id: item.id')}) "
                    f"SET a += item.props{self._scope.stamp_clause('a')} "
                    "WITH a, e "
                    f"MERGE (a)-[r:ALIAS_OF {self._scope.edge()}]->(e)"
                    f" {self._scope.stamp_assignment('r')}",
                    {"rows": [alias_row(a) for a in aliases]},
                    idempotent=True,
                    tag="resolver_aliases_write",
                )
            )
        return [e.model_copy(deep=True) for e in stored]

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
        self, after_revision: int, provisional: bool, sweep_limit: int
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
            after_revision,
            provisional,
            sweep_limit,
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
        entity = await self.get_one(entity_id)
        if entity is None:
            raise NotFoundError(f"entity {entity_id!r}")
        updated = entity.model_copy(update={"status": status})
        await self.write_entities([updated])

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
        await self._runner.run(
            QuerySpec(
                f"UNWIND $rows AS item MERGE (e:{ENTITY} "
                f"{self._scope.pattern('id: item.id')}) "
                f"SET e += item.props{self._scope.stamp_clause('e')}",
                {"rows": [entity_row(e) for e in entities]},
                idempotent=True,
                tag="resolver_entities_write",
            )
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
        entities = await self.get_entities(sorted(entity_ids))
        await self.write_entities(
            [
                entity.model_copy(
                    update={"mention_count": counts.get(entity.entity_id, 0)}
                )
                for entity in entities
            ]
        )

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
            nxt = found.get(current.merged_into)
            if nxt is None:
                return current
            current = nxt
        if current is None:
            raise NotFoundError(f"entity {entity_id!r}")
        return current
