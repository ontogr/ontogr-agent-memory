"""Snapshot purging and unsupported entity-text removal on graph stores."""

from collections.abc import Sequence

from ontogr_mem.data_models.entities import MergeRecord
from ontogr_mem.graphstore.cypher import TenantScope
from ontogr_mem.graphstore.executor import QuerySpec
from ontogr_mem.graphstore.protocol import ScopedRunner
from ontogr_mem.graphstore.resolver.entities import ResolverEntities
from ontogr_mem.graphstore.resolver.merges import ResolverMerges
from ontogr_mem.graphstore.resolver.text import (
    pruned_entity_text,
    purged_merge_snapshot,
)
from ontogr_mem.resolution.errors import ResolutionError


ALIAS = "Alias"
MERGE = "MergeRecord"


class ResolverPrune:
    """Prune stored text using the entity and merge storage operations."""

    def __init__(
        self,
        runner: ScopedRunner,
        scope: TenantScope,
        entities: ResolverEntities,
        merges: ResolverMerges,
    ) -> None:
        """Bind pruning operations to the tenant graph."""
        self._runner = runner
        self._scope = scope
        self._entities = entities
        self._merges = merges

    async def purge_snapshots(
        self, *, entity_id: str | None = None, before_seq: int | None = None
    ) -> int:
        """Blank merge snapshots and return how many records changed."""
        if entity_id is None and before_seq is None:
            raise ResolutionError("purge_snapshots needs entity_id or before_seq")
        rows = await self._runner.run(
            QuerySpec(
                f"MATCH (m:{MERGE} {self._scope.pattern()}) "
                "RETURN m.json AS value ORDER BY m.id",
                {},
                read_only=True,
                idempotent=True,
                tag="resolver_merge_list",
            )
        )
        changed = 0
        for row in rows.rows:
            record = MergeRecord.model_validate_json(row["value"])
            updated = purged_merge_snapshot(
                record, entity_id=entity_id, before_seq=before_seq
            )
            if updated is None:
                continue
            await self._merges.write_merge_record(updated)
            changed += 1
        return changed

    async def prune_entity_text(self, entity_ids: Sequence[str]) -> list[str]:
        """Remove unsupported aliases and blank cached text of existing entities."""
        pruned: list[str] = []
        for entity_id in sorted(set(entity_ids)):
            entity = await self._entities.get_one(entity_id)
            if entity is None:
                continue

            mentions = await self._entities.mentions_of(entity_id)
            aliases = await self._entities.aliases_of(entity_id)
            updated, retained, new_aliases = pruned_entity_text(
                entity, mentions, aliases
            )

            retained_ids = {row.alias_id for row in retained}
            removed_ids = [
                row.alias_id for row in aliases if row.alias_id not in retained_ids
            ]
            if removed_ids:
                await self._runner.run(
                    QuerySpec(
                        f"UNWIND $ids AS id MATCH (a:{ALIAS} "
                        f"{self._scope.pattern('id: id')}) DETACH DELETE a",
                        {"ids": removed_ids},
                        idempotent=True,
                        tag="resolver_aliases_prune",
                    )
                )

            if updated != entity:
                await self._entities.save_entity(updated)
            if new_aliases:
                await self._entities.write_entity_batch([], new_aliases)
            pruned.append(entity_id)
        return pruned
