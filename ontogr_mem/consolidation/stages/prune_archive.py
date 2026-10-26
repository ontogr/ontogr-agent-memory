"""Stage 8: archive low-score items; delete after retention only if enabled.

Defaults are conservative: ``dry_run`` is on and deletion is off. An entity past
retention is deleted when no live fact points at it. A fact past retention stays
archived, because the temporal store cannot delete facts, and the result says so.
"""

from collections.abc import Sequence
from datetime import datetime, timedelta

from ontogr_mem.consolidation.archive import archive_item, restore_item
from ontogr_mem.consolidation.errors import ConsolidationError
from ontogr_mem.consolidation.graph import archived_records
from ontogr_mem.consolidation.models import (
    AccessEvent,
    ArchiveRecord,
    ArchiveState,
    Score,
    StageResult,
    StageStatus,
)
from ontogr_mem.consolidation.ports import ResolutionPort
from ontogr_mem.consolidation.stage import StageContext
from ontogr_mem.data_models.entities import EntityStatus
from ontogr_mem.resolution.errors import EntityHasFactsError


async def _latest_access_by_target(ctx: StageContext) -> dict[str, datetime]:
    """Return the time of the latest recorded access of each target."""
    latest: dict[str, datetime] = {}
    for access in await ctx.store.list(AccessEvent):
        previous = latest.get(access.target_id)
        if previous is None or access.at > previous:
            latest[access.target_id] = access.at
    return latest


async def _mentioned_since_archive(
    resolution: ResolutionPort,
    record: ArchiveRecord,
    latest_access: dict[str, datetime],
) -> bool:
    """Return True when the item was read or changed after archiving."""
    last_access = latest_access.get(record.target_id)
    if last_access is not None and last_access > record.archived_at:
        return True
    if record.target_kind == "fact":
        return False
    found = await resolution.get_entities([record.target_id])
    return any(entity.updated_at > record.archived_at for entity in found)


async def _reconcile_entity_records(
    ctx: StageContext, resolution: ResolutionPort
) -> None:
    """Settle the records of entities that resolution restored or that are gone.

    A mention that matches an archived entity makes it active again without
    touching its archive record, and an entity can be deleted elsewhere, for
    example when its last mention is removed.
    """
    records = [r for r in await archived_records(ctx) if r.target_kind == "entity"]
    found = {
        e.entity_id: e
        for e in await resolution.get_entities([r.target_id for r in records])
    }
    for record in records:
        entity = found.get(record.target_id)
        if entity is None:
            await ctx.store.delete(ArchiveRecord, record.target_id)
            await ctx.store.delete(Score, record.target_id)
        elif entity.status is EntityStatus.ACTIVE:
            await ctx.store.put(
                record.model_copy(update={"state": ArchiveState.RESTORED})
            )


async def _delete_entity(
    ctx: StageContext, resolution: ResolutionPort, record: ArchiveRecord
) -> bool:
    """Delete an archived entity and its records; False when a live fact keeps it."""
    try:
        await resolution.delete(record.target_id)
    except EntityHasFactsError:
        return False
    await ctx.store.delete(ArchiveRecord, record.target_id)
    await ctx.store.delete(Score, record.target_id)
    return True


class PruneArchiveStage:
    """Move cold items to the archive; remove old archived entities."""

    name = "prune_archive"
    fatal = False

    async def gate(self, ctx: StageContext) -> str | None:
        """Skip when there are no scores and nothing archived."""
        if not await ctx.store.list(Score) and not await archived_records(ctx):
            return "no scores and nothing archived"
        return None

    async def run(self, ctx: StageContext) -> StageResult:
        """Archive, then apply retention."""
        cfg = ctx.settings.prune
        now = ctx.clock()
        if not cfg.dry_run and ctx.resolution is not None:
            await _reconcile_entity_records(ctx, ctx.resolution)
        archived = await archived_records(ctx)
        archived_ids = {r.target_id for r in archived}
        due = [
            s
            for s in await ctx.store.list(Score)
            if s.score < cfg.archive_below
            and now - s.created_at >= timedelta(days=cfg.min_age_days)
            and s.target_id not in archived_ids
        ]
        expired = [
            r
            for r in archived
            if now - r.archived_at >= timedelta(days=cfg.retention_days)
        ]
        counts = {
            "to_archive": len(due),
            "archived": 0,
            "past_retention": len(expired),
            "deleted": 0,
            "reactivated": 0,
        }
        notes: list[str] = []
        errors: list[str] = []
        if cfg.dry_run:
            notes.append("dry run: nothing changed")
        else:
            for s in due:
                try:
                    await archive_item(ctx, s.target_id, s.kind, f"score {s.score:.3f}")
                    counts["archived"] += 1
                except ConsolidationError as exc:
                    errors.append(f"{s.target_id}: {exc}")
            if cfg.delete_after_retention and ctx.resolution is not None:
                await _apply_retention(ctx, ctx.resolution, expired, counts, notes)
            ctx.invalidate_entity_graph()
        if cfg.dry_run:
            status = StageStatus.DRY_RUN
        else:
            status = StageStatus.PARTIAL if errors else StageStatus.OK
        return StageResult(
            stage=self.name, status=status, counts=counts, notes=notes, errors=errors
        )


async def _apply_retention(
    ctx: StageContext,
    resolution: ResolutionPort,
    expired: Sequence[ArchiveRecord],
    counts: dict[str, int],
    notes: list[str],
) -> None:
    """Restore re-mentioned items, delete expired entities and keep the rest."""
    latest_access = await _latest_access_by_target(ctx)
    for record in expired:
        if await _mentioned_since_archive(resolution, record, latest_access):
            await restore_item(ctx, record.target_id)
            counts["reactivated"] += 1
        elif record.target_kind == "fact":
            notes.append(f"fact {record.target_id} kept: deletion unsupported")
        elif await _delete_entity(ctx, resolution, record):
            counts["deleted"] += 1
        else:
            notes.append(f"{record.target_id} kept: has facts")
