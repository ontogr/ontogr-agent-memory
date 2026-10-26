"""Archive and restore entities and facts.

An archived entity gets ``EntityStatus.ARCHIVED``, which the resolution filters
already exclude from matching. A fact keeps its status; an ``ArchiveRecord``
hides it from ``consolidation.graph.current_facts``. Nothing is deleted here.
"""

from ontogr_mem.consolidation.errors import ConsolidationError
from ontogr_mem.consolidation.models import (
    ArchiveRecord,
    ArchiveState,
    ItemKind,
)
from ontogr_mem.consolidation.stage import StageContext
from ontogr_mem.data_models.entities import EntityStatus


async def archive_item(
    ctx: StageContext, target_id: str, kind: ItemKind, reason: str
) -> None:
    """Move one entity or fact to the cold state.

    Raises:
        ConsolidationError: If an entity is archived without a resolution port,
            or if the entity is not active.
    """
    if kind == "entity":
        if ctx.resolution is None:
            raise ConsolidationError("archiving an entity needs a resolution port")
        found = await ctx.resolution.get_entities([target_id])
        if not found or found[0].status is not EntityStatus.ACTIVE:
            raise ConsolidationError(f"entity {target_id!r} is not active")
        # The entity's updated_at is left alone: it is the last time the entity
        # was mentioned, and archiving must not make a cold item look recent.
        await ctx.resolution.archive(target_id)
    await ctx.store.put(
        ArchiveRecord(
            target_id=target_id,
            target_kind=kind,
            archived_at=ctx.clock(),
            reason=reason,
        )
    )
    ctx.invalidate_entity_graph()


async def restore_item(ctx: StageContext, target_id: str) -> None:
    """Bring an archived entity or fact back.

    Raises:
        ConsolidationError: If the item is not archived, or if it is an entity
            and no resolution port is wired.
    """
    record = await ctx.store.get(ArchiveRecord, target_id)
    if record is None or record.state is not ArchiveState.ARCHIVED:
        raise ConsolidationError(f"{target_id!r} is not archived")
    if record.target_kind == "entity":
        if ctx.resolution is None:
            raise ConsolidationError("restoring an entity needs a resolution port")
        found = await ctx.resolution.get_entities([target_id])
        if found and found[0].status is EntityStatus.ARCHIVED:
            # A save with a new revision lets deduplication compare the restored
            # entity again.
            await ctx.resolution.save_entity(
                found[0].model_copy(update={"status": EntityStatus.ACTIVE}),
                bump_revision=True,
            )
    await ctx.store.put(record.model_copy(update={"state": ArchiveState.RESTORED}))
    ctx.invalidate_entity_graph()
