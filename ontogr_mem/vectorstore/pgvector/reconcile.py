"""Outbox drain, orphan and backfill sweeps, and the episode completion gate."""

from dataclasses import dataclass, field

from ontogr_mem.vectorstore.models import (
    EdgeExtras,
    FailReason,
    ValidityUpdate,
    VectorKind,
)
from ontogr_mem.vectorstore.pgvector.outbox import (
    Outbox,
    OutboxEntry,
    OutboxOp,
    delete_entry,
    upsert_entry,
)
from ontogr_mem.vectorstore.protocol import GraphView
from ontogr_mem.vectorstore.vector_store import VectorStore


MAX_PAGES = 10_000
"""Iteration cap on sweeps."""


@dataclass(frozen=True, slots=True)
class ReconcileConfig:
    """Drain settings.

    Attributes:
        batch: Entries claimed per drain.
        grace_s: How long an upsert waits for its graph object to appear.
    """

    batch: int = 100
    grace_s: float = 30.0


@dataclass(slots=True)
class DrainReport:
    """What one drain did."""

    claimed: int = 0
    upserted: int = 0
    unchanged: int = 0
    validity_applied: int = 0
    validity_missed: int = 0
    deleted: int = 0
    dropped_missing: int = 0
    deferred: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class SweepReport:
    """Result of an orphan or backfill sweep.

    Attributes:
        scanned: Objects the sweep read.
        changed: Rows deleted or entries enqueued.
        pages: Keyset pages read.
        last_id: Last id read, usable as the next `after`.
        capped: True when `max_pages` stopped the sweep early.
    """

    scanned: int
    changed: int
    pages: int
    last_id: str | None
    capped: bool = False


class Reconciler:
    """Embeds graph-side intents and repairs drift between the two stores.

    Args:
        store: The pgvector store.
        outbox: The queue the graph side writes.
        graph: The graph the vectors mirror.
        config: Batch size and grace window.
    """

    def __init__(
        self,
        store: VectorStore,
        outbox: Outbox,
        graph: GraphView,
        config: ReconcileConfig | None = None,
    ) -> None:
        """Bind the store, the queue and the graph."""
        self.store = store
        self.outbox = outbox
        self.graph = graph
        self.config = config or ReconcileConfig()

    async def drain(self, limit: int | None = None) -> DrainReport:
        """Process one batch; unfinished entries are released if an error escapes."""
        report = DrainReport()
        entries = await self.outbox.claim(limit or self.config.batch)
        report.claimed = len(entries)
        unfinished = list(entries)
        try:
            for kind in VectorKind:
                upserts = [
                    entry
                    for entry in entries
                    if entry.kind is kind and entry.op is OutboxOp.UPSERT
                ]
                if upserts:
                    await self._upserts(kind, upserts, report, unfinished)
            validity = [entry for entry in entries if entry.op is OutboxOp.VALIDITY]
            if validity:
                await self._validity(validity, report, unfinished)
            for kind in VectorKind:
                deletes = [
                    entry
                    for entry in entries
                    if entry.kind is kind and entry.op is OutboxOp.DELETE
                ]
                if deletes:
                    report.deleted += await self.store.delete(
                        kind, [entry.object_id for entry in deletes]
                    )
                    await self.outbox.ack(deletes)
                    _forget(unfinished, deletes)
        finally:
            if unfinished:
                await self.outbox.release(unfinished)
        self.store.metrics.inc("outbox.drained", report.claimed)
        return report

    async def _upserts(
        self,
        kind: VectorKind,
        entries: list[OutboxEntry],
        report: DrainReport,
        unfinished: list[OutboxEntry],
    ) -> None:
        present = await self.graph.exists(kind, [entry.object_id for entry in entries])
        live = [entry for entry in entries if entry.object_id in present]
        missing = [entry for entry in entries if entry.object_id not in present]
        old = [entry for entry in missing if entry.age_s >= self.config.grace_s]
        young = [entry for entry in missing if entry.age_s < self.config.grace_s]
        if old:
            await self.outbox.ack(old)
            _forget(unfinished, old)
            report.dropped_missing += len(old)
        if young:
            await self.outbox.release(young)
            _forget(unfinished, young)
            report.deferred += len(young)
        if not live:
            return
        result = await self.store.upsert(kind, [entry.to_item() for entry in live])
        bad = {item.id: item for item in result.failed}
        done = [entry for entry in live if entry.object_id not in bad]
        if kind is VectorKind.EDGE:
            mirror = [
                ValidityUpdate(entry.object_id, entry.extras.validity)
                for entry in done
                if isinstance(entry.extras, EdgeExtras)
            ]
            if mirror:
                await self.store.update_validity(mirror)
        if done:
            await self.outbox.ack(done)
        grouped: dict[str, list[OutboxEntry]] = {}
        for entry in live:
            item = bad.get(entry.object_id)
            if item is None:
                continue
            detail = f"{item.reason.value}: {item.detail}"
            if item.reason is FailReason.EMPTY_TEXT:
                await self.outbox.ack([entry])
                report.dropped_missing += 1
            else:
                grouped.setdefault(detail, []).append(entry)
                report.failed.append((entry.object_id, detail))
        for detail, failed in grouped.items():
            await self.outbox.fail(failed, detail)
        _forget(unfinished, live)
        report.upserted += result.written
        report.unchanged += result.unchanged

    async def _validity(
        self,
        entries: list[OutboxEntry],
        report: DrainReport,
        unfinished: list[OutboxEntry],
    ) -> None:
        ids = [entry.object_id for entry in entries]
        present = await self.store.exists(VectorKind.EDGE, ids)
        apply = [
            entry
            for entry in entries
            if entry.object_id in present and entry.validity is not None
        ]
        missed = [entry for entry in entries if entry not in apply]
        if apply:
            report.validity_applied += await self.store.update_validity(
                [
                    ValidityUpdate(entry.object_id, entry.validity)
                    for entry in apply
                    if entry.validity is not None
                ]
            )
            await self.outbox.ack(apply)
        if missed:
            await self.outbox.ack(missed)
            report.validity_missed += len(missed)
        _forget(unfinished, entries)

    async def sweep_orphans(
        self, kind: VectorKind, *, page: int = 1000, max_pages: int = MAX_PAGES
    ) -> SweepReport:
        """Delete vector rows whose graph object no longer exists."""
        after: str | None = None
        scanned = deleted = pages = 0
        while pages < max_pages:
            ids = await self.store.iter_ids(kind, after, page)
            if not ids:
                self.store.metrics.inc(f"{kind.value}.orphans_deleted", deleted)
                return SweepReport(scanned, deleted, pages, after)
            pages += 1
            scanned += len(ids)
            present = await self.graph.exists(kind, ids)
            orphans = [object_id for object_id in ids if object_id not in present]
            if orphans:
                deleted += await self.store.delete(kind, orphans)
            after = ids[-1]
        self.store.metrics.inc(f"{kind.value}.orphans_deleted", deleted)
        return SweepReport(scanned, deleted, pages, after, capped=True)

    async def backfill(
        self, kind: VectorKind, *, page: int = 500, max_pages: int = MAX_PAGES
    ) -> SweepReport:
        """Enqueue graph objects that have no vector or a stale one."""
        after: str | None = None
        scanned = queued = pages = 0
        while pages < max_pages:
            items = await self.graph.iter_items(kind, after, page)
            if not items:
                return SweepReport(scanned, queued, pages, after)
            pages += 1
            scanned += len(items)
            stale = await self.store.needs_embedding(kind, items)
            if stale:
                await self.outbox.enqueue([upsert_entry(kind, item) for item in stale])
                queued += len(stale)
            after = items[-1].id
        return SweepReport(scanned, queued, pages, after, capped=True)

    async def delete_ids(self, kind: VectorKind, ids: list[str]) -> None:
        """Enqueue deletes for objects the graph removed."""
        await self.outbox.enqueue([delete_entry(kind, object_id) for object_id in ids])


def _forget(unfinished: list[OutboxEntry], done: list[OutboxEntry]) -> None:
    keys = {entry.key for entry in done}
    unfinished[:] = [entry for entry in unfinished if entry.key not in keys]


async def episode_ready(outbox: Outbox, episode_id: str) -> bool:
    """Return false while any outbox entry of the episode remains."""
    return await outbox.pending_for_episode(episode_id) == 0
