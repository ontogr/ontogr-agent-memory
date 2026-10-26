"""Stage 9: collect what the run found but did not act on. Never fails.

Near misses are the pairs the deduplication stage judged distinct. Each one looked
alike enough to be checked and was kept apart.
"""

from ontogr_mem.consolidation.graph import build_entity_graph
from ontogr_mem.consolidation.models import (
    Proposal,
    ProposalStatus,
    StageResult,
    StageStatus,
)
from ontogr_mem.consolidation.stage import StageContext
from ontogr_mem.consolidation.stages.deduplication import (
    PAIRS_REJECTED,
    DeduplicationStage,
)


class ReportStage:
    """Near misses, stale summaries, disputed facts and pending proposals."""

    name = "report"
    fatal = False

    async def gate(self, ctx: StageContext) -> str | None:
        """Always runs."""
        return None

    async def run(self, ctx: StageContext) -> StageResult:
        """Read the earlier stage results and the stores.

        Store and port reads that fail are recorded in ``errors`` and the stage
        ends PARTIAL. The report never raises, so a broken read cannot fail the
        run.
        """
        notes: list[str] = []
        near_misses = 0
        for result in ctx.results:
            if result.stage == DeduplicationStage.name:
                near_misses += result.counts.get(PAIRS_REJECTED, 0)
        failed = [r.stage for r in ctx.results if r.status is StageStatus.FAILED]
        partial = [r.stage for r in ctx.results if r.status is StageStatus.PARTIAL]
        if failed:
            notes.append("failed stages: " + ", ".join(failed))
        if partial:
            notes.append("partial stages: " + ", ".join(partial))

        stale = disputed = pending = 0
        errors: list[str] = []
        try:
            graph = await build_entity_graph(ctx)
            stale = sum(1 for e in graph.entities.values() if e.summary_stale)
            disputed = sum(1 for f in graph.facts if f.disputed)
            pending = len(
                await ctx.store.find(Proposal, "status", ProposalStatus.PENDING)
            )
        except Exception as exc:
            # The report must not fail the run, so a read error is recorded.
            errors.append(f"report reads failed: {exc}")

        return StageResult(
            stage=self.name,
            status=StageStatus.PARTIAL if errors else StageStatus.OK,
            counts={
                "near_misses": near_misses,
                "stale_summaries": stale,
                "disputed_facts": disputed,
                "pending_proposals": pending,
            },
            notes=notes,
            errors=errors,
        )
