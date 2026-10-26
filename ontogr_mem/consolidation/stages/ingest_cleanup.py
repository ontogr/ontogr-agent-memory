"""Stage 1 (fatal): finish the temporal writes that a crash left half done.

The temporal engine rolls forward its pending episodes. Retraction and orphan
scrubbing run when an episode is removed, so this stage does not repeat them.
"""

from ontogr_mem.consolidation.models import StageResult, StageStatus
from ontogr_mem.consolidation.stage import StageContext
from ontogr_mem.temporal.report import Outcome


class IngestCleanupStage:
    """Roll forward episodes left pending by an interrupted ingest."""

    name = "ingest_cleanup"
    fatal = True

    async def gate(self, ctx: StageContext) -> str | None:
        """Skip when there is no temporal engine or no pending episode."""
        if ctx.temporal is None:
            return "no temporal engine"
        if not await ctx.temporal.incomplete_episodes():
            return "no incomplete episodes"
        return None

    async def run(self, ctx: StageContext) -> StageResult:
        """Recover the pending episodes and report the slots that failed."""
        temporal = ctx.temporal
        if temporal is None:
            return StageResult(
                stage=self.name,
                status=StageStatus.SKIPPED,
                notes=["no temporal engine"],
            )
        report = await temporal.recover()
        errors = [f"{slot}: {why}" for slot, why in report.failures]
        return StageResult(
            stage=self.name,
            status=StageStatus.FAILED if errors else StageStatus.OK,
            counts={
                "episodes_recovered": report.counts.get(Outcome.EPISODE_RECOVERED, 0),
                "episodes_not_recoverable": report.counts.get(
                    Outcome.EPISODE_NOT_RECOVERABLE, 0
                ),
            },
            errors=errors,
        )
