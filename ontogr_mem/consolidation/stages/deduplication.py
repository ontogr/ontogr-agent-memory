"""Stage 2: entity deduplication, run by the entity resolution component.

The stage runs only when an entity changed since its last check, so an idle tenant
makes no model call.
"""

from ontogr_mem.consolidation.models import StageResult, StageStatus
from ontogr_mem.consolidation.stage import StageContext
from ontogr_mem.data_models.entities import EntityStatus, ResolutionState


STAGE = "deduplication"

PAIRS_REJECTED = "pairs_rejected"
"""The count of the pairs the resolver judged distinct."""


class DeduplicationStage:
    """Run ``deduplicate`` on the resolution port."""

    name = STAGE
    fatal = False

    async def gate(self, ctx: StageContext) -> str | None:
        """Skip without resolution, or when no active entity changed."""
        if ctx.resolution is None:
            return "no resolution"
        entities = await ctx.resolution.active_entities()
        changed = any(
            e.status is EntityStatus.ACTIVE
            and (
                e.revision > e.last_checked_version
                or e.resolution_state is ResolutionState.PROVISIONAL
            )
            for e in entities
        )
        if not changed:
            return "no entity changed since its last check"
        return None

    async def run(self, ctx: StageContext) -> StageResult:
        """Deduplicate and report what the resolver did."""
        resolution = ctx.resolution
        if resolution is None:
            return StageResult(
                stage=self.name, status=StageStatus.SKIPPED, notes=["no resolution"]
            )
        report = await resolution.deduplicate()
        ctx.invalidate_entity_graph()
        counts = {
            "selected": report.selected,
            "pairs_considered": report.pairs_considered,
            "pairs_asked": report.pairs_asked,
            "pairs_merged": report.pairs_merged,
            PAIRS_REJECTED: report.pairs_rejected,
        }
        problems = {
            "invalid_ids": report.invalid_ids,
            "parse_failed": report.parse_failed,
            "errors": report.errors,
            "no_verdict": report.no_verdict,
        }
        errors = [f"{name}: {n}" for name, n in problems.items() if n]
        return StageResult(
            stage=self.name,
            status=StageStatus.PARTIAL if errors else StageStatus.OK,
            counts={**counts, **problems},
            errors=errors,
        )
