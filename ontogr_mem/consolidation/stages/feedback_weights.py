"""Stage 3: apply new feedback to the streaming weights, then drop neutral ones."""

from ontogr_mem.consolidation.feedback import (
    NEUTRAL,
    stream_update_weight,
    weight_now,
)
from ontogr_mem.consolidation.models import (
    FeedbackEvent,
    FeedbackWeight,
    StageResult,
    StageStatus,
)
from ontogr_mem.consolidation.stage import StageContext


async def _pending(ctx: StageContext) -> list[FeedbackEvent]:
    """Return unapplied events that still have attempts left, in id order."""
    max_attempts = ctx.settings.feedback.max_attempts
    events = await ctx.store.find(FeedbackEvent, "applied", False)
    return [e for e in events if e.attempts < max_attempts]


class FeedbackWeightsStage:
    """Apply feedback events to the weights. Applying an event again changes nothing."""

    name = "feedback_weights"
    fatal = False

    async def gate(self, ctx: StageContext) -> str | None:
        """Skip when no unapplied event is left to try."""
        return None if await _pending(ctx) else "no new feedback"

    async def run(self, ctx: StageContext) -> StageResult:
        """Apply events in id order, then remove weights that decayed to neutral."""
        cfg = ctx.settings.feedback
        now = ctx.clock()
        applied = already_applied = 0
        for event in await _pending(ctx):
            weight = await ctx.store.get(FeedbackWeight, event.target_id)
            if weight is None:
                weight = FeedbackWeight(target_id=event.target_id, last_update_at=now)
            if event.event_id in weight.applied_ids:
                already_applied += 1
            else:
                alpha = cfg.alpha * (cfg.implicit_factor if event.implicit else 1.0)
                # Decay first, so the update starts from where the weight now stands.
                current = weight_now(weight, now, cfg.decay_beta)
                weight.weight = stream_update_weight(current, event.rating, alpha)
                weight.applied_ids.append(event.event_id)
                weight.last_update_at = now
                await ctx.store.put(weight)
                applied += 1
            # The weight is written before the event is marked. A crash in between is
            # repaired by the applied_ids check on the next run.
            await ctx.store.put(
                event.model_copy(
                    update={"applied": True, "attempts": event.attempts + 1}
                )
            )
        neutral = [
            w
            for w in await ctx.store.list(FeedbackWeight)
            if abs(weight_now(w, now, cfg.decay_beta) - NEUTRAL) < cfg.neutral_epsilon
        ]
        for w in neutral:
            await ctx.store.delete(FeedbackWeight, w.target_id)
        return StageResult(
            stage=self.name,
            status=StageStatus.OK,
            counts={
                "applied": applied,
                "already_applied": already_applied,
                "dropped_neutral": len(neutral),
            },
        )
