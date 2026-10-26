"""Stage 7: importance of entities and current facts."""

from collections import defaultdict
from datetime import datetime

from ontogr_mem.consolidation.config import ScoringSettings
from ontogr_mem.consolidation.feedback import weight_now
from ontogr_mem.consolidation.graph import build_entity_graph
from ontogr_mem.consolidation.importance import importance
from ontogr_mem.consolidation.models import (
    AccessEvent,
    FeedbackWeight,
    ItemKind,
    Score,
    StageResult,
    StageStatus,
)
from ontogr_mem.consolidation.stage import StageContext


async def _drop_stale_scores(ctx: StageContext, keep: set[str]) -> int:
    """Delete the stored scores whose target is not in ``keep``; return the count."""
    stale = [s for s in await ctx.store.list(Score) if s.target_id not in keep]
    for score in stale:
        await ctx.store.delete(Score, score.target_id)
    return len(stale)


def _score_of(
    *,
    kind: ItemKind,
    target_id: str,
    created_at: datetime,
    last_seen: datetime,
    count: int,
    degree: int,
    now: datetime,
    accesses: list[datetime],
    weight: FeedbackWeight | None,
    decay_beta: float,
    cfg: ScoringSettings,
) -> Score:
    """Return the score of one item, counting its recorded accesses."""
    imp = importance(
        last_seen=max([last_seen, *accesses]),
        now=now,
        count=count + len(accesses),
        degree=degree,
        feedback=weight_now(weight, now, decay_beta),
        cfg=cfg,
    )
    return Score(
        target_id=target_id,
        kind=kind,
        created_at=created_at,
        score=imp.score,
        recency=imp.recency,
        frequency=imp.frequency,
        salience=imp.salience,
        computed_at=now,
    )


class ScoringStage:
    """Recompute every score. Scores of items that no longer exist are removed."""

    name = "scoring"
    fatal = False

    async def gate(self, ctx: StageContext) -> str | None:
        """Skip when there is nothing to score."""
        if ctx.resolution is None and ctx.temporal is None:
            return "no resolver or temporal engine"
        return None

    async def run(self, ctx: StageContext) -> StageResult:
        """Write a ``Score`` per entity and per fact."""
        now = ctx.clock()
        graph = await build_entity_graph(ctx)
        accesses: dict[str, list[datetime]] = defaultdict(list)
        for access in await ctx.store.list(AccessEvent):
            accesses[access.target_id].append(access.at)
        weights = {w.target_id: w for w in await ctx.store.list(FeedbackWeight)}

        def score_item(
            *,
            kind: ItemKind,
            target_id: str,
            created_at: datetime,
            last_seen: datetime,
            count: int,
            degree: int,
        ) -> Score:
            return _score_of(
                kind=kind,
                target_id=target_id,
                created_at=created_at,
                last_seen=last_seen,
                count=count,
                degree=degree,
                now=now,
                accesses=accesses.get(target_id, []),
                weight=weights.get(target_id),
                decay_beta=ctx.settings.feedback.decay_beta,
                cfg=ctx.settings.scoring,
            )

        scores = [
            score_item(
                kind="entity",
                target_id=e.entity_id,
                created_at=e.created_at,
                last_seen=e.updated_at,
                count=e.mention_count,
                degree=graph.degree(e.entity_id),
            )
            for e in graph.entities.values()
        ] + [
            score_item(
                kind="fact",
                target_id=f.fact_id,
                created_at=f.recorded_at,
                last_seen=f.recorded_at,
                count=len(f.evidence),
                degree=0,
            )
            for f in graph.facts
        ]
        removed = await _drop_stale_scores(ctx, {s.target_id for s in scores})
        for score in scores:
            await ctx.store.put(score)
        return StageResult(
            stage=self.name,
            status=StageStatus.OK,
            counts={
                "entities": sum(1 for s in scores if s.kind == "entity"),
                "facts": sum(1 for s in scores if s.kind == "fact"),
                "removed": removed,
            },
        )
