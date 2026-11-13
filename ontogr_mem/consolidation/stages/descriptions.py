"""Stage 4: regenerate the summary of entities flagged ``summary_stale``.

Short descriptions are joined without a model call. Longer ones go to the model
with the entity's current facts as context. On any model failure the entity is left
unchanged and stays flagged, so the next run tries it again.
"""

import asyncio
from collections import Counter

from ontogr_mem.consolidation.agents import description_agent
from ontogr_mem.consolidation.graph import EntityGraph, build_entity_graph
from ontogr_mem.consolidation.models import StageResult, StageStatus
from ontogr_mem.consolidation.stage import StageContext, record_stage_usage
from ontogr_mem.consolidation.text import quote, truncate_at_sentence
from ontogr_mem.data_models.entities import Entity
from ontogr_mem.data_models.extraction import Usage
from ontogr_mem.llm.failures import ModelFailure
from ontogr_mem.llm.roles import ModelRole
from ontogr_mem.llm.run import run_agent


STAGE = "descriptions"

_CHEAP_LIMIT = 2000
_SUMMARY_LIMIT = 4000


def _joined_description(entity: Entity) -> str:
    """Return the distinct non-empty descriptions of ``entity``, joined by spaces."""
    distinct = dict.fromkeys(d.strip() for d in entity.descriptions if d.strip())
    return " ".join(distinct)


def _facts_by_entity(graph: EntityGraph) -> dict[str, list[str]]:
    """Return the statement of each fact, listed under both of its entities."""
    facts_of: dict[str, list[str]] = {}
    for fact in graph.facts:
        statement = fact.statement or f"{fact.subject_id} -> {fact.object_id}"
        for entity_id in (fact.subject_id, fact.object_id):
            facts_of.setdefault(entity_id, []).append(statement)
    return facts_of


def _model_prompt(entity: Entity, facts: list[str]) -> str:
    """Build the prompt for one summary; every stored text is JSON-quoted."""
    return (
        f"Entity name: {quote(entity.canonical_name)}\n"
        f"Entity type: {quote(entity.type_label)}\n"
        f"Descriptions: {quote(entity.descriptions)}\n"
        f"Facts: {quote(facts)}"
    )


class DescriptionsStage:
    """Rewrite stale summaries."""

    name = STAGE
    fatal = False

    async def gate(self, ctx: StageContext) -> str | None:
        """Skip without resolution, or when no active entity is flagged stale."""
        if ctx.resolution is None:
            return "no resolution"
        entities = await ctx.resolution.active_entities()
        if any(e.summary_stale for e in entities):
            return None
        return "no stale summaries"

    async def run(self, ctx: StageContext) -> StageResult:
        """Rewrite up to ``max_batch`` stale summaries, in entity id order."""
        resolution = ctx.resolution
        if resolution is None:
            return StageResult(
                stage=self.name, status=StageStatus.SKIPPED, notes=["no resolution"]
            )
        cfg = ctx.settings.description
        graph = await build_entity_graph(ctx)
        stale = sorted(
            (e for e in graph.entities.values() if e.summary_stale),
            key=lambda e: e.entity_id,
        )[: cfg.max_batch]
        facts_of = _facts_by_entity(graph)
        usage = Usage()
        errors: list[str] = []

        async def rewrite(entity: Entity) -> str:
            """Rewrite one summary and return its outcome.

            The outcome is "cheap", "llm", "failed" or "missing".
            """
            joined = _joined_description(entity)
            if len(joined) <= _CHEAP_LIMIT:
                text, how = joined, "cheap"
            elif not ctx.models.configured(ModelRole.CONSOLIDATION):
                errors.append(f"{entity.entity_id}: long description needs a model")
                return "failed"
            else:
                facts = facts_of.get(entity.entity_id, [])[: cfg.max_neighbours]
                try:
                    result = await run_agent(
                        description_agent,
                        ModelRole.CONSOLIDATION,
                        _model_prompt(entity, facts),
                        models=ctx.models,
                        deps=None,
                    )
                except ModelFailure as exc:
                    record_stage_usage(usage, STAGE, exc.usage)
                    errors.append(f"{entity.entity_id}: {exc}")
                    return "failed"
                record_stage_usage(usage, STAGE, result.usage)
                text, how = result.output.summary, "llm"
            # Re-read the entity so the write does not overwrite a concurrent change.
            found = await resolution.get_entities([entity.entity_id])
            if not found:
                return "missing"
            fresh = found[0].model_copy(
                update={
                    "summary": truncate_at_sentence(text, _SUMMARY_LIMIT),
                    "summary_stale": False,
                }
            )
            # Written without a new revision: identity did not change, so dedup
            # must not pick this entity up again.
            await resolution.save_entity(fresh, bump_revision=False)
            return how

        outcomes = Counter(await asyncio.gather(*(rewrite(e) for e in stale)))
        ctx.invalidate_entity_graph()
        return StageResult(
            stage=self.name,
            status=StageStatus.PARTIAL if outcomes["failed"] else StageStatus.OK,
            counts={
                "stale": len(stale),
                "rewritten_cheap": outcomes["cheap"],
                "rewritten_llm": outcomes["llm"],
                "failed": outcomes["failed"],
            },
            errors=errors,
            usage=usage,
        )
