"""Offline behavior of stale entity summary rewriting."""

from ontogr_mem.consolidation.config import ConsolidationSettings
from ontogr_mem.consolidation.models import StageStatus
from ontogr_mem.consolidation.stage import StageContext
from ontogr_mem.consolidation.stages.descriptions import DescriptionsStage
from ontogr_mem.data_models.extraction import TokenCount, Usage
from ontogr_mem.llm import ModelCallFailedError, RoleModels
from ontogr_mem.llm.roles import ModelRole
from ontogr_mem.llm.run import TokenUsage
from tests.support.scripted_model import scripted_model, scripted_role_models
from tests.unit.consolidation.stage_fakes import (
    NOW,
    MemoryResolution,
    MemoryStore,
    MemoryTemporal,
    entity,
)


LONG = "Founded in Berlin in 2010. " * 100


def _context(
    models: RoleModels,
    resolution: MemoryResolution,
    *,
    entity_store: MemoryStore | None = None,
) -> StageContext:
    """Build a stage context over in-memory ports."""
    return StageContext(
        run_id="descriptions-test",
        clock=lambda: NOW,
        store=entity_store or MemoryStore(),
        settings=ConsolidationSettings(),
        resolution=resolution,
        temporal=MemoryTemporal(),
        models=models,
    )


def _token_count(usage: Usage) -> TokenCount:
    """Return the only usage bucket recorded by a stage result."""
    assert len(usage.buckets) == 1
    return next(iter(usage.buckets.values()))


class TestDescriptionsStage:
    """Stale summaries use cheap joins or one typed model run."""

    async def test_short_descriptions_are_joined_without_a_model(self) -> None:
        """Short distinct descriptions refresh the summary without model usage."""
        resolution = MemoryResolution(
            entity("e1", ["Works at Acme.", "Works at Acme.", "Lives in Berlin."])
        )

        result = await DescriptionsStage().run(
            _context(scripted_role_models(), resolution)
        )

        stored = resolution.entities["e1"]
        assert stored.summary == "Works at Acme. Lives in Berlin."
        assert stored.summary_stale is False
        assert result.status is StageStatus.OK
        assert result.counts["rewritten_cheap"] == 1
        assert result.usage.buckets == {}
        assert resolution.bump_revisions == [False]

    async def test_malformed_output_is_reasked_and_rewrites_from_the_latest_entity(
        self,
    ) -> None:
        """A valid re-ask saves the summary without overwriting a concurrent edit."""
        current = entity("e1", [LONG])
        resolution = MemoryResolution(current)

        def concurrent_edit(_call: object) -> dict[str, str]:
            resolution.entities["e1"] = current.model_copy(
                update={"descriptions": [LONG, "Added while rewriting."]}
            )
            return {"summary": "Recovered summary."}

        model, calls = scripted_model([{"summary": ""}], answer=concurrent_edit)
        models = scripted_role_models(consolidation=model)

        result = await DescriptionsStage().run(_context(models, resolution))

        stored = resolution.entities["e1"]
        assert len(calls.calls) == 2
        assert stored.summary == "Recovered summary."
        assert stored.descriptions == [LONG, "Added while rewriting."]
        assert stored.summary_stale is False
        assert result.status is StageStatus.OK
        assert result.counts["rewritten_llm"] == 1
        assert _token_count(result.usage).calls == 2
        assert resolution.bump_revisions == [False]

    async def test_exhausted_output_keeps_the_entity_stale(self) -> None:
        """Three invalid summaries leave the stored entity unchanged."""
        current = entity("e1", [LONG])
        resolution = MemoryResolution(current)
        model, calls = scripted_model(answer=lambda _: {"summary": ""})
        models = scripted_role_models(consolidation=model)

        result = await DescriptionsStage().run(_context(models, resolution))

        assert len(calls.calls) == 3
        assert resolution.entities["e1"] == current
        assert result.status is StageStatus.PARTIAL
        assert result.counts["failed"] == 1
        assert _token_count(result.usage).calls == 3
        assert resolution.bump_revisions == []

    async def test_failed_run_usage_is_recorded_and_entity_stays_stale(self) -> None:
        """Provider failure usage is retained while the summary remains stale."""
        current = entity("e1", [LONG])
        resolution = MemoryResolution(current)
        model, _ = scripted_model(
            [ModelCallFailedError("provider down", usage=TokenUsage(17, 4, 2))]
        )
        models = scripted_role_models(consolidation=model)

        result = await DescriptionsStage().run(_context(models, resolution))

        bucket = _token_count(result.usage)
        assert result.status is StageStatus.PARTIAL
        assert resolution.entities["e1"] == current
        assert bucket.input_tokens == 17
        assert bucket.output_tokens == 4
        assert bucket.calls == 2
        assert next(iter(result.usage.buckets)) == (
            f"descriptions/{ModelRole.CONSOLIDATION.value}"
        )

    async def test_long_description_without_a_consolidation_model_stays_stale(
        self,
    ) -> None:
        """An unconfigured consolidation role cannot rewrite a long summary."""
        current = entity("e1", [LONG])
        resolution = MemoryResolution(current)

        result = await DescriptionsStage().run(
            _context(scripted_role_models(), resolution)
        )

        assert resolution.entities["e1"] == current
        assert result.status is StageStatus.PARTIAL
        assert result.counts["failed"] == 1
        assert result.usage.buckets == {}
