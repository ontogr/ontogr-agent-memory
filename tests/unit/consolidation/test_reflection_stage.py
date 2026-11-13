"""Offline behavior of lesson curation, review and session watermarks."""

import pytest

from ontogr_mem.consolidation.agents import CURATOR
from ontogr_mem.consolidation.config import ConsolidationSettings
from ontogr_mem.consolidation.models import (
    LessonVerdict,
    Proposal,
    ProposalStatus,
    SessionEntry,
    StageStatus,
    Watermark,
)
from ontogr_mem.consolidation.stage import StageContext
from ontogr_mem.consolidation.stages.reflection import ReflectionStage
from ontogr_mem.llm import ModelCallFailedError, RoleModels
from ontogr_mem.llm.run import TokenUsage
from tests.support.scripted_model import (
    ScriptedCall,
    scripted_model,
    scripted_role_models,
)
from tests.unit.consolidation.stage_fakes import NOW, MemoryStore


def _entry(entry_id: str = "entry-1") -> SessionEntry:
    """Build an eligible session entry."""
    return SessionEntry(
        entry_id=entry_id,
        session_id="session-1",
        seq=1,
        text="Use metric units.",
        helpfulness=1,
        created_at=NOW,
    )


def _context(models: RoleModels, store: MemoryStore) -> StageContext:
    """Build a stage context over an in-memory store."""
    return StageContext(
        run_id="reflection-test",
        clock=lambda: NOW,
        store=store,
        settings=ConsolidationSettings(),
        models=models,
    )


def _answer(call: ScriptedCall) -> dict[str, object]:
    """Return one supported lesson and accept it in the writer."""
    if call.system == CURATOR:
        return {"lessons": [{"text": "Use metric units", "entry_ids": ["entry-1"]}]}
    return {"verdict": "accept"}


class TestReflectionStage:
    """Eligible session entries become reviewed lesson proposals."""

    async def test_supported_lesson_is_saved_and_advances_the_watermark(self) -> None:
        """An accepted lesson keeps its source and advances the session watermark."""
        model, calls = scripted_model(answer=_answer)
        models = scripted_role_models(
            entity_extraction=scripted_model(answer=lambda _: {})[0],
            consolidation=model,
        )
        store = MemoryStore(_entry())

        result = await ReflectionStage().run(_context(models, store))

        proposals = await store.list(Proposal)
        watermark = await store.get(Watermark, "reflection|session-1")
        assert [
            (item.text, item.status, item.source_entry_ids) for item in proposals
        ] == [("Use metric units", ProposalStatus.PENDING, ["entry-1"])]
        assert watermark is not None and watermark.value == "1"
        assert result.status is StageStatus.OK
        assert len(calls.calls) == 2

    async def test_unknown_source_ids_are_filtered_without_a_writer_call(self) -> None:
        """A proposal with no valid source is rejected without writer approval."""
        model, calls = scripted_model(
            [{"lessons": [{"text": "Unsupported lesson", "entry_ids": ["unknown"]}]}]
        )
        models = scripted_role_models(consolidation=model)
        store = MemoryStore(_entry())

        result = await ReflectionStage().run(_context(models, store))

        proposals = await store.list(Proposal)
        assert [
            (item.status, item.reason, item.source_entry_ids) for item in proposals
        ] == [(ProposalStatus.REJECTED, "unsupported", [])]
        assert len(calls.calls) == 1
        assert result.status is StageStatus.OK
        assert await store.get(Watermark, "reflection|session-1") is not None

    async def test_exhausted_curator_output_holds_the_watermark(self) -> None:
        """Three invalid curator replies store no proposals or watermark."""
        model, calls = scripted_model(answer=lambda _: {"lessons": [{"text": ""}]})
        models = scripted_role_models(consolidation=model)
        store = MemoryStore(_entry())

        result = await ReflectionStage().run(_context(models, store))

        assert len(calls.calls) == 3
        assert await store.list(Proposal) == []
        assert await store.get(Watermark, "reflection|session-1") is None
        assert result.status is StageStatus.PARTIAL
        assert result.usage.buckets
        assert next(iter(result.usage.buckets.values())).calls == 3

    async def test_failed_writer_usage_holds_watermark_and_propagates_to_result(
        self,
    ) -> None:
        """Writer failure usage is counted and the session remains eligible."""
        model, _ = scripted_model(
            [
                {"lessons": [{"text": "Use metric units", "entry_ids": ["entry-1"]}]},
                ModelCallFailedError("provider down", usage=TokenUsage(11, 3, 2)),
            ]
        )
        models = scripted_role_models(consolidation=model)
        store = MemoryStore(_entry())

        result = await ReflectionStage().run(_context(models, store))

        bucket = next(iter(result.usage.buckets.values()))
        assert result.status is StageStatus.PARTIAL
        assert await store.list(Proposal) == []
        assert await store.get(Watermark, "reflection|session-1") is None
        assert bucket.input_tokens == 70
        assert bucket.output_tokens == 15
        assert bucket.calls == 3

    async def test_unexpected_model_callback_error_propagates(self) -> None:
        """A programmer error is not converted to a model failure result."""
        model, _ = scripted_model([RuntimeError("callback bug")])
        models = scripted_role_models(consolidation=model)
        store = MemoryStore(_entry())

        with pytest.raises(RuntimeError, match="callback bug"):
            await ReflectionStage().run(_context(models, store))

    async def test_unconfigured_consolidation_role_skips_the_stage(self) -> None:
        """Another configured role does not make reflection available."""
        model, calls = scripted_model(answer=lambda _: {})
        models = scripted_role_models(entity_extraction=model)
        store = MemoryStore(_entry())

        result = await ReflectionStage().run(_context(models, store))

        assert result.status is StageStatus.SKIPPED
        assert calls.calls == []
        assert await store.get(Watermark, "reflection|session-1") is None

    async def test_non_accept_verdict_stores_the_writer_reason(self) -> None:
        """A valid rejection is stored and still advances the watermark."""

        def reject(call: ScriptedCall) -> dict[str, object]:
            response = _answer(call)
            if call.system != CURATOR:
                return {"verdict": LessonVerdict.ALREADY_KNOWN.value}
            return response

        model, _ = scripted_model(answer=reject)
        models = scripted_role_models(consolidation=model)
        store = MemoryStore(_entry())

        result = await ReflectionStage().run(_context(models, store))

        proposal = (await store.list(Proposal))[0]
        assert (proposal.status, proposal.reason) == (
            ProposalStatus.REJECTED,
            LessonVerdict.ALREADY_KNOWN.value,
        )
        assert await store.get(Watermark, "reflection|session-1") is not None
        assert result.status is StageStatus.OK
