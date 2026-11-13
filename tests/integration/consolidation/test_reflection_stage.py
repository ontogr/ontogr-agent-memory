"""Reflection stage behavior on graph tenants with scripted model roles.

The model callback identifies the curator from its unchanged instructions, so
tests do not depend on the wording of the user prompts.
"""

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.profiles import ModelProfile

from ontogr_mem.consolidation.agents import CURATOR
from ontogr_mem.consolidation.config import ConsolidationSettings, ReflectionSettings
from ontogr_mem.consolidation.models import (
    LessonVerdict,
    Proposal,
    ProposalStatus,
    SessionEntry,
    StageStatus,
)
from ontogr_mem.consolidation.signals import record_session_entry
from ontogr_mem.consolidation.stage import StageContext
from ontogr_mem.consolidation.stages.reflection import ReflectionStage
from ontogr_mem.llm import ModelCallFailedError, RoleModels
from tests.support.consolidation import ConsWorld, new_world
from tests.support.graph_backend import GraphBackend
from tests.support.scripted_model import (
    Scripted,
    ScriptedCall,
    scripted_model,
    scripted_role_models,
)


Answer = Callable[[ScriptedCall], Any]


@dataclass(frozen=True, slots=True)
class ModelFixture:
    """A model pool and the calls its scripted model recorded."""

    models: RoleModels
    scripted: Scripted


def _model(answer: Answer, replies: list[Any] | None = None) -> ModelFixture:
    """Build a consolidation-only role pool around one scripted model."""
    model, scripted = scripted_model(replies or (), answer=answer)
    return ModelFixture(scripted_role_models(consolidation=model), scripted)


def _scripted(lessons: list[dict[str, Any]], verdict: str = "accept") -> Answer:
    def answer(call: ScriptedCall) -> dict[str, Any]:
        if call.system == CURATOR:
            return {"lessons": lessons}
        return {"verdict": verdict}

    return answer


def _ctx(world: ConsWorld, fixture: ModelFixture | None = None) -> StageContext:
    return StageContext(
        run_id="r1",
        clock=world.clock,
        store=world.store,
        settings=world.settings,
        models=fixture.models if fixture is not None else scripted_role_models(),
    )


class SlowModel:
    """A model that waits, so concurrent writer calls can be observed."""

    def __init__(self, lessons: list[dict[str, Any]]) -> None:
        """Create an asynchronous function model with fixed stage replies."""
        self.running = 0
        self.peak = 0

        async def respond(
            _messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            self.running += 1
            self.peak = max(self.peak, self.running)
            try:
                await asyncio.sleep(0.01)
                answer = (
                    {"lessons": lessons}
                    if info.instructions == CURATOR
                    else {"verdict": "accept"}
                )
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            info.output_tools[0].name,
                            json.dumps(answer),
                        )
                    ]
                )
            finally:
                self.running -= 1

        profile = ModelProfile(
            default_structured_output_mode="tool",
            supports_json_schema_output=True,
            supports_json_object_output=True,
        )
        model = FunctionModel(respond, profile=profile)
        self.models = scripted_role_models(consolidation=model)


class TestReflectionStage:
    """Session entries become PENDING, ACCEPTED or REJECTED lesson proposals."""

    async def test_gate_skips_without_a_model(
        self, graph_backend: GraphBackend
    ) -> None:
        """The stage is skipped when no model is wired."""
        world = await new_world(
            graph_backend,
        )
        await record_session_entry(
            world.store, "s1", "use km", world.clock(), helpfulness=1
        )

        assert await ReflectionStage().gate(_ctx(world)) == "no model"

    async def test_gate_skips_when_entries_are_below_the_confidence_floor(
        self, graph_backend: GraphBackend
    ) -> None:
        """Entries under the confidence floor do not open the gate."""
        world = await new_world(
            graph_backend,
        )
        fixture = _model(_scripted([]))
        await record_session_entry(
            world.store, "s1", "maybe", world.clock(), helpfulness=1, confidence=0.5
        )

        assert await ReflectionStage().gate(_ctx(world, fixture)) == (
            "no new eligible session entries"
        )
        await ReflectionStage().run(_ctx(world, fixture))
        assert fixture.scripted.calls == []

    async def test_accepted_lesson_becomes_a_pending_proposal(
        self, graph_backend: GraphBackend
    ) -> None:
        """An accepted lesson is stored as PENDING when auto-accept is off."""
        world = await new_world(
            graph_backend,
        )
        entry = await record_session_entry(
            world.store, "s1", "use km please", world.clock(), helpfulness=1
        )
        fixture = _model(
            _scripted(
                [{"text": "User prefers metric units", "entry_ids": [entry.entry_id]}]
            )
        )

        result = await ReflectionStage().run(_ctx(world, fixture))

        assert result.status is StageStatus.OK
        proposals = await world.store.list(Proposal)
        assert [(p.text, p.status) for p in proposals] == [
            ("User prefers metric units", ProposalStatus.PENDING)
        ]
        assert proposals[0].source_entry_ids == [entry.entry_id]

    async def test_accepted_lesson_is_stored_as_accepted_with_auto_accept(
        self,
        graph_backend: GraphBackend,
    ) -> None:
        """An accepted lesson is stored as ACCEPTED when auto-accept is on."""
        settings = ConsolidationSettings(
            reflection=ReflectionSettings(auto_accept=True)
        )
        world = await new_world(graph_backend, settings)
        entry = await record_session_entry(
            world.store, "s1", "use km please", world.clock(), helpfulness=1
        )
        fixture = _model(
            _scripted(
                [{"text": "User prefers metric units", "entry_ids": [entry.entry_id]}]
            )
        )

        await ReflectionStage().run(_ctx(world, fixture))

        proposals = await world.store.list(Proposal)
        assert [p.status for p in proposals] == [ProposalStatus.ACCEPTED]

    @pytest.mark.parametrize(
        "verdict",
        [
            LessonVerdict.ALREADY_KNOWN,
            LessonVerdict.NOT_DURABLE,
            LessonVerdict.UNSUPPORTED,
        ],
    )
    async def test_non_accept_verdict_becomes_rejected_with_the_verdict_as_reason(
        self, graph_backend: GraphBackend, verdict: LessonVerdict
    ) -> None:
        """A writer verdict other than accept rejects the lesson, with the reason."""
        world = await new_world(
            graph_backend,
        )
        entry = await record_session_entry(
            world.store, "s1", "use km please", world.clock(), helpfulness=1
        )
        fixture = _model(
            _scripted(
                [{"text": "User prefers metric units", "entry_ids": [entry.entry_id]}],
                verdict=verdict.value,
            )
        )

        result = await ReflectionStage().run(_ctx(world, fixture))

        proposals = await world.store.list(Proposal)
        assert [(p.status, p.reason) for p in proposals] == [
            (ProposalStatus.REJECTED, verdict.value)
        ]
        assert result.counts["rejected"] == 1
        assert result.counts["proposed"] == 0

    async def test_lesson_without_source_entries_is_unsupported_without_a_call(
        self,
        graph_backend: GraphBackend,
    ) -> None:
        """A lesson that cites no known entry is rejected and needs no writer call."""
        world = await new_world(
            graph_backend,
        )
        await record_session_entry(
            world.store, "s1", "use km please", world.clock(), helpfulness=1
        )
        fixture = _model(
            _scripted(
                [{"text": "Made up claim", "entry_ids": ["se-99999999"]}],
                verdict="accept",
            )
        )

        await ReflectionStage().run(_ctx(world, fixture))

        proposals = await world.store.list(Proposal)
        assert [(p.status, p.reason) for p in proposals] == [
            (ProposalStatus.REJECTED, "unsupported")
        ]
        assert len(fixture.scripted.calls) == 1

    async def test_known_lessons_are_passed_to_the_writer(
        self, graph_backend: GraphBackend
    ) -> None:
        """The writer sees the lessons already stored and can reject a repeat."""
        world = await new_world(
            graph_backend,
        )
        now = world.clock()
        await world.store.put(
            Proposal(
                proposal_id="pr-known",
                session_id="s0",
                text="User prefers metric units",
                source_entry_ids=[],
                status=ProposalStatus.PENDING,
                created_at=now,
            )
        )
        entry = await record_session_entry(
            world.store, "s1", "keep it in km", now, helpfulness=1
        )
        known = "User prefers metric units"

        def writer_knows_the_lesson(call: ScriptedCall) -> dict[str, Any]:
            if call.system == CURATOR:
                return {
                    "lessons": [
                        {
                            "text": "Use metric units in replies",
                            "entry_ids": [entry.entry_id],
                        }
                    ]
                }
            verdict = "already_known" if known in call.prompt else "accept"
            return {"verdict": verdict}

        fixture = _model(writer_knows_the_lesson)
        await ReflectionStage().run(_ctx(world, fixture))

        new = [p for p in await world.store.list(Proposal) if p.session_id == "s1"]
        assert [(p.status, p.reason) for p in new] == [
            (ProposalStatus.REJECTED, "already_known")
        ]

    async def test_a_failed_curator_call_holds_the_watermark_and_the_retry_succeeds(
        self,
        graph_backend: GraphBackend,
    ) -> None:
        """A failed call stores nothing, and the next run distils the same entries."""
        world = await new_world(
            graph_backend,
        )
        entry = await record_session_entry(
            world.store, "s1", "use km please", world.clock(), helpfulness=1
        )
        lessons = [{"text": "User prefers metric units", "entry_ids": [entry.entry_id]}]
        fixture = _model(_scripted(lessons), [ModelCallFailedError("provider down")])

        first = await ReflectionStage().run(_ctx(world, fixture))

        assert first.status is StageStatus.PARTIAL
        assert await world.store.list(Proposal) == []
        assert await ReflectionStage().gate(_ctx(world, fixture)) is None

        second = await ReflectionStage().run(_ctx(world, fixture))

        assert second.status is StageStatus.OK
        assert second.counts["proposed"] == 1
        proposals = await world.store.list(Proposal)
        assert [p.text for p in proposals] == ["User prefers metric units"]

    async def test_the_lessons_of_a_batch_are_judged_at_the_same_time(
        self, graph_backend: GraphBackend
    ) -> None:
        """The writer calls of a batch run together, not one after the other."""
        world = await new_world(
            graph_backend,
        )
        entry = await record_session_entry(
            world.store, "s1", "use km please", world.clock(), helpfulness=1
        )
        lessons = [
            {"text": f"Lesson {index}", "entry_ids": [entry.entry_id]}
            for index in range(3)
        ]
        slow_model = SlowModel(lessons)

        result = await ReflectionStage().run(
            _ctx(world, ModelFixture(slow_model.models, Scripted()))
        )

        assert result.counts["proposed"] == 3
        assert slow_model.peak == 3

    async def test_a_lesson_accepted_in_a_batch_is_known_in_the_next(
        self, graph_backend: GraphBackend
    ) -> None:
        """A lesson accepted in one batch is offered to the writer of the next."""
        settings = ConsolidationSettings(reflection=ReflectionSettings(batch_size=1))
        world = await new_world(graph_backend, settings)
        first = await record_session_entry(
            world.store, "s1", "use km please", world.clock(), helpfulness=1
        )
        second = await record_session_entry(
            world.store, "s1", "and be brief", world.clock(), helpfulness=1
        )
        accepted = "User prefers metric units"

        def answer(call: ScriptedCall) -> dict[str, Any]:
            if call.system == CURATOR:
                if "use km please" in call.prompt:
                    return {
                        "lessons": [{"text": accepted, "entry_ids": [first.entry_id]}]
                    }
                return {
                    "lessons": [
                        {"text": "Replies stay short", "entry_ids": [second.entry_id]}
                    ]
                }
            known = call.prompt.split("Known lessons: ", 1)[1]
            if accepted in known:
                return {"verdict": "already_known"}
            return {"verdict": "accept"}

        fixture = _model(answer)
        result = await ReflectionStage().run(_ctx(world, fixture))

        assert result.counts["proposed"] == 1
        assert result.counts["rejected"] == 1
        proposals = await world.store.list(Proposal)
        assert {p.text: (p.status, p.reason) for p in proposals} == {
            accepted: (ProposalStatus.PENDING, ""),
            "Replies stay short": (ProposalStatus.REJECTED, "already_known"),
        }

    async def test_a_non_llm_error_from_the_curator_fails_the_stage(
        self, graph_backend: GraphBackend
    ) -> None:
        """A non-LLM exception from the curator is not turned into a partial result."""
        world = await new_world(
            graph_backend,
        )
        await record_session_entry(
            world.store, "s1", "use km please", world.clock(), helpfulness=1
        )
        fixture = _model(lambda _: {}, [RuntimeError("client bug")])

        with pytest.raises(RuntimeError, match="client bug"):
            await ReflectionStage().run(_ctx(world, fixture))

    async def test_second_run_with_no_new_entries_makes_zero_calls_and_changes_nothing(
        self,
        graph_backend: GraphBackend,
    ) -> None:
        """Once distilled, a session is not read again and its proposals stay."""
        world = await new_world(
            graph_backend,
        )
        entry = await record_session_entry(
            world.store, "s1", "use km please", world.clock(), helpfulness=1
        )
        fixture = _model(
            _scripted(
                [{"text": "User prefers metric units", "entry_ids": [entry.entry_id]}]
            )
        )
        await ReflectionStage().run(_ctx(world, fixture))
        proposals_before = await world.store.list(Proposal)
        calls_before = len(fixture.scripted.calls)

        assert await ReflectionStage().gate(_ctx(world, fixture)) is not None
        await ReflectionStage().run(_ctx(world, fixture))

        assert len(fixture.scripted.calls) == calls_before
        assert await world.store.list(Proposal) == proposals_before
        assert len(await world.store.list(SessionEntry)) == 1
