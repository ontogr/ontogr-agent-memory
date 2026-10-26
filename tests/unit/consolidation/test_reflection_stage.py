"""The reflection stage on the in-memory twins, with scripted curator and writer.

The scripted model tells the curator and the writer apart by the schema of the
request: only the curator's answer has a ``lessons`` list. Every test that needs
the stage to see a model reply uses that split, so no test depends on the
wording of the system prompts.
"""

import asyncio
import json
from collections.abc import Callable
from typing import Any

import pytest

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
from ontogr_mem.llm import LLMClient, LLMGate, LLMRequest, LLMResponse
from ontogr_mem.llm.errors import LLMCallError
from tests.support.consolidation import ConsWorld, new_world
from tests.support.llm import ScriptedLLM


Answer = Callable[[LLMRequest], Any]


def _is_curator(request: LLMRequest) -> bool:
    properties = (request.schema or {}).get("properties", {})
    return "lessons" in properties


def _scripted(lessons: list[dict[str, Any]], verdict: str = "accept") -> Answer:
    def answer(request: LLMRequest) -> dict[str, Any]:
        if _is_curator(request):
            return {"lessons": lessons}
        return {"verdict": verdict}

    return answer


def _gate(llm: LLMClient) -> LLMGate:
    """Build a gate around a client."""
    return LLMGate(llm, max_input_tokens=10_000)


def _ctx(world: ConsWorld, llm: LLMClient | None = None) -> StageContext:
    return StageContext(
        run_id="r1",
        clock=world.clock,
        store=world.store,
        settings=world.settings,
        llm=_gate(llm) if llm is not None else None,
    )


class SlowLLM:
    """A model that waits, so calls that run at the same time can be seen."""

    def __init__(self, answer: Answer) -> None:
        """Keep the answer function."""
        self._answer = answer
        self.requests: list[LLMRequest] = []
        self.running = 0
        self.peak = 0

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Answer after a short wait, and count the calls in flight."""
        self.requests.append(request)
        self.running += 1
        self.peak = max(self.peak, self.running)
        await asyncio.sleep(0.01)
        self.running -= 1
        return LLMResponse(
            text=json.dumps(self._answer(request)),
            input_tokens=len(request.prompt),
            output_tokens=1,
        )


class TestReflectionStage:
    """Session entries become PENDING, ACCEPTED or REJECTED lesson proposals."""

    async def test_gate_skips_without_a_model(self) -> None:
        """The stage is skipped when no model is wired."""
        world = await new_world()
        await record_session_entry(
            world.store, "s1", "use km", world.clock(), helpfulness=1
        )

        assert await ReflectionStage().gate(_ctx(world)) == "no llm"

    async def test_gate_skips_when_entries_are_below_the_confidence_floor(self) -> None:
        """Entries under the confidence floor do not open the gate."""
        world = await new_world()
        llm = ScriptedLLM(answer=_scripted([]))
        await record_session_entry(
            world.store, "s1", "maybe", world.clock(), helpfulness=1, confidence=0.5
        )

        assert await ReflectionStage().gate(_ctx(world, llm)) == (
            "no new eligible session entries"
        )
        await ReflectionStage().run(_ctx(world, llm))
        assert llm.requests == []

    async def test_accepted_lesson_becomes_a_pending_proposal(self) -> None:
        """An accepted lesson is stored as PENDING when auto-accept is off."""
        world = await new_world()
        entry = await record_session_entry(
            world.store, "s1", "use km please", world.clock(), helpfulness=1
        )
        llm = ScriptedLLM(
            answer=_scripted(
                [{"text": "User prefers metric units", "entry_ids": [entry.entry_id]}]
            )
        )

        result = await ReflectionStage().run(_ctx(world, llm))

        assert result.status is StageStatus.OK
        proposals = await world.store.list(Proposal)
        assert [(p.text, p.status) for p in proposals] == [
            ("User prefers metric units", ProposalStatus.PENDING)
        ]
        assert proposals[0].source_entry_ids == [entry.entry_id]

    async def test_accepted_lesson_is_stored_as_accepted_with_auto_accept(
        self,
    ) -> None:
        """An accepted lesson is stored as ACCEPTED when auto-accept is on."""
        settings = ConsolidationSettings(
            reflection=ReflectionSettings(auto_accept=True)
        )
        world = await new_world(settings)
        entry = await record_session_entry(
            world.store, "s1", "use km please", world.clock(), helpfulness=1
        )
        llm = ScriptedLLM(
            answer=_scripted(
                [{"text": "User prefers metric units", "entry_ids": [entry.entry_id]}]
            )
        )

        await ReflectionStage().run(_ctx(world, llm))

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
        self, verdict: LessonVerdict
    ) -> None:
        """A writer verdict other than accept rejects the lesson, with the reason."""
        world = await new_world()
        entry = await record_session_entry(
            world.store, "s1", "use km please", world.clock(), helpfulness=1
        )
        llm = ScriptedLLM(
            answer=_scripted(
                [{"text": "User prefers metric units", "entry_ids": [entry.entry_id]}],
                verdict=verdict.value,
            )
        )

        result = await ReflectionStage().run(_ctx(world, llm))

        proposals = await world.store.list(Proposal)
        assert [(p.status, p.reason) for p in proposals] == [
            (ProposalStatus.REJECTED, verdict.value)
        ]
        assert result.counts["rejected"] == 1
        assert result.counts["proposed"] == 0

    async def test_lesson_without_source_entries_is_unsupported_without_a_call(
        self,
    ) -> None:
        """A lesson that cites no known entry is rejected and needs no writer call."""
        world = await new_world()
        await record_session_entry(
            world.store, "s1", "use km please", world.clock(), helpfulness=1
        )
        llm = ScriptedLLM(
            answer=_scripted(
                [{"text": "Made up claim", "entry_ids": ["se-99999999"]}],
                verdict="accept",
            )
        )

        await ReflectionStage().run(_ctx(world, llm))

        proposals = await world.store.list(Proposal)
        assert [(p.status, p.reason) for p in proposals] == [
            (ProposalStatus.REJECTED, "unsupported")
        ]
        assert len(llm.requests) == 1

    async def test_known_lessons_are_passed_to_the_writer(self) -> None:
        """The writer sees the lessons already stored and can reject a repeat."""
        world = await new_world()
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

        def writer_knows_the_lesson(request: LLMRequest) -> dict[str, Any]:
            if _is_curator(request):
                return {
                    "lessons": [
                        {
                            "text": "Use metric units in replies",
                            "entry_ids": [entry.entry_id],
                        }
                    ]
                }
            verdict = "already_known" if known in request.prompt else "accept"
            return {"verdict": verdict}

        await ReflectionStage().run(
            _ctx(world, ScriptedLLM(answer=writer_knows_the_lesson))
        )

        new = [p for p in await world.store.list(Proposal) if p.session_id == "s1"]
        assert [(p.status, p.reason) for p in new] == [
            (ProposalStatus.REJECTED, "already_known")
        ]

    async def test_a_failed_curator_call_holds_the_watermark_and_the_retry_succeeds(
        self,
    ) -> None:
        """A failed call stores nothing, and the next run distils the same entries."""
        world = await new_world()
        entry = await record_session_entry(
            world.store, "s1", "use km please", world.clock(), helpfulness=1
        )
        lessons = [{"text": "User prefers metric units", "entry_ids": [entry.entry_id]}]
        llm = ScriptedLLM([LLMCallError("provider down")], answer=_scripted(lessons))

        first = await ReflectionStage().run(_ctx(world, llm))

        assert first.status is StageStatus.PARTIAL
        assert await world.store.list(Proposal) == []
        assert await ReflectionStage().gate(_ctx(world, llm)) is None

        second = await ReflectionStage().run(_ctx(world, llm))

        assert second.status is StageStatus.OK
        assert second.counts["proposed"] == 1
        proposals = await world.store.list(Proposal)
        assert [p.text for p in proposals] == ["User prefers metric units"]

    async def test_the_lessons_of_a_batch_are_judged_at_the_same_time(self) -> None:
        """The writer calls of a batch run together, not one after the other."""
        world = await new_world()
        entry = await record_session_entry(
            world.store, "s1", "use km please", world.clock(), helpfulness=1
        )
        lessons = [
            {"text": f"Lesson {index}", "entry_ids": [entry.entry_id]}
            for index in range(3)
        ]
        llm = SlowLLM(_scripted(lessons))

        result = await ReflectionStage().run(_ctx(world, llm))

        assert result.counts["proposed"] == 3
        assert llm.peak == 3

    async def test_a_lesson_accepted_in_a_batch_is_known_in_the_next(self) -> None:
        """A lesson accepted in one batch is offered to the writer of the next."""
        settings = ConsolidationSettings(reflection=ReflectionSettings(batch_size=1))
        world = await new_world(settings)
        first = await record_session_entry(
            world.store, "s1", "use km please", world.clock(), helpfulness=1
        )
        second = await record_session_entry(
            world.store, "s1", "and be brief", world.clock(), helpfulness=1
        )
        accepted = "User prefers metric units"

        def answer(request: LLMRequest) -> dict[str, Any]:
            if _is_curator(request):
                if "use km please" in request.prompt:
                    return {
                        "lessons": [{"text": accepted, "entry_ids": [first.entry_id]}]
                    }
                return {
                    "lessons": [
                        {"text": "Replies stay short", "entry_ids": [second.entry_id]}
                    ]
                }
            known = request.prompt.split("Known lessons: ", 1)[1]
            if accepted in known:
                return {"verdict": "already_known"}
            return {"verdict": "accept"}

        result = await ReflectionStage().run(_ctx(world, ScriptedLLM(answer=answer)))

        assert result.counts["proposed"] == 1
        assert result.counts["rejected"] == 1
        proposals = await world.store.list(Proposal)
        assert {p.text: (p.status, p.reason) for p in proposals} == {
            accepted: (ProposalStatus.PENDING, ""),
            "Replies stay short": (ProposalStatus.REJECTED, "already_known"),
        }

    async def test_a_non_llm_error_from_the_curator_fails_the_stage(self) -> None:
        """A non-LLM exception from the curator is not turned into a partial result."""
        world = await new_world()
        await record_session_entry(
            world.store, "s1", "use km please", world.clock(), helpfulness=1
        )
        llm = ScriptedLLM([RuntimeError("client bug")])

        with pytest.raises(RuntimeError, match="client bug"):
            await ReflectionStage().run(_ctx(world, llm))

    async def test_second_run_with_no_new_entries_makes_zero_calls_and_changes_nothing(
        self,
    ) -> None:
        """Once distilled, a session is not read again and its proposals stay."""
        world = await new_world()
        entry = await record_session_entry(
            world.store, "s1", "use km please", world.clock(), helpfulness=1
        )
        llm = ScriptedLLM(
            answer=_scripted(
                [{"text": "User prefers metric units", "entry_ids": [entry.entry_id]}]
            )
        )
        await ReflectionStage().run(_ctx(world, llm))
        proposals_before = await world.store.list(Proposal)
        calls_before = len(llm.requests)

        assert await ReflectionStage().gate(_ctx(world, llm)) is not None
        await ReflectionStage().run(_ctx(world, llm))

        assert len(llm.requests) == calls_before
        assert await world.store.list(Proposal) == proposals_before
        assert len(await world.store.list(SessionEntry)) == 1
