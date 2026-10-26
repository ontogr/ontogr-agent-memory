"""Tests for the model judge: two passes, the floor, failures and order.

The model is a scripted ``LLMClient`` behind a real ``LLMGate``. Assertions are on
the verdicts and counters a caller sees, not on the requests the client saw.
"""

import asyncio
import json
from datetime import UTC, datetime

import pytest

from ontogr_mem.llm import LLMCallError, LLMGate, LLMResponse, ModelSize
from ontogr_mem.temporal.judge import (
    JudgeFact,
    LLMPairJudge,
    PairInput,
    build_prompt,
    parse_answers,
)
from ontogr_mem.temporal.report import Outcome
from ontogr_mem.temporal.settings import JudgeSettings
from ontogr_mem.temporal.verdicts import VerdictKind, VerdictSource, pair_key
from tests.support.llm import ScriptedLLM


NOW = datetime(2026, 4, 1, tzinfo=UTC)


def pair(name: str, a_text: str, b_text: str) -> PairInput:
    """A pair of two facts with fixed statements."""
    a = JudgeFact(f"{name}-a", a_text, "2010-01-01 to open", "src")
    b = JudgeFact(f"{name}-b", b_text, "2015-01-01 to open", "src")
    return PairInput(pair_key(a.fact_id, b.fact_id), f"{name}-hash", a, b)


def answer(pair_id: str, verdict: str, confidence: float = 0.9) -> dict:
    """One answer in the reply format the judge asks for."""
    return {
        "id": pair_id,
        "quote_a": "a",
        "quote_b": "b",
        "rule": "exclusive_value",
        "reason": "scripted",
        "kind": "exclusive_value",
        "confidence": confidence,
        "verdict": verdict,
    }


def judge_with(replies=None, *, answer_fn=None, **settings) -> tuple:
    """A judge over a scripted model; returns the judge and the client."""
    llm = ScriptedLLM(replies, answer=answer_fn)
    gate = LLMGate(llm, max_input_tokens=20_000)
    return LLMPairJudge(gate, JudgeSettings(**settings)), llm


async def judge(replies, pairs, **settings):
    """Run one judging pass over ``pairs`` with scripted replies."""
    judge_obj, _ = judge_with(replies, **settings)
    return await judge_obj.judge_pairs(pairs, now=NOW)


class TestLLMPairJudge:
    """Verdicts that come out of the two-pass judge."""

    async def test_contradiction_confirmed_by_the_swapped_pass_is_stored(self):
        """Both passes agree above the floor, so the verdict is contradicts."""
        target = pair("p", "acme", "globex")
        outcome = await judge(
            [
                {"pairs": [answer("P1", "contradicts", 0.9)]},
                {"pairs": [answer("P1", "contradicts", 0.95)]},
            ],
            [target],
        )

        [verdict] = outcome.verdicts
        assert verdict.verdict is VerdictKind.CONTRADICTS
        assert verdict.confidence == 0.9
        assert verdict.votes == (("p1", "contradicts"), ("p2", "contradicts"))
        assert verdict.source is VerdictSource.LLM

    async def test_a_second_pass_that_disagrees_downgrades_to_uncertain(self):
        """A swapped pass that does not confirm leaves the pair uncertain."""
        outcome = await judge(
            [
                {"pairs": [answer("P1", "contradicts", 0.9)]},
                {"pairs": [answer("P1", "compatible", 0.9)]},
            ],
            [pair("p", "acme", "globex")],
        )

        [verdict] = outcome.verdicts
        assert verdict.verdict is VerdictKind.UNCERTAIN

    async def test_a_contradiction_under_the_floor_is_stored_as_uncertain(self):
        """Too unsure to act on, kept as a review item with one vote only."""
        outcome = await judge(
            [{"pairs": [answer("P1", "contradicts", 0.5)]}],
            [pair("p", "acme", "globex")],
        )

        [verdict] = outcome.verdicts
        assert verdict.verdict is VerdictKind.UNCERTAIN
        assert verdict.votes == (("p1", "contradicts"),)

    async def test_a_pair_the_model_skips_is_counted_as_failed_and_not_stored(self):
        """Silence is a failure to retry, never a verdict of no contradiction."""
        outcome = await judge(
            [{"pairs": []}],
            [pair("p", "acme", "globex")],
            reask=0,
        )

        assert outcome.verdicts == []
        assert outcome.counts[Outcome.JUDGE_FAILED.value] == 1

    async def test_a_reply_that_is_not_json_is_counted_as_a_parse_failure(self):
        """Unreadable output fails the batch and stores nothing."""
        outcome = await judge(
            ["not json at all"],
            [pair("p", "acme", "globex")],
            reask=0,
        )

        assert outcome.verdicts == []
        assert outcome.counts[Outcome.JUDGE_PARSE_FAILED.value] == 1

    async def test_an_invented_pair_id_is_counted_and_does_not_answer_for_a_pair(
        self,
    ):
        """An id the judge was not given is dropped and counted."""
        outcome = await judge(
            [{"pairs": [answer("P9", "contradicts")]}],
            [pair("p", "acme", "globex")],
            reask=0,
        )

        assert outcome.verdicts == []
        assert outcome.counts[Outcome.JUDGE_INVALID_ID.value] == 1

    async def test_a_provider_error_fails_the_batch_and_is_counted(self):
        """A failed call counts against its pairs; the pass still returns."""
        llm = ScriptedLLM([LLMCallError("provider down")])
        judge_obj = LLMPairJudge(
            LLMGate(llm, max_input_tokens=20_000),
            JudgeSettings(reask=0),
        )

        outcome = await judge_obj.judge_pairs([pair("p", "acme", "globex")], now=NOW)

        assert outcome.verdicts == []
        assert outcome.counts[Outcome.JUDGE_LLM_ERROR.value] == 1

    @pytest.mark.parametrize("order", [(0, 1), (1, 0)])
    async def test_verdicts_do_not_depend_on_the_order_of_the_pairs(self, order):
        """Shuffled input gives the same verdict for each pair."""
        pairs = [pair("acme", "acme works here", "x"), pair("other", "y", "z")]

        def answer_fn(request):
            verdicts = []
            for number, block in enumerate(request.prompt.split("<pair ")[1:], 1):
                kind = "contradicts" if "acme" in block else "compatible"
                verdicts.append(answer(f"P{number}", kind, 0.9))
            return {"pairs": verdicts}

        judge_obj, _ = judge_with(answer_fn=answer_fn)
        shuffled = [pairs[i] for i in order]
        outcome = await judge_obj.judge_pairs(shuffled, now=NOW)

        by_pair = {v.pair_key: v.verdict for v in outcome.verdicts}
        assert by_pair[pairs[0].pair_key] is VerdictKind.CONTRADICTS
        assert by_pair[pairs[1].pair_key] is VerdictKind.COMPATIBLE

    async def test_separate_model_batches_run_within_gate_concurrency(self):
        """Independent batches overlap while the gate limits active calls."""

        class ConcurrentClient:
            def __init__(self):
                self.active = 0
                self.peak = 0

            async def complete(self, request):
                self.active += 1
                self.peak = max(self.peak, self.active)
                await asyncio.sleep(0)
                self.active -= 1
                return LLMResponse(
                    text=json.dumps({"pairs": [answer("P1", "compatible")]}),
                    input_tokens=1,
                    output_tokens=1,
                )

        client = ConcurrentClient()
        gate = LLMGate(
            client,
            max_input_tokens=20_000,
            concurrency={ModelSize.SMALL: 2},
        )
        judge_obj = LLMPairJudge(gate, JudgeSettings(max_pairs=1, reask=0))
        pairs = [pair(str(index), "one", "two") for index in range(3)]

        outcome = await judge_obj.judge_pairs(pairs, now=NOW)

        assert [verdict.verdict for verdict in outcome.verdicts] == [
            VerdictKind.COMPATIBLE,
            VerdictKind.COMPATIBLE,
            VerdictKind.COMPATIBLE,
        ]
        assert client.peak == 2


class TestParseAnswers:
    """The strict reader of the judge's reply."""

    def test_a_fenced_reply_is_read(self):
        """A Markdown fence around the JSON is tolerated."""
        text = (
            "```json\n"
            '{"pairs": [{"id": "P1", "verdict": "refines", "confidence": 0.8}]}\n'
            "```"
        )

        answers, invalid, parsed = parse_answers(text, 1)

        assert parsed is True
        assert invalid == 0
        assert answers[0].verdict is VerdictKind.REFINES

    def test_an_answer_without_a_confidence_is_a_missing_answer(self):
        """An answer that does not fill the shape asked for is dropped."""
        text = '{"pairs": [{"id": "P1", "verdict": "refines"}]}'

        answers, invalid, parsed = parse_answers(text, 1)

        assert (parsed, invalid, answers) == (True, 0, {})

    def test_the_prompt_lists_each_pair_with_its_id(self):
        """Every pair appears once, numbered from P1."""
        prompt = build_prompt([pair("a", "one", "two"), pair("b", "three", "four")])

        assert prompt.count("<pair id=") == 2
        assert '<pair id="P2">' in prompt
