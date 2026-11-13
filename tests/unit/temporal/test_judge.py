"""Exercise typed pair judgments, confirmation, and failure accounting."""

import asyncio
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage

from ontogr_mem.llm import LLMConfig, ModelRole, build_models
from ontogr_mem.temporal.judge import (
    JudgeFact,
    LLMPairJudge,
    PairInput,
    build_prompt,
)
from ontogr_mem.temporal.report import Outcome
from ontogr_mem.temporal.settings import JudgeSettings
from ontogr_mem.temporal.verdicts import VerdictKind, VerdictSource, pair_key
from tests.support.scripted_model import scripted_model, scripted_role_models


NOW = datetime(2026, 4, 1, tzinfo=UTC)


def pair(name: str, a_text: str, b_text: str) -> PairInput:
    """A pair of two facts with fixed statements."""
    a = JudgeFact(f"{name}-a", a_text, "2010-01-01 to open", "src")
    b = JudgeFact(f"{name}-b", b_text, "2015-01-01 to open", "src")
    return PairInput(pair_key(a.fact_id, b.fact_id), f"{name}-hash", a, b)


def answer(pair_id: str, verdict: str, confidence: Any = 0.9) -> dict[str, Any]:
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


def judge_with(
    replies: Sequence[Any] | None = None,
    *,
    answer_fn=None,
    **settings: Any,
) -> tuple[LLMPairJudge, Any]:
    """Build a judge with an offline model and return its call recorder."""
    model, calls = scripted_model(replies or (), answer=answer_fn)
    models = scripted_role_models(**{ModelRole.JUDGE.value: model})
    return LLMPairJudge(models, JudgeSettings(**settings)), calls


async def judge(replies: Sequence[Any], pairs: Sequence[PairInput], **settings: Any):
    """Run one judging pass over ``pairs`` with scripted replies."""
    judge_obj, _ = judge_with(replies, **settings)
    return await judge_obj.judge_pairs(pairs, now=NOW)


def usage_model(replies: Sequence[dict[str, Any]]) -> FunctionModel:
    """Return typed-tool responses with fixed token usage for each request."""
    pending = iter(replies)

    def respond(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[ToolCallPart(info.output_tools[0].name, json.dumps(next(pending)))],
            usage=RequestUsage(input_tokens=10, output_tokens=4),
        )

    return FunctionModel(respond)


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
        """A missing answer is counted once and does not trigger a repair call."""
        judge_obj, calls = judge_with([{"pairs": []}])

        outcome = await judge_obj.judge_pairs([pair("p", "acme", "globex")], now=NOW)

        assert outcome.verdicts == []
        assert outcome.counts[Outcome.JUDGE_FAILED.value] == 1
        assert outcome.requests == 1
        assert len(calls.calls) == 1

    async def test_one_malformed_sibling_reasks_the_entire_reply(self):
        """No answer from a partially invalid typed reply is salvaged."""
        judge_obj, calls = judge_with(
            [
                {"pairs": [answer("P1", "compatible"), {"id": "P2"}]},
                {"pairs": [answer("P1", "refines")]},
            ]
        )

        outcome = await judge_obj.judge_pairs([pair("p", "acme", "globex")], now=NOW)

        assert [item.verdict for item in outcome.verdicts] == [VerdictKind.REFINES]
        assert len(calls.calls) == 2
        assert outcome.requests == 2

    async def test_confidence_requires_a_number_and_clamps_numeric_values(self):
        """Strings fail schema validation and numeric confidence is clamped."""
        judge_obj, calls = judge_with(
            [
                {"pairs": [answer("P1", "compatible", "0.7")]},
                {"pairs": [answer("P1", "compatible", 1.4)]},
            ]
        )

        outcome = await judge_obj.judge_pairs([pair("p", "acme", "globex")], now=NOW)

        assert outcome.verdicts[0].confidence == 1.0
        assert len(calls.calls) == 2
        assert outcome.requests == 2

    async def test_boolean_confidence_is_rejected_as_invalid_output(self):
        """Booleans do not pass as numeric confidence values."""
        judge_obj, calls = judge_with(
            [
                {"pairs": [answer("P1", "compatible", True)]},
                {"pairs": [answer("P1", "compatible", 0.7)]},
            ]
        )

        outcome = await judge_obj.judge_pairs([pair("p", "acme", "globex")], now=NOW)

        assert outcome.verdicts[0].confidence == 0.7
        assert len(calls.calls) == 2

    async def test_unknown_ids_are_domain_drops_and_duplicates_use_the_last_answer(
        self,
    ):
        """Valid unknown ids are counted; the last valid duplicate wins."""
        judge_obj, calls = judge_with(
            [
                {
                    "pairs": [
                        answer("P0", "compatible"),
                        answer("P1", "compatible"),
                        answer("P1", "refines"),
                        answer("P2x", "compatible"),
                    ]
                }
            ]
        )

        outcome = await judge_obj.judge_pairs([pair("p", "acme", "globex")], now=NOW)

        assert [item.verdict for item in outcome.verdicts] == [VerdictKind.REFINES]
        assert outcome.counts[Outcome.JUDGE_INVALID_ID.value] == 2
        assert len(calls.calls) == 1

    async def test_exhausted_malformed_output_fails_the_whole_batch(self):
        """Three invalid replies store no partial judgments."""
        malformed = {"pairs": [{"id": "P1", "verdict": "compatible"}]}
        judge_obj, calls = judge_with([malformed, malformed, malformed])

        outcome = await judge_obj.judge_pairs([pair("p", "acme", "globex")], now=NOW)

        assert outcome.verdicts == []
        assert outcome.counts[Outcome.JUDGE_PARSE_FAILED.value] == 1
        assert outcome.counts[Outcome.JUDGE_FAILED.value] == 1
        assert outcome.requests == 3
        assert len(calls.calls) == 3

    async def test_validation_retry_usage_is_included_in_request_and_token_totals(self):
        """Malformed output contributes usage before a valid vote and confirmation."""
        malformed = {"pairs": [{"id": "P1", "verdict": "contradicts"}]}
        models = scripted_role_models(
            **{
                ModelRole.JUDGE.value: usage_model(
                    [
                        malformed,
                        {"pairs": [answer("P1", "contradicts")]},
                        {"pairs": [answer("P1", "contradicts")]},
                    ]
                )
            }
        )
        judge_obj = LLMPairJudge(models, JudgeSettings())

        outcome = await judge_obj.judge_pairs([pair("p", "acme", "globex")], now=NOW)

        assert outcome.requests == 3
        assert (outcome.tokens_in, outcome.tokens_out) == (30, 12)
        assert outcome.verdicts[0].verdict is VerdictKind.CONTRADICTS

    async def test_provider_failure_fails_the_batch_and_is_counted(self):
        """Mapped provider errors fail the batch and preserve the judge result."""
        model, _ = scripted_model([ModelAPIError("model", "provider down")])
        judge_obj = LLMPairJudge(
            scripted_role_models(**{ModelRole.JUDGE.value: model}), JudgeSettings()
        )

        outcome = await judge_obj.judge_pairs([pair("p", "acme", "globex")], now=NOW)

        assert outcome.verdicts == []
        assert outcome.counts[Outcome.JUDGE_LLM_ERROR.value] == 1
        assert outcome.requests == 0

    async def test_failed_swapped_confirmation_does_not_store_a_verdict(self):
        """A contradiction without a successful second pass stays unanswered."""
        judge_obj, _ = judge_with(
            [
                {"pairs": [answer("P1", "contradicts")]},
                {"pairs": []},
            ]
        )

        outcome = await judge_obj.judge_pairs([pair("p", "acme", "globex")], now=NOW)

        assert outcome.verdicts == []
        assert outcome.counts[Outcome.JUDGE_FAILED.value] == 1
        assert outcome.requests == 2

    @pytest.mark.parametrize("order", [(0, 1), (1, 0)])
    async def test_verdicts_do_not_depend_on_the_order_of_the_pairs(self, order):
        """Shuffled input gives the same verdict for each pair."""
        pairs = [pair("acme", "acme works here", "x"), pair("other", "y", "z")]

        def answer_fn(call):
            verdicts = []
            for number, block in enumerate(call.prompt.split("<pair ")[1:], 1):
                kind = "contradicts" if "acme" in block else "compatible"
                verdicts.append(answer(f"P{number}", kind, 0.9))
            return {"pairs": verdicts}

        judge_obj, _ = judge_with(answer_fn=answer_fn)
        shuffled = [pairs[index] for index in order]
        outcome = await judge_obj.judge_pairs(shuffled, now=NOW)

        by_pair = {verdict.pair_key: verdict.verdict for verdict in outcome.verdicts}
        assert by_pair[pairs[0].pair_key] is VerdictKind.CONTRADICTS
        assert by_pair[pairs[1].pair_key] is VerdictKind.COMPATIBLE

    async def test_separate_model_batches_run_within_role_concurrency(self):
        """Independent batches overlap up to the shared role limit."""
        active = 0
        peak = 0

        async def respond(
            _messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0)
            active -= 1
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        info.output_tools[0].name,
                        json.dumps({"pairs": [answer("P1", "compatible")]}),
                    )
                ],
                usage=RequestUsage(input_tokens=1, output_tokens=1),
            )

        models = build_models(
            LLMConfig.model_construct(max_concurrency=2),
            overrides={ModelRole.JUDGE: FunctionModel(respond)},
        )
        judge_obj = LLMPairJudge(models, JudgeSettings(max_pairs=1))
        pairs = [pair(str(index), "one", "two") for index in range(3)]

        outcome = await judge_obj.judge_pairs(pairs, now=NOW)

        assert [verdict.verdict for verdict in outcome.verdicts] == [
            VerdictKind.COMPATIBLE,
            VerdictKind.COMPATIBLE,
            VerdictKind.COMPATIBLE,
        ]
        assert peak == 2
        assert outcome.requests == 3

    def test_the_prompt_lists_each_pair_with_its_id(self):
        """Every pair appears once, numbered from P1."""
        prompt = build_prompt([pair("a", "one", "two"), pair("b", "three", "four")])

        assert prompt.count("<pair id=") == 2
        assert '<pair id="P2">' in prompt
