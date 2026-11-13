"""Score the labelled temporal set with offline typed model responses."""

import asyncio
import html
import json
import re
from collections import Counter
from collections.abc import Sequence
from typing import Any

from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage

from ontogr_mem.llm import ModelRole
from tests.eval.labelled import LABELLED, Labelled
from tests.eval.runner import run_eval
from tests.support.scripted_model import (
    ScriptedCall,
    scripted_model,
    scripted_role_models,
)


PAIR_BLOCK = re.compile(r'<pair id="(P\d+)">(.*?)</pair>', re.S)
STATEMENT = re.compile(r"<statement_([ab])[^>]*>(.*?)</statement_[ab]>", re.S)


def _labelled_by_statements(first: str, second: str) -> Labelled:
    """Find the labelled pair whose statements match in either order."""
    wanted = {" ".join(first.split()), " ".join(second.split())}
    return next(item for item in LABELLED if {item.a[0], item.b[0]} == wanted)


def _oracle(truth: dict[str, str], call: ScriptedCall) -> dict[str, Any]:
    """Return the labels for the pair blocks in a scripted model call."""
    answers = []
    for pair_id, block in PAIR_BLOCK.findall(call.prompt):
        sides = dict(STATEMENT.findall(block))
        shown = [html.unescape(sides[side]) for side in ("a", "b")]
        item = _labelled_by_statements(*shown)
        answers.append({"id": pair_id, "verdict": truth[item.id], "confidence": 0.95})
    return {"pairs": answers}


def _always_contradicts(call: ScriptedCall) -> dict[str, Any]:
    """Return a contradiction judgment for every pair in a batch."""
    count = len(PAIR_BLOCK.findall(call.prompt))
    return {
        "pairs": [
            {"id": f"P{number}", "verdict": "contradicts", "confidence": 0.99}
            for number in range(1, count + 1)
        ]
    }


def _usage_model(replies: Sequence[dict[str, Any]]) -> FunctionModel:
    """Return queued structured replies with fixed input and output usage."""
    pending = iter(replies)

    def respond(_messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[ToolCallPart(info.output_tools[0].name, json.dumps(next(pending)))],
            usage=RequestUsage(input_tokens=10, output_tokens=4),
        )

    return FunctionModel(respond)


def test_the_labelled_set_is_unique_and_covers_every_category() -> None:
    """The dataset has unique ids and covers every required label and category."""
    assert len(LABELLED) == 39
    assert len({item.id for item in LABELLED}) == len(LABELLED)
    assert set(Counter(item.label for item in LABELLED)) == {
        "contradicts",
        "compatible",
        "refines",
        "duplicate",
        "unrelated",
    }
    categories = {item.category for item in LABELLED}
    for needed in (
        "exclusive_value",
        "numeric",
        "different_events",
        "scope",
        "different_times",
        "opinion",
        "quote",
        "hypothetical",
        "refinement",
        "paraphrase",
        "injection",
        "negation",
    ):
        assert needed in categories, needed


def test_a_perfect_judge_scores_perfectly() -> None:
    """A judge that answers with the labels has no errors at all."""
    truth = {item.id: item.label for item in LABELLED}
    model, _ = scripted_model(answer=lambda call: _oracle(truth, call))

    report = asyncio.run(run_eval(scripted_role_models(model)))

    assert (report.precision, report.recall) == (1.0, 1.0)
    assert report.false_supersession_rate == 0.0
    assert (report.unanswered, report.wrong) == (0, [])


def test_a_judge_that_always_contradicts_has_a_high_false_supersession_rate():
    """Perfect recall, poor precision, and every non-contradiction is wrong."""
    model, _ = scripted_model(answer=_always_contradicts)

    report = asyncio.run(run_eval(scripted_role_models(model)))

    assert report.recall == 1.0
    assert report.precision < 0.5
    assert report.false_supersession_rate == 1.0
    assert "false-supersession 1.00" in report.render()


def test_retry_usage_counts_malformed_and_valid_responses() -> None:
    """Evaluation totals include failed output validation and the final result."""
    malformed = {"pairs": [{"id": "P1", "verdict": "contradicts"}]}
    model = _usage_model(
        [
            malformed,
            {"pairs": [{"id": "P1", "verdict": "contradicts", "confidence": 0.9}]},
            {"pairs": [{"id": "P1", "verdict": "contradicts", "confidence": 0.95}]},
        ]
    )

    report = asyncio.run(
        run_eval(
            scripted_role_models(**{ModelRole.JUDGE.value: model}),
            items=(LABELLED[0],),
        )
    )

    assert report.calls == 3
    assert (report.tokens_in, report.tokens_out) == (30, 12)
    assert (report.tp, report.unanswered) == (1, 0)


def test_exhausted_malformed_output_is_reported_as_unanswered_with_all_usage():
    """Three invalid output attempts count as three calls and retain their tokens."""
    malformed = {"pairs": [{"id": "P1", "verdict": "contradicts"}]}
    model = _usage_model([malformed, malformed, malformed])

    report = asyncio.run(
        run_eval(
            scripted_role_models(**{ModelRole.JUDGE.value: model}),
            items=(LABELLED[0],),
        )
    )

    assert report.calls == 3
    assert (report.tokens_in, report.tokens_out) == (30, 12)
    assert report.unanswered == 1
    assert report.fn == 1
