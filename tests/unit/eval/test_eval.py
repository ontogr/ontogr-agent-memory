"""The labelled set and the evaluation runner, offline with scripted judges."""

import asyncio
import html
import json
import re
from collections import Counter

from ontogr_mem.llm import LLMRequest, LLMResponse
from tests.eval.labelled import LABELLED, Labelled
from tests.eval.runner import run_eval


PAIR_BLOCK = re.compile(r'<pair id="(P\d+)">(.*?)</pair>', re.S)
STATEMENT = re.compile(r"<statement_([ab])[^>]*>(.*?)</statement_[ab]>", re.S)


def _labelled_by_statements(first: str, second: str) -> Labelled:
    """The labelled pair whose two statements are the given texts, in any order."""
    wanted = {" ".join(first.split()), " ".join(second.split())}
    return next(item for item in LABELLED if {item.a[0], item.b[0]} == wanted)


class Oracle:
    """A judge that answers every pair with its label, read from the prompt."""

    def __init__(self, truth: dict[str, str]) -> None:
        """Map each pair id to the verdict the oracle gives it."""
        self.truth = truth

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Answer each pair in the prompt with its label, with high confidence."""
        answers = []
        for pair_id, block in PAIR_BLOCK.findall(request.prompt):
            sides = dict(STATEMENT.findall(block))
            shown = [html.unescape(sides[side]) for side in ("a", "b")]
            item = _labelled_by_statements(*shown)
            answers.append(
                {"id": pair_id, "verdict": self.truth[item.id], "confidence": 0.95}
            )
        return LLMResponse('{"pairs": ' + json.dumps(answers) + "}", 10, 5)


class AlwaysContradicts:
    """A judge that says ``contradicts`` to every pair."""

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Answer every pair in the prompt with contradicts."""
        count = request.prompt.count("<pair ")
        answers = [
            {"id": f"P{n}", "verdict": "contradicts", "confidence": 0.99}
            for n in range(1, count + 1)
        ]
        return LLMResponse('{"pairs": ' + json.dumps(answers) + "}", 10, 5)


def test_the_labelled_set_is_unique_and_covers_every_category() -> None:
    """Ids are unique, all five labels appear and each category is present."""
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

    report = asyncio.run(run_eval(Oracle(truth)))

    assert (report.precision, report.recall) == (1.0, 1.0)
    assert report.false_supersession_rate == 0.0
    assert (report.unanswered, report.wrong) == (0, [])


def test_a_judge_that_always_contradicts_has_a_high_false_supersession_rate() -> None:
    """Perfect recall, poor precision, and every non-contradiction is wrong."""
    report = asyncio.run(run_eval(AlwaysContradicts()))

    assert report.recall == 1.0
    assert report.precision < 0.5
    assert report.false_supersession_rate == 1.0
    assert "false-supersession 1.00" in report.render()
