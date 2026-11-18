"""Check closure prompt construction, reply filtering, and failure accounting."""

import json
from types import SimpleNamespace

import pytest

from ontogr_mem.llm.failures import ModelCallFailedError, ModelOutputError
from ontogr_mem.llm.run import ModelRun, TokenUsage
from ontogr_mem.temporal import closure
from ontogr_mem.temporal.closure import ClosureJudge
from ontogr_mem.temporal.settings import ClosureSettings


class TestClosureJudge:
    """The judge maps only prompted ids back to stored fact ids."""

    async def test_prompt_has_fenced_facts_without_dates(self, monkeypatch):
        """Include text and candidate ids while omitting temporal metadata."""
        captured = {}

        async def run_agent(_agent, role, prompt, **kwargs):
            captured.update(role=role, prompt=prompt, kwargs=kwargs)
            return ModelRun(SimpleNamespace(contradicted=["F2"]), TokenUsage(3, 2, 1))

        monkeypatch.setattr(closure, "run_agent", run_agent)
        judge = ClosureJudge(None, ClosureSettings(_env_file=None))

        result = await judge.judge(
            "new fact", [("stored-1", "first"), ("stored-2", "second")]
        )

        assert result.contradicted == ("stored-2",)
        assert (
            "Task: list the existing facts that the new fact contradicts"
            in captured["prompt"]
        )
        assert "<new_fact>\nnew fact\n</new_fact>" in captured["prompt"]
        candidate_json = captured["prompt"].split("<existing_facts>\n", 1)[1].split(
            "\n</existing_facts>", 1
        )[0]
        assert json.loads(candidate_json) == [
            {"id": "F1", "text": "first"},
            {"id": "F2", "text": "second"},
        ]
        assert result.input_tokens == 3
        assert result.output_tokens == 2
        assert result.requests == 1

    async def test_unknown_ids_are_counted_and_ignored(self, monkeypatch):
        """Ignore ids outside the candidate list and count each one."""

        async def run_agent(*_args, **_kwargs):
            output = SimpleNamespace(contradicted=["F2", "F9"])
            return ModelRun(output, TokenUsage(0, 0, 1))

        monkeypatch.setattr(closure, "run_agent", run_agent)
        result = await ClosureJudge(None, ClosureSettings(_env_file=None)).judge(
            "new", [("stored", "old")]
        )

        assert result.contradicted == ()
        assert result.invalid_ids == 2

    async def test_duplicate_valid_ids_produce_one_contradiction(self, monkeypatch):
        """Count a prompted fact once while counting invalid ids per occurrence."""

        async def run_agent(*_args, **_kwargs):
            output = SimpleNamespace(contradicted=["F1", "F1", "F9", "F9"])
            return ModelRun(output, TokenUsage(0, 0, 1))

        monkeypatch.setattr(closure, "run_agent", run_agent)
        result = await ClosureJudge(None, ClosureSettings(_env_file=None)).judge(
            "new", [("stored", "old")]
        )

        assert result.contradicted == ("stored",)
        assert result.invalid_ids == 2
        assert result.counts["judge_invalid_id"] == 2

    async def test_empty_reply_has_no_contradictions(self, monkeypatch):
        """An empty list is a successful answer with no contradictions."""

        async def run_agent(*_args, **_kwargs):
            return ModelRun(SimpleNamespace(contradicted=[]), TokenUsage(0, 0, 1))

        monkeypatch.setattr(closure, "run_agent", run_agent)
        judge = ClosureJudge(None, ClosureSettings(_env_file=None))
        result = await judge.judge("new", [])

        assert result.contradicted == ()
        assert not result.failed

    @pytest.mark.parametrize(
        ("error", "counter"),
        [
            (ModelOutputError("bad reply"), "judge_parse_failed"),
            (ModelCallFailedError("provider down"), "judge_llm_error"),
        ],
    )
    async def test_failures_return_failed_outcome_without_verdicts(
        self, monkeypatch, error, counter
    ):
        """Report parsing and provider failures under their distinct counters."""

        async def run_agent(*_args, **_kwargs):
            raise error

        monkeypatch.setattr(closure, "run_agent", run_agent)
        result = await ClosureJudge(None, ClosureSettings(_env_file=None)).judge(
            "new", [("stored", "old")]
        )

        assert result.failed
        assert result.contradicted == ()
        assert result.counts[counter] == 1
        assert result.counts["judge_failed"] == 1
