"""Check that a configured model can judge and confirm a clear contradiction."""

from datetime import UTC, datetime

import pytest
from pydantic_ai import models

from ontogr_mem.llm import LLMConfig, ModelRole, build_models
from ontogr_mem.temporal.judge import JudgeFact, LLMPairJudge, PairInput
from ontogr_mem.temporal.settings import JudgeSettings
from ontogr_mem.temporal.verdicts import VerdictKind, pair_key


pytestmark = [pytest.mark.slow, pytest.mark.enable_socket]


async def test_live_judge_confirms_a_clear_contradiction() -> None:
    """A live judge returns a confirmed contradiction for conflicting locations."""
    config = LLMConfig()
    if config.resolve(ModelRole.JUDGE).model is None:
        pytest.skip("the judge model is not configured")

    pair = PairInput(
        pair_key("fact-a", "fact-b"),
        "live-judge-check",
        JudgeFact("fact-a", "Mira lives in Paris.", "2025-01-01 to open", "test"),
        JudgeFact("fact-b", "Mira lives in Berlin.", "2025-01-01 to open", "test"),
    )

    with models.override_allow_model_requests(True):
        role_models = build_models(config)
        try:
            outcome = await LLMPairJudge(role_models, JudgeSettings()).judge_pairs(
                [pair], now=datetime.now(UTC)
            )
        finally:
            await role_models.aclose()

    assert len(outcome.verdicts) == 1
    assert outcome.verdicts[0].verdict is VerdictKind.CONTRADICTS
    assert outcome.verdicts[0].votes == (
        ("p1", "contradicts"),
        ("p2", "contradicts"),
    )
