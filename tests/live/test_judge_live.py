"""Check that a configured model can judge and confirm a clear contradiction."""

import pytest
from pydantic_ai import models

from ontogr_mem.llm import LLMConfig, ModelRole, build_models
from ontogr_mem.temporal.closure import ClosureJudge
from ontogr_mem.temporal.settings import ClosureSettings


pytestmark = [pytest.mark.slow, pytest.mark.enable_socket]


async def test_live_judge_confirms_a_clear_contradiction() -> None:
    """A live judge returns a confirmed contradiction for conflicting locations."""
    config = LLMConfig()
    if config.resolve(ModelRole.JUDGE).model is None:
        pytest.skip("the judge model is not configured")

    with models.override_allow_model_requests(True):
        role_models = build_models(config)
        try:
            outcome = await ClosureJudge(role_models, ClosureSettings()).judge(
                "Mira lives in Berlin.", [("fact-a", "Mira lives in Paris.")]
            )
        finally:
            await role_models.aclose()

    assert outcome.contradicted == ("fact-a",)
