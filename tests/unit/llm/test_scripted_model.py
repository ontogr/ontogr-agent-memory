"""Tests for the scripted model double."""

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent

from ontogr_mem.llm import ModelRole, run_agent
from tests.support.scripted_model import scripted_model, scripted_role_models


class Answer(BaseModel):
    """Test data."""

    value: int


async def test_exhausted_queue_raises_and_fallback_runs() -> None:
    """Exhausted queue raises and fallback runs."""
    fn, rec = scripted_model([{"value": 1}], answer=lambda call: {"value": 9})
    models = scripted_role_models(fn)
    agent = Agent[None, Answer](output_type=Answer)
    first = await run_agent(agent, ModelRole.JUDGE, "a", models=models, deps=None)
    second = await run_agent(agent, ModelRole.JUDGE, "b", models=models, deps=None)
    assert (first.output.value, second.output.value) == (1, 9)
    assert [c.prompt for c in rec.calls] == ["a", "b"]
    empty, _ = scripted_model()
    with pytest.raises(AssertionError, match="no scripted reply"):
        await run_agent(
            agent, ModelRole.JUDGE, "a", models=scripted_role_models(empty), deps=None
        )


async def test_per_role_models_override_the_shared_one() -> None:
    """Per role models override the shared one."""
    shared, shared_rec = scripted_model([{"value": 1}])
    special, special_rec = scripted_model([{"value": 2}])
    models = scripted_role_models(shared, judge=special)
    agent = Agent[None, Answer](output_type=Answer)
    a = await run_agent(agent, ModelRole.JUDGE, "p", models=models, deps=None)
    b = await run_agent(agent, ModelRole.ANSWER, "p", models=models, deps=None)
    assert (a.output.value, b.output.value) == (2, 1)
    assert len(special_rec.calls) == len(shared_rec.calls) == 1


@pytest.mark.parametrize("with_endpoint", [False, True])
def test_ambient_model_settings_do_not_configure_unlisted_roles(
    monkeypatch: pytest.MonkeyPatch, with_endpoint: bool
) -> None:
    """Ambient model settings do not configure roles missing from the overrides."""
    monkeypatch.setenv("MEMORY_LLM_MODEL", "m")
    if with_endpoint:
        monkeypatch.setenv("MEMORY_LLM_BASE_URL", "http://x/v1")
        monkeypatch.setenv("MEMORY_LLM_API_KEY", "k")
    judge, _ = scripted_model()
    pool = scripted_role_models(judge=judge)
    assert [role for role in ModelRole if pool.configured(role)] == [ModelRole.JUDGE]
    assert pool.config_for(ModelRole.FACT_EXTRACTION).max_output_tokens == 16384
    shared, _ = scripted_model()
    assert all(scripted_role_models(shared).configured(role) for role in ModelRole)
