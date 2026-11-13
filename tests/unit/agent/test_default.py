"""Default assistant agent: its model, instructions, capability and settings."""

from typing import Any

import pytest
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.models.function import FunctionModel

from ontogr_mem.agent import MemoryCapability, build_default_agent
from ontogr_mem.agent.instructions import DEFAULT_INSTRUCTIONS
from ontogr_mem.llm import (
    LLMConfig,
    ModelNotConfiguredError,
    ModelRole,
    RoleOverride,
    build_models,
)
from tests.support.fake_memory import FakeMemory, fact_context
from tests.support.scripted_model import Scripted, scripted_model, scripted_role_models


@pytest.fixture(autouse=True)
def quiet_pydantic_ai(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the pydantic_ai startup banner out of test output."""
    monkeypatch.setenv("PYDANTIC_AI_NO_BANNER", "1")


def _text_model() -> tuple[FunctionModel, Scripted]:
    """Return a scripted model that replies "ok" in plain text.

    The default tool output mode needs an output tool, and a str output type
    declares none, so the scripted model replies in prompted mode.
    """
    return scripted_model(["ok"], mode="prompted")


def _user_prompt_texts(messages: list[ModelMessage]) -> list[str]:
    """Return the text of every user prompt part in the request messages."""
    return [
        part.content
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart) and isinstance(part.content, str)
    ]


class TestDefaultAgent:
    """Build the default assistant agent from a memory client."""

    def test_raises_when_no_model_is_configured(self) -> None:
        """Building fails when the model pool has no configured model."""
        memory = FakeMemory(models=build_models(LLMConfig.model_construct()))
        with pytest.raises(ModelNotConfiguredError):
            build_default_agent(memory)

    def test_raises_when_only_other_roles_have_models(self) -> None:
        """Building fails when the answer role is unset but another role is set."""
        model, _ = _text_model()
        memory = FakeMemory(models=scripted_role_models(judge=model))
        with pytest.raises(ModelNotConfiguredError):
            build_default_agent(memory)

    @pytest.mark.parametrize(
        ("instructions", "expected"),
        [
            (None, DEFAULT_INSTRUCTIONS),
            ("Reply in one word.", "Reply in one word."),
            ("", ""),
        ],
    )
    async def test_model_sees_caller_instructions_or_the_default(
        self, instructions: str | None, expected: str
    ) -> None:
        """The model sees the caller's instructions, or the default text if none."""
        model, recorder = _text_model()
        memory = FakeMemory(models=scripted_role_models(model))
        agent = build_default_agent(memory, instructions=instructions)
        await agent.run("hello")
        assert recorder.calls[0].system == expected

    async def test_retrieved_memory_is_a_request_part_not_instructions(self) -> None:
        """Retrieved memory reaches the model in the request, not the instructions."""
        model, recorder = _text_model()
        memory = FakeMemory(
            context=fact_context(("fact-1", "Ana works at Acme")),
            models=scripted_role_models(model),
        )
        agent = build_default_agent(memory)
        await agent.run("Where does Ana work?")
        call = recorder.calls[0]
        assert "Ana works at Acme" not in call.system
        assert any(
            "[fact-1] Ana works at Acme" in text
            for text in _user_prompt_texts(call.messages)
        )

    async def test_supplied_capability_sets_the_tools_and_writes(self) -> None:
        """A supplied capability decides the tools offered and the writes made."""
        model, recorder = _text_model()
        memory = FakeMemory(models=scripted_role_models(model))
        capability = MemoryCapability(memory, tools=(), write=False)
        agent = build_default_agent(memory, capability=capability)
        result = await agent.run("hello")
        assert result.output == "ok"
        assert recorder.calls[0].info.function_tools == []
        assert memory.texts == []

    def test_rejects_a_capability_built_on_another_memory(self) -> None:
        """The capability must read from the same memory client as the agent."""
        model, _ = _text_model()
        memory = FakeMemory(models=scripted_role_models(model))
        capability = MemoryCapability(FakeMemory(), tools=())
        with pytest.raises(ValueError):
            build_default_agent(memory, capability=capability)

    async def test_default_answer_settings_send_only_the_timeout(self) -> None:
        """Without answer overrides the model gets the default timeout and no cap."""
        model, recorder = _text_model()
        memory = FakeMemory(models=scripted_role_models(model))
        agent = build_default_agent(memory)
        await agent.run("hello")
        settings: dict[str, Any] = dict(recorder.calls[0].info.model_settings or {})
        assert settings["timeout"] == 300.0
        assert "max_tokens" not in settings
        assert "openai_reasoning_effort" not in settings
        assert "extra_body" not in settings

    async def test_configured_answer_settings_reach_the_model(self) -> None:
        """Answer settings reach the model, and the extraction role's do not."""
        config = LLMConfig.model_construct(
            roles={
                ModelRole.ANSWER: RoleOverride(
                    timeout_s=42,
                    max_output_tokens=500,
                    reasoning_effort="low",
                    extra_body={"thinking": {"type": "disabled"}},
                ),
                ModelRole.ENTITY_EXTRACTION: RoleOverride(
                    timeout_s=7,
                    max_output_tokens=77,
                    reasoning_effort="high",
                    extra_body={"other": 1},
                ),
            },
        )
        model, recorder = _text_model()
        models = build_models(config, overrides={ModelRole.ANSWER: model})
        agent = build_default_agent(FakeMemory(models=models))
        await agent.run("hello")
        settings: dict[str, Any] = dict(recorder.calls[0].info.model_settings or {})
        assert settings["timeout"] == 42
        assert settings["max_tokens"] == 500
        assert settings["openai_reasoning_effort"] == "low"
        assert settings["extra_body"] == {"thinking": {"type": "disabled"}}
