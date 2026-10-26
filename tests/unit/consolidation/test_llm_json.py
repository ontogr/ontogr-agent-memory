"""Validated JSON answers from the language model."""

import pytest

from ontogr_mem.consolidation.config import LLMSettings
from ontogr_mem.consolidation.llm_json import LLMOutputError, SummaryAnswer, ask
from ontogr_mem.data_models.extraction import Usage
from ontogr_mem.llm import LLMGate, ModelSize
from tests.support.llm import ScriptedLLM


def _gate(llm: ScriptedLLM) -> LLMGate:
    """Build a gate around a scripted client."""
    return LLMGate(llm, max_input_tokens=10_000)


async def test_a_fenced_json_answer_is_parsed_into_the_model() -> None:
    """A fenced JSON answer is parsed into the model."""
    llm = ScriptedLLM(replies=['```json\n{"summary": "Acme makes tools."}\n```'])
    usage = Usage()

    answer = await ask(
        _gate(llm),
        SummaryAnswer,
        "system",
        "prompt",
        settings=LLMSettings(),
        usage=usage,
        stage="communities",
    )

    assert answer == SummaryAnswer(summary="Acme makes tools.")


async def test_a_malformed_answer_raises_the_output_error() -> None:
    """A malformed answer raises the output error."""
    llm = ScriptedLLM(replies=["not json at all"])

    with pytest.raises(LLMOutputError):
        await ask(
            _gate(llm),
            SummaryAnswer,
            "system",
            "prompt",
            settings=LLMSettings(),
            usage=Usage(),
            stage="communities",
        )


async def test_an_answer_off_the_schema_raises_the_output_error() -> None:
    """An answer that does not match the schema raises the output error."""
    llm = ScriptedLLM(replies=['{"summary": ""}'])

    with pytest.raises(LLMOutputError):
        await ask(
            _gate(llm),
            SummaryAnswer,
            "system",
            "prompt",
            settings=LLMSettings(),
            usage=Usage(),
            stage="communities",
        )


async def test_the_tokens_of_the_call_are_recorded_in_the_usage() -> None:
    """The usage holds the tokens and the call count of the answer."""
    llm = ScriptedLLM(replies=['{"summary": "Acme makes tools."}'])
    usage = Usage()

    await ask(
        _gate(llm),
        SummaryAnswer,
        "system",
        "prompt",
        settings=LLMSettings(model_size=ModelSize.MEDIUM),
        usage=usage,
        stage="reflection",
    )

    count = usage.buckets["reflection/medium"]
    assert count.calls == 1
    assert count.output_tokens == 1


async def test_the_model_size_and_the_output_cap_come_from_the_settings() -> None:
    """The call carries the model size and the output cap of the settings."""
    llm = ScriptedLLM(replies=['{"summary": "Acme makes tools."}'])

    await ask(
        _gate(llm),
        SummaryAnswer,
        "system",
        "prompt",
        settings=LLMSettings(model_size=ModelSize.MEDIUM, max_output_tokens=4096),
        usage=Usage(),
        stage="communities",
    )

    request = llm.requests[0]
    assert request.model_size is ModelSize.MEDIUM
    assert request.max_output_tokens == 4096
