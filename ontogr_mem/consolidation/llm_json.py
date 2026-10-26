"""Ask for JSON and validate it against a pydantic model.

Small models sometimes return malformed or off-schema JSON. A bad answer raises
``LLMOutputError``, and the caller decides whether to keep the old value.
"""

from typing import TypeVar

from pydantic import BaseModel, Field

from ontogr_mem.consolidation.config import LLMSettings
from ontogr_mem.data_models.extraction import Usage
from ontogr_mem.llm.base import LLMRequest
from ontogr_mem.llm.errors import LLMError
from ontogr_mem.llm.gate import LLMGate
from ontogr_mem.llm.json_reply import load_json


class LLMOutputError(LLMError):
    """The model answered, but not with JSON of the requested shape."""


class SummaryAnswer(BaseModel):
    """A rewritten or merged summary."""

    summary: str = Field(min_length=1)


M = TypeVar("M", bound=BaseModel)


async def ask(
    gate: LLMGate,
    model: type[M],
    system: str,
    prompt: str,
    *,
    settings: LLMSettings,
    usage: Usage,
    stage: str,
) -> M:
    """Make one call and return a validated ``model``.

    The call goes through the gate, so the input cap, the timeout and the
    concurrency limit of the model size apply. What the call cost is recorded
    in ``usage``.

    Args:
        gate: The gate every model call passes through.
        model: The shape the answer must have.
        system: The system prompt.
        prompt: The user prompt.
        settings: The model tier and the output cap of the call.
        usage: Where the tokens of the call are recorded.
        stage: The name of the stage that made the call.

    Returns:
        The validated answer.

    Raises:
        LLMError: If the call fails.
        LLMOutputError: If the answer is not JSON of the requested shape.
    """
    response = await gate.complete(
        LLMRequest(
            system=system,
            prompt=prompt,
            model_size=settings.model_size,
            schema=model.model_json_schema(),
            max_output_tokens=settings.max_output_tokens,
        )
    )
    usage.add(
        stage,
        settings.model_size.value,
        response.input_tokens,
        response.output_tokens,
    )
    try:
        return model.model_validate(load_json(response.text))
    except ValueError as exc:
        raise LLMOutputError(f"bad JSON for {model.__name__}: {exc}") from exc
