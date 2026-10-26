"""Request and response types and the client protocol."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol


class ModelSize(StrEnum):
    """Model tier; each tier has its own concurrency limit."""

    SMALL = "small"
    MEDIUM = "medium"


@dataclass(frozen=True, slots=True)
class LLMRequest:
    """One completion request.

    Attributes:
        system: System prompt.
        prompt: User prompt.
        model_size: Tier of model to use.
        schema: JSON schema for provider-enforced output, or None for free text.
        max_output_tokens: Cap on generated tokens.
    """

    system: str
    prompt: str
    model_size: ModelSize
    schema: dict[str, Any] | None = None
    max_output_tokens: int = 2048


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """A completion and its token counts."""

    text: str
    input_tokens: int
    output_tokens: int


class LLMClient(Protocol):
    """Provider adapter. Raises `LLMCallError` on provider failures."""

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Run one completion."""
        ...
