"""Shared language model interface and call gate."""

from ontogr_mem.llm.base import LLMClient, LLMRequest, LLMResponse, ModelSize
from ontogr_mem.llm.errors import (
    CircuitOpen,
    DeadlineExceeded,
    InputTooLargeError,
    LLMCallError,
    LLMError,
    StructuredOutputError,
)
from ontogr_mem.llm.gate import BatchItem, LLMGate, estimate_tokens
from ontogr_mem.llm.json_reply import SchemaError, load_json
from ontogr_mem.llm.prompt import TAGS, fence
from ontogr_mem.llm.settings import LLMSettings


__all__ = [
    "BatchItem",
    "CircuitOpen",
    "DeadlineExceeded",
    "InputTooLargeError",
    "LLMCallError",
    "LLMClient",
    "LLMError",
    "LLMGate",
    "LLMSettings",
    "LLMRequest",
    "LLMResponse",
    "ModelSize",
    "SchemaError",
    "StructuredOutputError",
    "TAGS",
    "estimate_tokens",
    "fence",
    "load_json",
]
