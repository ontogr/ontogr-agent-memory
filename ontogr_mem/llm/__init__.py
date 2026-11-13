"""Shared language model interface and call gate."""

from ontogr_mem.llm.base import LLMClient, LLMRequest, LLMResponse, ModelSize
from ontogr_mem.llm.config import LLMConfig, RoleConfig, RoleOverride
from ontogr_mem.llm.errors import (
    CircuitOpenError,
    DeadlineExceededError,
    InputTooLargeError,
    LLMCallError,
    LLMError,
    StructuredOutputError,
)
from ontogr_mem.llm.failures import (
    ModelCallFailedError,
    ModelFailure,
    ModelNotConfiguredError,
    ModelOutputError,
)
from ontogr_mem.llm.gate import BatchItem, LLMGate, estimate_tokens
from ontogr_mem.llm.json_reply import SchemaError, load_json
from ontogr_mem.llm.models import RoleModels, build_models, split_model_string
from ontogr_mem.llm.prompt import TAGS, fence
from ontogr_mem.llm.roles import ModelRole
from ontogr_mem.llm.run import OUTPUT_RETRIES, ModelRun, TokenUsage, run_agent
from ontogr_mem.llm.settings import LLMSettings


__all__ = [
    "BatchItem",
    "CircuitOpenError",
    "DeadlineExceededError",
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
    "LLMConfig",
    "ModelCallFailedError",
    "ModelFailure",
    "ModelNotConfiguredError",
    "ModelOutputError",
    "ModelRole",
    "ModelRun",
    "OUTPUT_RETRIES",
    "RoleConfig",
    "RoleModels",
    "RoleOverride",
    "TokenUsage",
    "build_models",
    "run_agent",
    "split_model_string",
]
