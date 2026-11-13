"""Configuration and typed model calls shared across components."""

from ontogr_mem.llm.config import LLMConfig, RoleConfig, RoleOverride
from ontogr_mem.llm.failures import (
    ModelCallFailedError,
    ModelFailure,
    ModelNotConfiguredError,
    ModelOutputError,
)
from ontogr_mem.llm.models import RoleModels, build_models, split_model_string
from ontogr_mem.llm.prompt import TAGS, fence
from ontogr_mem.llm.roles import ModelRole
from ontogr_mem.llm.run import OUTPUT_RETRIES, ModelRun, TokenUsage, run_agent


__all__ = [
    "TAGS",
    "fence",
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
