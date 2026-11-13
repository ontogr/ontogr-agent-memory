"""One execution boundary for typed model work."""

from dataclasses import dataclass
from typing import Any, Generic, TypeVar, cast

from pydantic_ai import Agent
from pydantic_ai.exceptions import (
    ModelAPIError,
    ModelHTTPError,
    OutputTokenLimitExceeded,
    UnexpectedModelBehavior,
)
from pydantic_ai.models.openai import OpenAIChatModelSettings
from pydantic_ai.usage import RunUsage

from ontogr_mem.llm.config import RoleConfig
from ontogr_mem.llm.failures import (
    ModelCallFailedError,
    ModelNotConfiguredError,
    ModelOutputError,
)
from ontogr_mem.llm.models import RoleModels
from ontogr_mem.llm.roles import ModelRole
from ontogr_mem.request_context import record_usage


Deps = TypeVar("Deps")
Out = TypeVar("Out")
OUTPUT_RETRIES = 2


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Unpriced input tokens, output tokens and model requests for one run."""

    input_tokens: int
    output_tokens: int
    requests: int


@dataclass(frozen=True, slots=True)
class ModelRun(Generic[Out]):
    """Typed output and usage collected across the run's requests."""

    output: Out
    usage: TokenUsage


def _tokens(usage: RunUsage) -> TokenUsage:
    return TokenUsage(usage.input_tokens, usage.output_tokens, usage.requests)


def role_model_settings(
    config: RoleConfig, max_output_tokens: int | None
) -> OpenAIChatModelSettings:
    """Build the model settings for one kind of work from its role config.

    Args:
        config: Resolved settings for the role, including the timeout, reasoning
            effort and extra request body.
        max_output_tokens: Output token cap to send, or None to omit the cap.

    Returns:
        Settings to pass to an agent run.
    """
    settings = OpenAIChatModelSettings(timeout=config.timeout_s)
    if max_output_tokens is not None:
        settings["max_tokens"] = max_output_tokens
    if config.reasoning_effort is not None:
        # Providers accept effort strings beyond the literals that PydanticAI types, so
        # the configured string is passed through.
        settings["openai_reasoning_effort"] = cast("Any", config.reasoning_effort)
    if config.extra_body is not None:
        settings["extra_body"] = dict(config.extra_body)
    return settings


async def run_agent(
    agent: Agent[Deps, Out],
    role: ModelRole,
    prompt: str,
    *,
    models: RoleModels,
    deps: Deps,
    max_output_tokens: int | None = None,
) -> ModelRun[Out]:
    """Run typed model work and map provider and output failures.

    Args:
        agent: Agent with its output type already declared.
        role: Kind of work whose model and settings apply.
        prompt: User text for this run.
        models: Process-wide model pool.
        deps: Dependencies passed to the agent.
        max_output_tokens: Positive per-call cap, or None for the configured cap.

    Returns:
        Output and usage across output-validation re-asks.

    Raises:
        ModelNotConfiguredError: If the kind of work has no model.
        ValueError: If max_output_tokens is not positive.
        ModelCallFailedError: If a provider call fails.
        ModelOutputError: If output validation fails or output is truncated.
    """
    model = models.model_for(role)
    if model is None:
        raise ModelNotConfiguredError(f"no model configured for {role.value}")
    role_config = models.config_for(role)
    output_cap = (
        max_output_tokens
        if max_output_tokens is not None
        else role_config.max_output_tokens
    )
    if output_cap is not None and output_cap < 1:
        raise ValueError("max_output_tokens must be positive")
    settings = role_model_settings(role_config, output_cap)
    usage = RunUsage()
    try:
        result = await agent.run(
            prompt,
            model=model,
            deps=deps,
            model_settings=settings,
            retries={"output": OUTPUT_RETRIES},
            usage=usage,
        )
    except OutputTokenLimitExceeded as exc:
        raise ModelOutputError(
            str(exc),
            truncated=True,
            usage=_tokens(usage),
        ) from exc
    except UnexpectedModelBehavior as exc:
        raise ModelOutputError(str(exc), usage=_tokens(usage)) from exc
    except ModelHTTPError as exc:
        raise ModelCallFailedError(
            str(exc),
            status_code=exc.status_code,
            usage=_tokens(usage),
        ) from exc
    except ModelAPIError as exc:
        raise ModelCallFailedError(str(exc), usage=_tokens(usage)) from exc
    finally:
        record_usage(usage.input_tokens, usage.output_tokens, None)
    return ModelRun(result.output, _tokens(usage))
