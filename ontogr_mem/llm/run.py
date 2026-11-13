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
        ModelCallFailedError: If a provider call fails.
        ModelOutputError: If output validation fails or output is truncated.
    """
    model = models.model_for(role)
    if model is None:
        raise ModelNotConfiguredError(f"no model configured for {role.value}")
    cfg = models.config_for(role)
    cap = max_output_tokens if max_output_tokens is not None else cfg.max_output_tokens
    if cap is not None and cap < 1:
        raise ValueError("max_output_tokens must be positive")
    settings = OpenAIChatModelSettings(timeout=cfg.timeout_s)
    if cap is not None:
        settings["max_tokens"] = cap
    if cfg.reasoning_effort is not None:
        settings["openai_reasoning_effort"] = cast("Any", cfg.reasoning_effort)
    if cfg.extra_body is not None:
        settings["extra_body"] = dict(cfg.extra_body)
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
