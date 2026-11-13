"""Tests for the model execution boundary."""

from typing import Any

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from pydantic import BaseModel
from pydantic_ai import Agent, InstrumentationSettings
from pydantic_ai.exceptions import (
    ContentFilterError,
    ModelAPIError,
    ModelHTTPError,
    ModelRequestTimeout,
    UsageLimitExceeded,
)
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage

from ontogr_mem.llm import (
    OUTPUT_RETRIES,
    LLMConfig,
    ModelCallFailedError,
    ModelNotConfiguredError,
    ModelOutputError,
    ModelRole,
    TokenUsage,
    build_models,
    run_agent,
)
from ontogr_mem.request_context import RequestContext, bind_context
from tests.support.scripted_model import scripted_model, scripted_role_models


class Answer(BaseModel):
    """Test data."""

    value: int


def agent() -> Agent[None, Answer]:
    """Helper: agent."""
    return Agent[None, Answer](output_type=Answer, instructions="sys")


@pytest.mark.parametrize("mode", ["tool", "native", "prompted"])
async def test_typed_success_in_every_mode(mode: str) -> None:
    """Typed success in every mode."""
    fn, rec = scripted_model([{"value": 5}], mode=mode)
    run = await run_agent(
        agent(), ModelRole.JUDGE, "hello", models=scripted_role_models(fn), deps=None
    )
    assert run.output == Answer(value=5)
    assert run.usage.requests == 1
    assert rec.calls[0].prompt == "hello"
    assert rec.calls[0].system.startswith("sys")


async def test_invalid_then_valid_output_is_reasked() -> None:
    """Invalid then valid output is reasked."""
    fn, rec = scripted_model([{"value": "x"}, {"value": 2}])
    run = await run_agent(
        agent(), ModelRole.JUDGE, "p", models=scripted_role_models(fn), deps=None
    )
    assert run.output.value == 2
    assert run.usage.requests == 2
    assert len(rec.calls) == 2


async def test_exhausted_validation_makes_exactly_three_requests() -> None:
    """Exhausted validation makes exactly three requests."""
    assert OUTPUT_RETRIES == 2
    fn, rec = scripted_model(answer=lambda call: {"value": "bad"})
    with pytest.raises(ModelOutputError) as info:
        await run_agent(
            agent(), ModelRole.JUDGE, "p", models=scripted_role_models(fn), deps=None
        )
    assert len(rec.calls) == 3
    assert info.value.truncated is False
    assert info.value.usage is not None and info.value.usage.requests == 3


async def test_truncated_reply_is_flagged() -> None:
    """Truncated reply is flagged."""

    def cut(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[TextPart('{"value": ')],
            finish_reason="length",
        )

    fn = FunctionModel(
        cut,
        profile={"default_structured_output_mode": "prompted"},
    )
    with pytest.raises(ModelOutputError) as info:
        await run_agent(
            agent(), ModelRole.JUDGE, "p", models=scripted_role_models(fn), deps=None
        )
    assert info.value.truncated is True


async def test_content_filter_is_an_output_error() -> None:
    """Content filter is an output error."""

    def refuse(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[], finish_reason="content_filter")

    with pytest.raises(ModelOutputError) as info:
        await run_agent(
            agent(),
            ModelRole.JUDGE,
            "p",
            models=scripted_role_models(FunctionModel(refuse)),
            deps=None,
        )
    assert isinstance(info.value.__cause__, ContentFilterError)


@pytest.mark.parametrize("status", [400, 401, 403, 408, 413, 429, 500, 503])
async def test_http_errors_keep_the_status(status: int) -> None:
    """Http errors keep the status."""
    fn, _ = scripted_model([ModelHTTPError(status, "m", {"error": "x"})])
    with pytest.raises(ModelCallFailedError) as info:
        await run_agent(
            agent(), ModelRole.JUDGE, "p", models=scripted_role_models(fn), deps=None
        )
    assert info.value.status_code == status
    assert info.value.usage == TokenUsage(0, 0, 0)


@pytest.mark.parametrize(
    "error", [ModelAPIError("m", "down"), ModelRequestTimeout("m", 1.0)]
)
async def test_connection_failures_have_no_status(error: ModelAPIError) -> None:
    """Connection failures have no status."""
    fn, _ = scripted_model([error])
    with pytest.raises(ModelCallFailedError) as info:
        await run_agent(
            agent(), ModelRole.JUDGE, "p", models=scripted_role_models(fn), deps=None
        )
    assert info.value.status_code is None


async def test_missing_model() -> None:
    """Missing model."""
    models = build_models(LLMConfig.model_construct())
    with pytest.raises(ModelNotConfiguredError, match="judge"):
        await run_agent(agent(), ModelRole.JUDGE, "p", models=models, deps=None)


@pytest.mark.parametrize("error", [RuntimeError("bug"), KeyError("k")])
async def test_unexpected_errors_propagate(error: Exception) -> None:
    """Unexpected errors propagate."""
    fn, _ = scripted_model([error])
    with pytest.raises(type(error)):
        await run_agent(
            agent(), ModelRole.JUDGE, "p", models=scripted_role_models(fn), deps=None
        )


async def test_unmapped_library_errors_propagate() -> None:
    """Unmapped library errors propagate."""
    fn, _ = scripted_model([UsageLimitExceeded("limit")])
    with pytest.raises(UsageLimitExceeded):
        await run_agent(
            agent(), ModelRole.JUDGE, "p", models=scripted_role_models(fn), deps=None
        )


def _settings_probe() -> tuple[FunctionModel, list[dict[str, Any]]]:
    """Helper: settings probe."""
    seen: list[dict[str, Any]] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(dict(info.model_settings or {}))
        return ModelResponse(
            parts=[ToolCallPart(info.output_tools[0].name, '{"value": 1}')]
        )

    return FunctionModel(respond), seen


async def test_settings_reach_the_model_and_per_call_cap_wins() -> None:
    """Settings reach the model and per call cap wins."""
    fn, seen = _settings_probe()
    cfg = LLMConfig(
        _env_file=None,
        max_output_tokens=500,
        timeout_s=42,
        reasoning_effort="low",
        extra_body={"thinking": {"type": "disabled"}},
    )
    models = build_models(cfg, overrides={ModelRole.JUDGE: fn, ModelRole.ANSWER: fn})
    await run_agent(agent(), ModelRole.JUDGE, "p", models=models, deps=None)
    await run_agent(
        agent(), ModelRole.JUDGE, "p", models=models, deps=None, max_output_tokens=77
    )
    assert seen[0]["max_tokens"] == 500
    assert seen[0]["timeout"] == 42
    assert seen[0]["openai_reasoning_effort"] == "low"
    assert seen[0]["extra_body"] == {"thinking": {"type": "disabled"}}
    assert seen[1]["max_tokens"] == 77


async def test_role_defaults_and_uncapped_answer() -> None:
    """Role defaults and uncapped answer."""
    fn, seen = _settings_probe()
    models = build_models(
        LLMConfig(_env_file=None),
        overrides={ModelRole.FACT_EXTRACTION: fn, ModelRole.ANSWER: fn},
    )
    await run_agent(agent(), ModelRole.FACT_EXTRACTION, "p", models=models, deps=None)
    await run_agent(agent(), ModelRole.ANSWER, "p", models=models, deps=None)
    assert seen[0]["max_tokens"] == 16384
    assert seen[0]["timeout"] == 300.0
    assert "max_tokens" not in seen[1]
    assert "extra_body" not in seen[1]
    assert "openai_reasoning_effort" not in seen[1]


async def test_nonpositive_cap_is_rejected() -> None:
    """Nonpositive cap is rejected."""
    fn, _ = scripted_model([{"value": 1}])
    with pytest.raises(ValueError, match="positive"):
        await run_agent(
            agent(),
            ModelRole.JUDGE,
            "p",
            models=scripted_role_models(fn),
            deps=None,
            max_output_tokens=0,
        )


def _usage_model(replies: list[Any]) -> FunctionModel:
    """Helper: usage model."""
    queue = list(replies)

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return ModelResponse(
            parts=[ToolCallPart(info.output_tools[0].name, item)],
            usage=RequestUsage(input_tokens=10, output_tokens=4),
        )

    return FunctionModel(respond)


async def test_exact_usage_on_success_failure_and_context_accounting() -> None:
    """Exact usage on success failure and context accounting."""
    ok = _usage_model(['{"value": "x"}', '{"value": 1}'])
    context = RequestContext(principal_id="owner", tenant="acme")
    with bind_context(context):
        run = await run_agent(
            agent(), ModelRole.JUDGE, "p", models=scripted_role_models(ok), deps=None
        )
        assert run.usage == TokenUsage(20, 8, 2)
        assert (context.usage.tokens_in, context.usage.tokens_out) == (20, 8)
        assert context.usage.unpriced_calls == 1
        assert context.usage.priced_cost_usd == 0.0

        bad = _usage_model(['{"value": "x"}', ModelHTTPError(503, "m", None)])
        with pytest.raises(ModelCallFailedError) as info:
            await run_agent(
                agent(),
                ModelRole.JUDGE,
                "p",
                models=scripted_role_models(bad),
                deps=None,
            )
        assert info.value.usage == TokenUsage(10, 4, 1)
        assert (context.usage.tokens_in, context.usage.tokens_out) == (30, 12)
        assert context.usage.unpriced_calls == 2

        exhausted = _usage_model(['{"value": "x"}'] * 3)
        with pytest.raises(ModelOutputError) as out:
            await run_agent(
                agent(),
                ModelRole.JUDGE,
                "p",
                models=scripted_role_models(exhausted),
                deps=None,
            )
        assert out.value.usage == TokenUsage(30, 12, 3)
        assert context.usage.unpriced_calls == 3


async def test_cost_is_never_called(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cost is never called."""

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("cost() must not be called")

    monkeypatch.setattr(ModelResponse, "cost", forbidden)
    run = await run_agent(
        agent(),
        ModelRole.JUDGE,
        "p",
        models=scripted_role_models(_usage_model(['{"value": 1}'])),
        deps=None,
    )
    assert run.output.value == 1


async def test_opt_in_tracing_is_content_free() -> None:
    """Opt in tracing is content free."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    fn, _ = scripted_model([{"value": "x"}, {"value": 1}])
    Agent.instrument_all(
        InstrumentationSettings(
            tracer_provider=provider,
            include_content=False,
            include_binary_content=False,
            include_model_request_parameters=False,
        )
    )
    try:
        await run_agent(
            Agent[None, Answer](output_type=Answer, instructions="INSTR-SENTINEL"),
            ModelRole.JUDGE,
            "PROMPT-SENTINEL",
            models=scripted_role_models(fn),
            deps=None,
        )
    finally:
        Agent.instrument_all(False)
    spans = exporter.get_finished_spans()
    assert spans
    dump = repr(
        [(dict(s.attributes or {}), [e.attributes for e in s.events]) for s in spans]
    )
    assert "INSTR-SENTINEL" not in dump
    assert "PROMPT-SENTINEL" not in dump
    assert '"value": 1' not in dump
