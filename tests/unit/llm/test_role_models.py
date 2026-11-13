"""Tests for model construction, provider sharing and the shared limiter."""

import asyncio
import json
from collections.abc import Callable
from typing import Any

import httpx2
import pytest
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai import models as ai_models
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.concurrency import ConcurrencyLimitedModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel

from ontogr_mem.llm import (
    LLMConfig,
    ModelRole,
    RoleModels,
    build_models,
    run_agent,
    split_model_string,
)
from tests.support.scripted_model import scripted_model, scripted_role_models


class Answer(BaseModel):
    """Test data."""

    value: int


def make(**values: Any) -> LLMConfig:
    """Helper: make."""
    return LLMConfig(_env_file=None, **values)


def inner(models: RoleModels, role: ModelRole) -> Any:
    """Helper: inner."""
    wrapped = models.model_for(role)
    assert isinstance(wrapped, ConcurrencyLimitedModel)
    return wrapped.wrapped


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("openai-chat:m", ("openai-chat", "m")),
        ("openai-responses:m", ("openai-responses", "m")),
        ("anthropic:claude", ("anthropic", "claude")),
        ("llama3:8b", ("openai-chat", "llama3:8b")),
        ("plain", ("openai-chat", "plain")),
        ("openai-chat:llama3:8b", ("openai-chat", "llama3:8b")),
    ],
)
def test_split_model_string(value: str, expected: tuple[str, str]) -> None:
    """Split model string."""
    assert split_model_string(value) == expected


def test_bare_openai_prefix_is_rejected() -> None:
    """Bare openai prefix is rejected."""
    with pytest.raises(ValueError, match="openai-responses"):
        split_model_string("openai:gpt")


async def test_unconfigured_pool_builds_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unconfigured pool builds nothing."""
    monkeypatch.setenv("MEMORY_LLM_MODEL", "m")
    monkeypatch.setenv("MEMORY_LLM_BASE_URL", "http://x")
    monkeypatch.setenv("OPENAI_API_KEY", "ambient")

    def boom() -> None:
        raise AssertionError("client built")

    monkeypatch.setattr("ontogr_mem.llm.models.create_async_httpx2_client", boom)
    models = build_models(LLMConfig.model_construct())
    assert not models.any_configured()
    assert all(not models.configured(role) for role in ModelRole)
    assert models.model_for(ModelRole.JUDGE) is None
    assert models.config_for(ModelRole.FACT_EXTRACTION).max_output_tokens == 16384
    await models.aclose()


async def test_three_protocols_build_with_explicit_url_and_key() -> None:
    """Three protocols build with explicit url and key."""
    cfg = make(
        model="llama3:8b",
        base_url="http://chat.test/v1",
        api_key="chat-key",
        roles={
            "judge": {
                "model": "openai-responses:r",
                "base_url": "http://resp.test/v1",
                "api_key": "resp-key",
            },
            "answer": {
                "model": "anthropic:c",
                "base_url": "http://anth.test",
                "api_key": "anth-key",
            },
        },
    )
    models = build_models(cfg)
    chat = inner(models, ModelRole.ENTITY_EXTRACTION)
    resp = inner(models, ModelRole.JUDGE)
    anth = inner(models, ModelRole.ANSWER)
    assert isinstance(chat, OpenAIChatModel)
    assert isinstance(resp, OpenAIResponsesModel)
    assert isinstance(anth, AnthropicModel)
    assert chat.model_name == "llama3:8b"
    assert (str(chat.client.base_url), chat.client.api_key) == (
        "http://chat.test/v1/",
        "chat-key",
    )
    assert (str(resp.client.base_url), resp.client.api_key) == (
        "http://resp.test/v1/",
        "resp-key",
    )
    assert (str(anth.client.base_url).rstrip("/"), anth.client.api_key) == (
        "http://anth.test",
        "anth-key",
    )
    await models.aclose()


async def test_roles_with_same_protocol_url_and_key_share_one_provider() -> None:
    """Roles with same protocol url and key share one provider."""
    cfg = make(
        model="m",
        base_url="http://x/v1",
        api_key="k",
        roles={
            "judge": {"output_mode": "native"},
            "ontology": {"api_key": "other"},
            "resolution": {"base_url": "http://y/v1"},
            "answer": {"model": "openai-responses:m"},
        },
    )
    models = build_models(cfg)
    base = inner(models, ModelRole.ENTITY_EXTRACTION).client
    assert inner(models, ModelRole.FACT_EXTRACTION).client is base
    assert inner(models, ModelRole.JUDGE).client is base
    assert inner(models, ModelRole.ONTOLOGY).client is not base
    assert inner(models, ModelRole.RESOLUTION).client is not base
    assert inner(models, ModelRole.ANSWER).client is not base
    assert len(models._clients) == 4
    await models.aclose()


async def test_two_roles_on_one_endpoint_keep_their_own_output_mode() -> None:
    """Two roles on one endpoint keep their own output mode."""
    cfg = make(
        model="m",
        base_url="http://x/v1",
        roles={"judge": {"output_mode": "prompted"}},
    )
    models = build_models(cfg)
    judge = inner(models, ModelRole.JUDGE)
    extraction = inner(models, ModelRole.ENTITY_EXTRACTION)
    assert judge.client is extraction.client
    assert judge.profile["default_structured_output_mode"] == "prompted"
    assert extraction.profile["default_structured_output_mode"] == "tool"
    await models.aclose()


@pytest.mark.parametrize("blank", ["sentinel", "blank"])
@pytest.mark.parametrize("keyed", [True, False])
async def test_ambient_provider_variables_are_never_read(
    monkeypatch: pytest.MonkeyPatch, blank: str, keyed: bool
) -> None:
    """Ambient provider variables are never read."""
    value = "" if blank == "blank" else "ambient-sentinel"
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.setenv(name, value)
    if blank == "sentinel":
        monkeypatch.setenv("OPENAI_BASE_URL", "http://ambient-sentinel.test/v1")
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://ambient-sentinel.test")
    else:
        monkeypatch.setenv("OPENAI_BASE_URL", "")
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "")
    key: dict[str, Any] = {"api_key": "explicit-key"} if keyed else {}
    cfg = make(
        model="m",
        base_url="http://explicit.test/v1",
        roles={"answer": {"model": "anthropic:c", "base_url": "http://explicit.test"}},
        **key,
    )
    models = build_models(cfg)
    for role in (ModelRole.JUDGE, ModelRole.ANSWER):
        client = inner(models, role).client
        assert "ambient-sentinel" not in str(client.base_url)
        assert "explicit.test" in str(client.base_url)
        assert "ambient-sentinel" not in str(client.api_key)
        if keyed:
            assert client.api_key == "explicit-key"
    await models.aclose()


def test_missing_base_url_names_the_variable() -> None:
    """Missing base url names the variable."""
    cfg = LLMConfig.model_construct(model="m")
    with pytest.raises(ValueError, match="MEMORY_LLM_ENTITY_EXTRACTION_BASE_URL"):
        build_models(cfg)


def test_no_client_is_created_when_a_later_role_lacks_a_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No HTTP client is created before every role's configuration is validated."""
    created: list[httpx2.AsyncClient] = []

    def counting_client() -> httpx2.AsyncClient:
        client = ai_models.create_async_httpx2_client()
        created.append(client)
        return client

    monkeypatch.setattr(
        "ontogr_mem.llm.models.create_async_httpx2_client", counting_client
    )
    config = make(
        roles={"entity_extraction": {"model": "m", "base_url": "http://x/v1"}}
    )
    with pytest.raises(ValueError, match="MEMORY_LLM_ANSWER_BASE_URL"):
        build_models(config, overrides={ModelRole.ANSWER: "m"})
    assert created == []


async def test_model_and_string_overrides() -> None:
    """Model and string overrides."""
    fn, _ = scripted_model([{"value": 1}])
    cfg = make(model="m", base_url="http://x/v1")
    models = build_models(
        cfg,
        overrides={
            ModelRole.JUDGE: fn,
            ModelRole.ANSWER: "other",
            ModelRole.ONTOLOGY: " ",
        },
    )
    assert inner(models, ModelRole.JUDGE) is fn
    assert inner(models, ModelRole.ANSWER).model_name == "other"
    assert not models.configured(ModelRole.ONTOLOGY)
    await models.aclose()


async def test_injected_models_are_not_closed_and_pool_owns_none() -> None:
    """Injected models are not closed and pool owns none."""
    fn, _ = scripted_model()
    models = scripted_role_models(fn)
    assert models._clients == ()
    await models.aclose()
    assert all(models.configured(role) for role in ModelRole)


async def test_aclose_closes_owned_clients() -> None:
    """Aclose closes owned clients."""
    models = build_models(make(model="m", base_url="http://x/v1"))
    client = inner(models, ModelRole.JUDGE).client
    http = client._client
    assert not http.is_closed
    await models.aclose()
    assert http.is_closed


async def test_all_roles_share_one_limiter() -> None:
    """All roles share one limiter."""
    models = build_models(
        make(
            model="m",
            base_url="http://x/v1",
            max_concurrency=2,
            roles={"judge": {"output_mode": "prompted"}},
        )
    )
    limiters = {models.model_for(role)._limiter for role in ModelRole}  # type: ignore[union-attr]
    assert limiters == {models.limiter}
    await models.aclose()


async def test_limiter_peak_and_recovery_after_failure() -> None:
    """Limiter peak and recovery after failure."""
    active = peak = 0
    gate = asyncio.Event()

    async def slow(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        try:
            await gate.wait()
            if (info.model_settings or {}).get("max_tokens") == 7:
                raise RuntimeError("boom")
            return ModelResponse(
                parts=[ToolCallPart(info.output_tools[0].name, '{"value": 1}')]
            )
        finally:
            active -= 1

    models = build_models(
        make(max_concurrency=2),
        overrides={role: FunctionModel(slow) for role in ModelRole},
    )
    agent = Agent(output_type=Answer)
    runs = [
        asyncio.create_task(
            run_agent(
                agent,
                ModelRole.JUDGE,
                "p",
                models=models,
                deps=None,
                max_output_tokens=7 if i == 0 else 8,
            )
        )
        for i in range(6)
    ]
    for _ in range(100):
        await asyncio.sleep(0.01)
        if active == 2:
            break
    assert active == 2
    gate.set()
    results = await asyncio.gather(*runs, return_exceptions=True)
    assert peak == 2
    assert sum(isinstance(r, Exception) for r in results) == 1
    again = await run_agent(agent, ModelRole.JUDGE, "p", models=models, deps=None)
    assert again.output.value == 1
    await models.aclose()


async def test_cancelled_run_releases_its_limiter_slot() -> None:
    """Cancelled run releases its limiter slot."""
    active = 0
    gate = asyncio.Event()

    async def hold(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal active
        active += 1
        try:
            await gate.wait()
            return ModelResponse(
                parts=[ToolCallPart(info.output_tools[0].name, '{"value": 1}')]
            )
        finally:
            active -= 1

    models = build_models(
        make(max_concurrency=1),
        overrides={role: FunctionModel(hold) for role in ModelRole},
    )
    agent = Agent(output_type=Answer)

    async def call() -> Any:
        return await run_agent(agent, ModelRole.JUDGE, "p", models=models, deps=None)

    first = asyncio.create_task(call())
    for _ in range(100):
        await asyncio.sleep(0.01)
        if active == 1:
            break
    assert active == 1
    second = asyncio.create_task(call())
    await asyncio.sleep(0.05)
    assert active == 1
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    for _ in range(100):
        await asyncio.sleep(0.01)
        if active == 1:
            break
    assert active == 1
    gate.set()
    assert (await second).output.value == 1
    assert (await call()).output.value == 1
    await models.aclose()


@pytest.mark.parametrize("mode", ["tool", "native", "prompted"])
async def test_output_mode_survives_the_limiter_wrapper(mode: str) -> None:
    """Output mode survives the limiter wrapper."""
    fn, rec = scripted_model([{"value": 3}], mode=mode)
    models = scripted_role_models(fn)
    run = await run_agent(
        Agent(output_type=Answer, instructions="sys"),
        ModelRole.JUDGE,
        "p",
        models=models,
        deps=None,
    )
    assert run.output == Answer(value=3)
    params = rec.calls[0].info.model_request_parameters
    assert params.output_mode == mode
    assert bool(rec.calls[0].info.output_tools) == (mode == "tool")


def _chat_reply(body: dict[str, Any], mode: str) -> dict[str, Any]:
    """Helper: chat reply."""
    message: dict[str, Any] = {"role": "assistant", "content": None}
    if mode == "tool":
        name = body["tools"][0]["function"]["name"]
        message["tool_calls"] = [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": name, "arguments": '{"value": 4}'},
            }
        ]
    else:
        message["content"] = '{"value": 4}'
    return {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }


@pytest.mark.parametrize("mode", ["tool", "native", "prompted"])
async def test_request_shape_on_the_wire(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """Request shape on the wire."""
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(200, json=_chat_reply(json.loads(request.content), mode))

    real: Callable[..., httpx2.AsyncClient] = httpx2.AsyncClient

    def fake_client() -> httpx2.AsyncClient:
        return real(transport=httpx2.MockTransport(handler))

    monkeypatch.setattr("ontogr_mem.llm.models.create_async_httpx2_client", fake_client)
    models = build_models(
        make(
            model="m",
            base_url="http://wire.test/v1",
            api_key="k",
            output_mode=mode,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            reasoning_effort="low",
            timeout_s=12,
        )
    )
    with ai_models.override_allow_model_requests(True):
        run = await run_agent(
            Agent(output_type=Answer, instructions="sys"),
            ModelRole.FACT_EXTRACTION,
            "p",
            models=models,
            deps=None,
        )
    assert run.output == Answer(value=4)
    assert (run.usage.input_tokens, run.usage.output_tokens, run.usage.requests) == (
        3,
        2,
        1,
    )
    body = json.loads(seen[0].content)
    assert seen[0].url.path == "/v1/chat/completions"
    assert seen[0].headers["authorization"] == "Bearer k"
    assert body["max_completion_tokens"] == 16384
    assert body["reasoning_effort"] == "low"
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    if mode == "tool":
        assert body["tools"] and "response_format" not in body
    elif mode == "native":
        assert body["response_format"]["type"] == "json_schema"
        assert "tools" not in body
    else:
        assert "tools" not in body
        assert body["response_format"] == {"type": "json_object"}
        assert "value" in json.dumps(body["messages"])
    await models.aclose()
