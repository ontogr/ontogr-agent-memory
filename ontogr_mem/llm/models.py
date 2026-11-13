"""Construct models with explicit providers and a shared concurrency limit."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from pydantic_ai import ConcurrencyLimiter
from pydantic_ai.models import Model, create_async_httpx2_client
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.concurrency import ConcurrencyLimitedModel
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai_compatible import OpenAICompatibleProvider

from ontogr_mem.llm.config import LLMConfig, RoleConfig
from ontogr_mem.llm.roles import ModelRole


Protocol = Literal["openai-chat", "openai-responses", "anthropic"]


def split_model_string(value: str) -> tuple[Protocol, str]:
    """Split only supported protocol prefixes; keep other colons in the name.

    Raises:
        ValueError: If the ambiguous openai prefix is used.
    """
    if value.startswith("openai:"):
        raise ValueError("use openai-chat: or openai-responses:, not openai:")
    for protocol in ("openai-chat", "openai-responses", "anthropic"):
        prefix = protocol + ":"
        if value.startswith(prefix):
            return protocol, value[len(prefix) :]
    return "openai-chat", value


@dataclass(frozen=True, slots=True)
class RoleModels:
    """Process-wide models and the HTTP clients owned by their builder.

    Attributes:
        limiter: Shared slots used by every configured model.
    """

    limiter: ConcurrencyLimiter
    _models: Mapping[ModelRole, Model] = field(repr=False)
    _configs: Mapping[ModelRole, RoleConfig] = field(repr=False)
    _clients: tuple[Any, ...] = field(default=(), repr=False)

    def configured(self, role: ModelRole) -> bool:
        """Return whether the kind of work has a model."""
        return role in self._models

    def any_configured(self) -> bool:
        """Return whether any kind of work has a model."""
        return bool(self._models)

    def model_for(self, role: ModelRole) -> Model | None:
        """Return the limited model, or None when not configured."""
        return self._models.get(role)

    def config_for(self, role: ModelRole) -> RoleConfig:
        """Return resolved per-run settings even for an unconfigured model."""
        return self._configs[role]

    async def aclose(self) -> None:
        """Close only builder-owned clients, never an injected model's client."""
        for client in self._clients:
            await client.aclose()


def build_models(  # noqa: PLR0912
    config: LLMConfig | None = None,
    *,
    overrides: Mapping[ModelRole, Model | str] | None = None,
) -> RoleModels:
    """Build one limited model per configured kind of work.

    Args:
        config: Settings, or None to read the environment and .env.
        overrides: Model objects or names that replace configured names.

    Returns:
        Models whose owned clients the caller closes after all runs finish.

    Raises:
        ValueError: If a model name has no endpoint or uses openai:.
    """
    selected = config if config is not None else LLMConfig()
    limiter = ConcurrencyLimiter(selected.max_concurrency, name="memory-llm")
    configs: dict[ModelRole, RoleConfig] = {}
    injected: dict[ModelRole, Model] = {}
    planned: list[tuple[ModelRole, Protocol, str, str, RoleConfig]] = []
    for role in ModelRole:
        role_config = selected.resolve(role)
        override = (overrides or {}).get(role)
        if isinstance(override, str):
            role_config = replace(
                role_config, model=override if override.strip() else None
            )
        configs[role] = role_config
        if isinstance(override, Model):
            injected[role] = override
            continue
        if not role_config.model:
            continue
        protocol, name = split_model_string(role_config.model)
        if not role_config.base_url:
            raise ValueError(
                f"set MEMORY_LLM_{role.name}_BASE_URL or MEMORY_LLM_BASE_URL"
            )
        planned.append((role, protocol, name, role_config.base_url, role_config))
    built: dict[ModelRole, Model] = {
        role: ConcurrencyLimitedModel(model, limiter=limiter)
        for role, model in injected.items()
    }
    providers: dict[
        tuple[Protocol, str, str | None],
        OpenAICompatibleProvider | AnthropicProvider,
    ] = {}
    clients = []
    for role, protocol, name, base_url, role_config in planned:
        key = (
            role_config.api_key.get_secret_value()
            if role_config.api_key is not None
            else None
        )
        provider_key = protocol, base_url, key
        provider = providers.get(provider_key)
        if provider is None:
            client = create_async_httpx2_client()
            clients.append(client)
            if protocol == "anthropic":
                # Passing None lets the SDK read ambient credentials.
                provider = AnthropicProvider(
                    base_url=base_url,
                    api_key=key or "not-set",
                    http_client=client,
                )
            else:
                provider = OpenAICompatibleProvider(
                    "custom",
                    base_url=base_url,
                    api_key=key,
                    http_client=client,
                )
            providers[provider_key] = provider
        profile = ModelProfile(default_structured_output_mode=role_config.output_mode)
        if role_config.output_mode == "native":
            profile["supports_json_schema_output"] = True
        if protocol == "anthropic":
            if not isinstance(provider, AnthropicProvider):
                raise TypeError("expected an Anthropic provider")
            model = AnthropicModel(name, provider=provider, profile=profile)
        else:
            if not isinstance(provider, OpenAICompatibleProvider):
                raise TypeError("expected an OpenAI-compatible provider")
            factory = (
                OpenAIResponsesModel
                if protocol == "openai-responses"
                else OpenAIChatModel
            )
            model = factory(name, provider=provider, profile=profile)
        built[role] = ConcurrencyLimitedModel(model, limiter=limiter)
    return RoleModels(limiter, built, configs, tuple(clients))
