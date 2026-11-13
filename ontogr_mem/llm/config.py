"""Global and per-kind model settings."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from pydantic_settings import (
    BaseSettings,
    EnvSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from ontogr_mem.llm.roles import ModelRole


OutputMode = Literal["tool", "native", "prompted"]


def _without_blanks(value: Any) -> Any:
    if not isinstance(value, Mapping):
        return value
    return {
        key: item
        for key, item in value.items()
        if not isinstance(item, str) or item.strip()
    }


class RoleOverride(BaseModel):
    """Optional settings that replace global values for one kind of work.

    Attributes:
        model: Model name, with an optional protocol prefix.
        base_url: Explicit provider endpoint.
        api_key: Provider credential, hidden from representations.
        output_mode: Structured output protocol.
        max_output_tokens: Positive output cap.
        reasoning_effort: Provider reasoning effort.
        timeout_s: Positive HTTP attempt timeout in seconds.
        extra_body: Provider request additions; replaces the global object.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")
    model: str | None = None
    base_url: str | None = None
    api_key: SecretStr | None = None
    output_mode: OutputMode | None = None
    max_output_tokens: int | None = Field(default=None, ge=1)
    reasoning_effort: str | None = None
    timeout_s: float | None = Field(default=None, gt=0)
    extra_body: dict[str, Any] | None = None

    @model_validator(mode="before")
    @classmethod
    def read_values(cls, value: Any) -> Any:
        """Treat blank values as absent and decode the provider JSON object."""
        cleaned = _without_blanks(value)
        if isinstance(cleaned, dict) and isinstance(cleaned.get("extra_body"), str):
            cleaned["extra_body"] = json.loads(cleaned["extra_body"])
        return cleaned


@dataclass(frozen=True, slots=True)
class RoleConfig:
    """Resolved settings for one kind of work.

    Attributes:
        role: Kind of work.
        model: Full model name, or None when not configured.
        base_url: Explicit endpoint, required when a model name is set.
        api_key: Optional credential.
        output_mode: Structured output protocol.
        max_output_tokens: Output cap, or None to omit it.
        reasoning_effort: Optional provider reasoning effort.
        timeout_s: HTTP attempt timeout in seconds.
        extra_body: Optional provider request additions.
    """

    role: ModelRole
    model: str | None
    base_url: str | None
    api_key: SecretStr | None
    output_mode: OutputMode
    max_output_tokens: int | None
    reasoning_effort: str | None
    timeout_s: float
    extra_body: Mapping[str, Any] | None


def _read_role_source(source: PydanticBaseSettingsSource) -> dict[str, Any]:
    if not isinstance(source, EnvSettingsSource):
        return source()
    source.env_vars = {
        key: value
        for key, value in source.env_vars.items()
        if value is not None and value.strip()
    }
    values = source()
    roles: dict[str, Any] = {}
    for role in ModelRole:
        prefix = f"memory_llm_{role.value}_"
        overrides = {
            name: source.env_vars[prefix + name]
            for name in RoleOverride.model_fields
            if prefix + name in source.env_vars
        }
        if overrides:
            roles[role.value] = overrides
    if roles:
        values["roles"] = roles
    return values


class LLMConfig(BaseSettings):
    """Resolve model settings from MEMORY_LLM_ variables.

    Attributes:
        model: Global model. Env: MEMORY_LLM_MODEL.
        base_url: Endpoint. Env: MEMORY_LLM_BASE_URL.
        api_key: Credential. Env: MEMORY_LLM_API_KEY.
        output_mode: Output protocol. Env: MEMORY_LLM_OUTPUT_MODE.
        max_output_tokens: Global cap. Env: MEMORY_LLM_MAX_OUTPUT_TOKENS.
        reasoning_effort: Effort. Env: MEMORY_LLM_REASONING_EFFORT.
        timeout_s: Attempt seconds. Env: MEMORY_LLM_TIMEOUT_S.
        extra_body: JSON object. Env: MEMORY_LLM_EXTRA_BODY.
        max_concurrency: Shared slots. Env: MEMORY_LLM_MAX_CONCURRENCY.
        roles: Overrides from MEMORY_LLM_<ROLE>_<FIELD> variables.
    """

    model_config = SettingsConfigDict(
        env_prefix="MEMORY_LLM_",
        env_file=".env",
        extra="ignore",
        frozen=True,
    )
    model: str | None = None
    base_url: str | None = None
    api_key: SecretStr | None = None
    output_mode: OutputMode = "tool"
    max_output_tokens: int | None = Field(default=None, ge=1)
    reasoning_effort: str | None = None
    timeout_s: float = Field(default=300.0, gt=0)
    extra_body: dict[str, Any] | None = None
    max_concurrency: int = Field(default=4, ge=1)
    roles: Mapping[ModelRole, RoleOverride] = Field(default_factory=dict)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[Any, ...]:
        """Read flat role variables with the normal settings precedence."""
        return (
            init_settings,
            partial(_read_role_source, env_settings),
            partial(_read_role_source, dotenv_settings),
            file_secret_settings,
        )

    @model_validator(mode="before")
    @classmethod
    def read_values(cls, value: Any) -> Any:
        """Ignore blanks and accept a JSON object from explicit construction."""
        cleaned = _without_blanks(value)
        if isinstance(cleaned, dict) and isinstance(cleaned.get("extra_body"), str):
            cleaned["extra_body"] = json.loads(cleaned["extra_body"])
        return cleaned

    @model_validator(mode="after")
    def validate_models(self) -> Self:
        """Reject ambiguous protocols and missing explicit endpoints."""
        for role in ModelRole:
            resolved = self.resolve(role)
            if resolved.model and resolved.model.startswith("openai:"):
                raise ValueError("use openai-chat: or openai-responses:, not openai:")
            if resolved.model and not resolved.base_url:
                raise ValueError(
                    f"set MEMORY_LLM_{role.name}_BASE_URL or MEMORY_LLM_BASE_URL"
                )
        return self

    def resolve(self, role: ModelRole) -> RoleConfig:
        """Return inherited values and the kind-specific default output cap."""
        override = self.roles.get(role, RoleOverride())
        cap = override.max_output_tokens
        if cap is None:
            cap = self.max_output_tokens
        if cap is None and role is not ModelRole.ANSWER:
            cap = 16384 if role is ModelRole.FACT_EXTRACTION else 2048
        return RoleConfig(
            role=role,
            model=override.model or self.model,
            base_url=override.base_url or self.base_url,
            api_key=override.api_key if override.api_key is not None else self.api_key,
            output_mode=override.output_mode or self.output_mode,
            max_output_tokens=cap,
            reasoning_effort=override.reasoning_effort or self.reasoning_effort,
            timeout_s=(
                override.timeout_s if override.timeout_s is not None else self.timeout_s
            ),
            extra_body=(
                override.extra_body
                if override.extra_body is not None
                else self.extra_body
            ),
        )
