"""Settings for the built-in embedding providers."""

from typing import Literal

from pydantic import Field, JsonValue, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingSettings(BaseSettings):
    """Configure an embedding endpoint independently of language models.

    Attributes:
        model: Endpoint model name. Env: MEMORY_EMBEDDING_MODEL.
        base_url: Explicit endpoint URL. Env: MEMORY_EMBEDDING_BASE_URL.
        api_key: Endpoint key. Env: MEMORY_EMBEDDING_API_KEY.
        dimensions: Expected vector width, or discover it on open.
            Env: MEMORY_EMBEDDING_DIMENSIONS.
        max_input_tokens: Declared endpoint input limit.
            Env: MEMORY_EMBEDDING_MAX_INPUT_TOKENS.
        provider: Embedding provider. Env: MEMORY_EMBEDDING_PROVIDER.
        fastembed_kwargs: Native FastEmbed options as a JSON object.
            Env: MEMORY_EMBEDDING_FASTEMBED_KWARGS.
        batch_size: Native FastEmbed call size. Env: MEMORY_EMBEDDING_BATCH_SIZE.
    """

    model_config = SettingsConfigDict(
        env_prefix="MEMORY_EMBEDDING_",
        env_file=".env",
        extra="ignore",
        frozen=True,
        hide_input_in_errors=True,
    )

    provider: Literal["auto", "fastembed", "endpoint"] = "auto"
    fastembed_kwargs: dict[str, JsonValue] = Field(default_factory=dict, repr=False)
    batch_size: int = Field(default=32, ge=1)
    model: str | None = None
    base_url: str | None = None
    api_key: SecretStr | None = None
    dimensions: int | None = Field(default=None, ge=1, le=4000)
    max_input_tokens: int = Field(default=8192, ge=1)

    @field_validator("model", "base_url", "api_key", "dimensions", mode="before")
    @classmethod
    def _blank_is_missing(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value
