"""Settings for the built-in endpoint embedder."""

from pydantic import Field, SecretStr, field_validator
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
    """

    model_config = SettingsConfigDict(
        env_prefix="MEMORY_EMBEDDING_",
        env_file=".env",
        extra="ignore",
        frozen=True,
    )

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
