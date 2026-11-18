"""Settings for temporal reconciliation."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class TemporalSettings(BaseSettings):
    """Retry and batch limits for the temporal engine.

    Environment variables use the prefix ``MEMORY_TEMPORAL_``.

    Attributes:
        max_retries: Retries for a slot that lost a revision race. Env:
            ``MEMORY_TEMPORAL_MAX_RETRIES``.
        batch_size: Observations written in one store call. Env:
            ``MEMORY_TEMPORAL_BATCH_SIZE``.
        episode_ttl_s: Seconds an episode may stay ``PENDING`` before the audit
            reports it. Env: ``MEMORY_TEMPORAL_EPISODE_TTL_S``.
    """

    model_config = SettingsConfigDict(
        env_prefix="MEMORY_TEMPORAL_",
        env_file=".env",
        extra="ignore",
        frozen=True,
    )

    max_retries: int = Field(default=1, ge=0)
    batch_size: int = Field(default=100, ge=1)
    episode_ttl_s: int = Field(default=3600, ge=1)


class ClosureSettings(BaseSettings):
    """Settings of the step that closes older facts by one model call.

    Environment variables use the ``MEMORY_CLOSURE_`` prefix.

    Attributes:
        enabled: Whether the step runs. Env: ``MEMORY_CLOSURE_ENABLED``.
        max_candidates: Most existing facts sent with one new fact. This count
            is the only bound on a model call; there is no token budget. Env:
            ``MEMORY_CLOSURE_MAX_CANDIDATES``.
        prompt_version: Version stored with each verdict. Env:
            ``MEMORY_CLOSURE_PROMPT_VERSION``.
    """

    model_config = SettingsConfigDict(
        env_prefix="MEMORY_CLOSURE_",
        env_file=".env",
        extra="ignore",
        frozen=True,
    )

    enabled: bool = True
    max_candidates: int = Field(default=20, ge=1)
    prompt_version: str = "closure-1"
