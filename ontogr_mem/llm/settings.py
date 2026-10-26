"""Settings for the chat model gateway."""

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from ontogr_mem.resilience import ProviderResilienceSettings


class LLMSettings(ProviderResilienceSettings):
    """Limits and retry policy for LLM calls.

    Environment variables use the prefix ``MEMORY_LLM_``.

    Attributes:
        max_concurrency: Calls allowed per model size.
            Env: ``MEMORY_LLM_MAX_CONCURRENCY``.
        max_attempts: Total provider tries; 1 disables retries.
            Env: ``MEMORY_LLM_MAX_ATTEMPTS``.
        base_delay_s: Initial retry delay. Env: ``MEMORY_LLM_BASE_DELAY_S``.
        max_delay_s: Maximum retry delay. Env: ``MEMORY_LLM_MAX_DELAY_S``.
        breaker_threshold: Consecutive failures that open the breaker, or None.
            Env: ``MEMORY_LLM_BREAKER_THRESHOLD``.
        breaker_recovery_s: Seconds before one breaker probe.
            Env: ``MEMORY_LLM_BREAKER_RECOVERY_S``.
        requests_per_minute: Optional provider request limit.
            Env: ``MEMORY_LLM_REQUESTS_PER_MINUTE``.
        structured_attempts: Maximum validated-output attempts.
            Env: ``MEMORY_LLM_STRUCTURED_ATTEMPTS``.
    """

    model_config = SettingsConfigDict(
        env_prefix="MEMORY_LLM_", env_file=".env", extra="ignore", frozen=True
    )

    max_concurrency: int = Field(default=4, ge=1)
    max_attempts: int = Field(default=1, ge=1)
    base_delay_s: float = Field(default=0.5, ge=0)
    max_delay_s: float = Field(default=30.0, ge=0)
    structured_attempts: int = Field(default=3, ge=1)
