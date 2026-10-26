"""Retry, breaker, batching and lock settings shared by every backend."""

from pydantic import BaseModel, ConfigDict, Field


class ResilienceSettings(BaseModel):
    """Retry, breaker, batching and lock settings.

    `max_attempts` is the total number of tries, not the number of retries.
    """

    model_config = ConfigDict(frozen=True)

    max_attempts: int = Field(default=3, ge=1)
    retry_delay: float = Field(default=1.0, ge=0)
    query_timeout_ms: int = Field(default=10_000, ge=1)
    breaker_failure_threshold: int = Field(default=5, ge=1)
    breaker_recovery_s: float = Field(default=30.0, ge=0)
    batch_size: int = Field(default=500, ge=1)
    slow_query_ms: int = Field(default=1_000, ge=0)
