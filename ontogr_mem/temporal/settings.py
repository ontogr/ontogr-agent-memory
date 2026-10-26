"""Settings for temporal reconciliation."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from ontogr_mem.llm import ModelSize


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


class JudgeSettings(BaseSettings):
    """Limits and thresholds of the model judge.

    Environment variables use the prefix ``MEMORY_JUDGE_``.

    Attributes:
        floor: Confidence a contradiction needs before it is confirmed. Below it
            the pair is stored as uncertain. Env: ``MEMORY_JUDGE_FLOOR``.
        prompt_version: Version recorded with each verdict. A new version makes
            stored verdicts stale. Env: ``MEMORY_JUDGE_PROMPT_VERSION``.
        model_size: Model tier the judge calls. Env:
            ``MEMORY_JUDGE_MODEL_SIZE``.
        max_pairs: Pairs per model call. Env: ``MEMORY_JUDGE_MAX_PAIRS``.
        token_budget: Estimated input tokens per model call. Env:
            ``MEMORY_JUDGE_TOKEN_BUDGET``.
        reask: Extra calls for pairs the model left unanswered. Env:
            ``MEMORY_JUDGE_REASK``.
    """

    model_config = SettingsConfigDict(
        env_prefix="MEMORY_JUDGE_",
        env_file=".env",
        extra="ignore",
        frozen=True,
    )

    floor: float = Field(default=0.8, ge=0.0, le=1.0)
    prompt_version: str = "judge-1"
    model_size: ModelSize = ModelSize.SMALL
    max_pairs: int = Field(default=4, ge=1)
    token_budget: int = Field(default=3000, ge=1)
    reask: int = Field(default=1, ge=0)
