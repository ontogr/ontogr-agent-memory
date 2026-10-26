"""Settings of the vector store."""

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from ontogr_mem.resilience import ProviderResilienceSettings


class VectorStoreSettings(ProviderResilienceSettings):
    """Embedding gateway limits, batch sizes and the over-fetch factor.

    Environment variables use the prefix `MEMORY_VECTORSTORE_`.

    Attributes:
        batch_size: Texts per embedder call.
        concurrency: Embedder calls at the same time.
        attempt_timeout_s: Time limit of one embedder call.
        deadline_s: Time limit of one `embed` request, retries included.
        max_attempts: Tries for one text after transient failures.
        backoff_base_s: First retry delay.
        backoff_max_s: Longest retry delay.
        token_cap: Longest text, in tokens, sent in one piece.
        chars_per_token: Characters assumed per token when cutting long texts.
        write_batch: Rows per backend write.
        id_batch: Ids per backend lookup.
        overfetch: Factor by which a filtered search fetches extra neighbours.
        max_rechecks: Additional search rounds after the first query when
            filtering removes too many neighbours.
        ef_runtime: FalkorDB HNSW search width, set on the vector index.
        breaker_threshold: Failures that open the breaker, or None to disable it.
        breaker_recovery_s: Seconds before the breaker allows one probe.
        requests_per_minute: Optional request-start limit for the embedder.
    """

    model_config = SettingsConfigDict(
        env_prefix="MEMORY_VECTORSTORE_", env_file=".env", extra="ignore", frozen=True
    )

    batch_size: int = Field(default=36, ge=1)
    concurrency: int = Field(default=4, ge=1)
    attempt_timeout_s: float = Field(default=60.0, gt=0)
    deadline_s: float = Field(default=300.0, gt=0)
    max_attempts: int = Field(default=3, ge=1)
    backoff_base_s: float = Field(default=0.5, ge=0)
    backoff_max_s: float = Field(default=8.0, ge=0)
    token_cap: int = Field(default=4096, ge=1)
    chars_per_token: int = Field(default=4, ge=1)
    write_batch: int = Field(default=500, ge=1)
    id_batch: int = Field(default=1000, ge=1)
    overfetch: float = Field(default=2.0, ge=1.0)
    max_rechecks: int = Field(default=3, ge=0)
    ef_runtime: int = Field(default=200, ge=1)
