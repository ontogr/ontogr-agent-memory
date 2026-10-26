"""Settings for entity extraction."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EntityExtractionSettings(BaseSettings):
    """Concurrency, context size and model thresholds.

    Environment variables use the prefix `MEMORY_ENTITY_EXTRACTION_`. The defaults
    are not measured values.

    Attributes:
        concurrency: Episodes extracted at the same time.
        max_previous: Earlier episodes shown to the model as context. Zero turns
            the context off.
        threshold: Model confidence below which an entity is typed Unknown.
        max_repairs: Extra model calls allowed after an invalid reply.
        default_confidence: Confidence used when the model gives none.
    """

    model_config = SettingsConfigDict(
        env_prefix="MEMORY_ENTITY_EXTRACTION_",
        env_file=".env",
        extra="ignore",
        frozen=True,
    )

    concurrency: int = Field(default=8, ge=1)
    max_previous: int = Field(default=3, ge=0)
    threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    max_repairs: int = Field(default=1, ge=0)
    default_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
