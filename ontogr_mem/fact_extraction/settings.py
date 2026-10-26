"""Settings for fact extraction."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class FactExtractionSettings(BaseSettings):
    """Concurrency, context size and model thresholds.

    Environment variables use the prefix `MEMORY_FACT_EXTRACTION_`. The defaults
    are not measured values.

    Attributes:
        concurrency: Episodes extracted at the same time.
        max_previous: Earlier episodes shown to the model as context. Zero turns
            the context off.
        min_entities: Candidate entities below which a non-message episode is
            skipped unless forced.
        default_confidence: Confidence used when the model gives none.
        languages: Languages the extractor accepts, or None for every language.
        instructions: Extra instructions added to the prompt.
        version: Extractor version; part of the per-episode status key.
    """

    model_config = SettingsConfigDict(
        env_prefix="MEMORY_FACT_EXTRACTION_",
        env_file=".env",
        extra="ignore",
        frozen=True,
    )

    concurrency: int = Field(default=8, ge=1)
    max_previous: int = Field(default=3, ge=0)
    min_entities: int = Field(default=1, ge=0)
    default_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    languages: frozenset[str] | None = None
    instructions: str = ""
    version: str = "fact-1"
