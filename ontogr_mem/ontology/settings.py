"""Settings for the ontology manager."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class OntologySettings(BaseSettings):
    """Thresholds for proposals and retries of the ontology manager.

    Environment variables use the prefix `MEMORY_ONTOLOGY_`. The default numbers are
    judgment calls, not measurements.
    """

    model_config = SettingsConfigDict(
        env_prefix="MEMORY_ONTOLOGY_",
        env_file=".env",
        extra="ignore",
        frozen=True,
    )

    min_support: int = Field(default=3, ge=1)
    regrow_factor: int = Field(default=2, ge=1)
    max_draft_retries: int = Field(default=2, ge=0)
    commit_retries: int = Field(default=3, ge=1)
    prompt_version: str = "1"
