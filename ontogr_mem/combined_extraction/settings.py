"""Settings that choose how a text write extracts entities and facts."""

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class ExtractionSettings(BaseSettings):
    """Choose the extraction structure of a text write."""

    model_config = SettingsConfigDict(
        env_prefix="MEMORY_EXTRACTION_",
        env_file=".env",
        extra="ignore",
        frozen=True,
    )

    mode: Literal["two_pass", "combined"] = "two_pass"
