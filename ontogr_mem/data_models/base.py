"""Base model for the pydantic records of this package."""

from pydantic import BaseModel, ConfigDict


class _Model(BaseModel):
    """Base model: unknown fields are errors."""

    model_config = ConfigDict(extra="forbid")
