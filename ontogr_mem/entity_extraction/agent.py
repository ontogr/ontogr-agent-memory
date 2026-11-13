"""Typed shape of entity extraction output."""

from typing import Annotated

from pydantic import BaseModel, Field, StrictInt, StrictStr
from pydantic_ai import Agent

from ontogr_mem.entity_extraction.prompts import NER_SYSTEM


class EntityOutput(BaseModel):
    """A proposed name and dense type id, before domain binding."""

    name: StrictStr
    type_id: StrictInt
    type_label: StrictStr | None = None
    description: StrictStr = ""
    confidence: Annotated[float, Field(strict=True, ge=0, le=1)] | None = None


class EntitiesOutput(BaseModel):
    """The whole entity reply; one malformed item invalidates the reply."""

    entities: list[EntityOutput]


entity_agent = Agent[None, EntitiesOutput](
    output_type=EntitiesOutput,
    instructions=NER_SYSTEM,
    name="entity_extraction",
)
