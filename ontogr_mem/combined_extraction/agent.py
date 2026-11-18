"""Validated output shape for combined extraction."""

from pydantic import BaseModel, StrictStr, model_validator
from pydantic_ai import Agent

from ontogr_mem.combined_extraction.prompts import COMBINED_SYSTEM
from ontogr_mem.entity_extraction.agent import EntityOutput
from ontogr_mem.fact_extraction.agent import FactOutput
from ontogr_mem.fact_extraction.candidates import ACTOR_ID


class CombinedEntityOutput(EntityOutput):
    """An entity of the combined reply, with the ref that facts use for it."""

    ref: StrictStr


class CombinedOutput(BaseModel):
    """The whole combined reply; one malformed item invalidates the reply."""

    entities: list[CombinedEntityOutput]
    facts: list[FactOutput]

    @model_validator(mode="after")
    def _refs_are_unique(self) -> "CombinedOutput":
        refs = [entity.ref for entity in self.entities]
        if len(set(refs)) != len(refs) or ACTOR_ID in refs:
            raise ValueError('entity refs must be unique and never "actor"')
        return self


combined_agent = Agent[None, CombinedOutput](
    output_type=CombinedOutput,
    instructions=COMBINED_SYSTEM,
    name="combined_extraction",
)
