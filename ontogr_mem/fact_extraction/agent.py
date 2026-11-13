"""Typed shape of fact extraction output."""

from typing import Annotated

from pydantic import BaseModel, Field, StrictInt, StrictStr
from pydantic_ai import Agent

from ontogr_mem.data_models.facts import FactKind, Modality, Polarity, Precision
from ontogr_mem.fact_extraction.prompts import SYSTEM


class FactOutput(BaseModel):
    """A proposed fact with unchecked references, evidence and time strings."""

    text: StrictStr
    entity_refs: list[StrictStr]
    relation_type_id: StrictInt | None = None
    fact_kind: FactKind
    polarity: Polarity
    modality: Modality
    evidence_quote: StrictStr
    relation_label: StrictStr | None = None
    valid_from: StrictStr | None = None
    valid_to: StrictStr | None = None
    precision: Precision | None = None
    time_expression: StrictStr | None = None
    confidence: Annotated[float, Field(strict=True, ge=0, le=1)] | None = None


class FactsOutput(BaseModel):
    """The whole fact reply; one malformed item invalidates the reply."""

    facts: list[FactOutput]


fact_agent = Agent[None, FactsOutput](
    output_type=FactsOutput,
    instructions=SYSTEM,
    name="fact_extraction",
)
