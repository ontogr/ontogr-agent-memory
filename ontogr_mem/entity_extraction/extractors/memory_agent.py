"""Typed annotations for entities already found in an Episode."""

from pydantic import BaseModel, Field, StrictBool, StrictFloat, StrictInt, StrictStr
from pydantic_ai import Agent


ENRICH_SYSTEM = (
    "You annotate entities found in a user's message. Reply with JSON only: "
    '{"entities": [{"id": "e0", "description": "...", "attributes": '
    '{"<name>": {"value": "...", "evidence": "<exact quote from the text>"}}}]}. '
    "Use only facts stated in the text. Only use the allowed attribute names "
    "listed for each entity. Omit an attribute when the text does not state it."
)


class AttributeAnswer(BaseModel):
    """An attribute value and the Episode text that supports it.

    Attributes:
        value: A string, integer, float or boolean without type coercion.
        evidence: Exact Episode text that supports the value.
    """

    value: StrictStr | StrictInt | StrictFloat | StrictBool
    evidence: str


class EntityAnnotation(BaseModel):
    """Optional description and attributes for one extracted entity.

    Attributes:
        id: Local id of a base entity.
        description: Optional description grounded in the Episode.
        attributes: Scalar values with supporting Episode text.
    """

    id: str
    description: str = ""
    attributes: dict[str, AttributeAnswer] = Field(default_factory=dict)


class MemoryAnswer(BaseModel):
    """Annotations returned for one batch of base entities.

    Attributes:
        entities: Optional annotations keyed by the base entities' local ids.
    """

    entities: list[EntityAnnotation]


memory_agent: Agent[None, MemoryAnswer] = Agent(
    output_type=MemoryAnswer,
    instructions=ENRICH_SYSTEM,
)
