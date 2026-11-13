"""Named kinds of model work."""

from enum import StrEnum


class ModelRole(StrEnum):
    """A kind of work with independently resolved model configuration."""

    ENTITY_EXTRACTION = "entity_extraction"
    FACT_EXTRACTION = "fact_extraction"
    RESOLUTION = "resolution"
    JUDGE = "judge"
    CONSOLIDATION = "consolidation"
    ONTOLOGY = "ontology"
    ANSWER = "answer"
