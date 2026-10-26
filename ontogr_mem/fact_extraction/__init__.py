"""Fact extraction: turn episodes into grounded, typed claims."""

from ontogr_mem.data_models.facts import (
    Evidence,
    ExtractedFact,
    FactKind,
    Modality,
    Polarity,
    Precision,
    TimeBasis,
    TypedFact,
)
from ontogr_mem.data_models.type_catalog import TypeMode, TypeView
from ontogr_mem.fact_extraction.errors import FactExtractionError, UnknownKindError
from ontogr_mem.fact_extraction.extractor import (
    FactExtractor,
    LLMFactExtractor,
    RoutingFactExtractor,
)
from ontogr_mem.fact_extraction.models import (
    FactDropped,
    FactDropReason,
    FactReport,
    FactResult,
    SkipReason,
)
from ontogr_mem.fact_extraction.pipeline import (
    FactExtraction,
    FactStatusKey,
    FactStore,
    InMemoryFactStore,
)
from ontogr_mem.fact_extraction.record_facts import (
    FactMapping,
    FactTemplate,
    RecordFactMapper,
)
from ontogr_mem.fact_extraction.settings import FactExtractionSettings


__all__ = [
    "Evidence",
    "ExtractedFact",
    "FactDropReason",
    "FactDropped",
    "FactExtraction",
    "FactExtractionError",
    "FactExtractionSettings",
    "FactExtractor",
    "FactKind",
    "FactMapping",
    "FactReport",
    "FactResult",
    "FactStatusKey",
    "FactStore",
    "FactTemplate",
    "InMemoryFactStore",
    "LLMFactExtractor",
    "Modality",
    "Polarity",
    "Precision",
    "RecordFactMapper",
    "RoutingFactExtractor",
    "SkipReason",
    "TimeBasis",
    "TypeMode",
    "TypeView",
    "TypedFact",
    "UnknownKindError",
]
