"""The retrieval lanes and their shared result type."""

from dataclasses import dataclass, field

from ontogr_mem.retrieval.models import LaneRecord
from ontogr_mem.temporal.query import FactView


@dataclass(frozen=True, slots=True)
class LaneOutput:
    """Ordered results and status from one lane."""

    record: LaneRecord
    facts: tuple[FactView, ...] = ()
    episodes: tuple[str, ...] = ()
    observation_ids: tuple[str, ...] = ()
    entities: tuple[str, ...] = ()
    entity_sources: dict[str, tuple[str, ...]] = field(default_factory=dict)
    error: Exception | None = None
