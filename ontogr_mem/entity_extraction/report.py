"""The result of extracting a batch of episodes."""

from collections import Counter
from dataclasses import dataclass, field

from ontogr_mem.data_models import RETRYABLE, ExtractionResult, Status


@dataclass(slots=True)
class ExtractionReport:
    """Results for a batch, with counts for deciding what to retry.

    Attributes:
        results: One result per episode, in input order.
        trigger_stats: How often each escalation trigger fired in the batch.
    """

    results: list[ExtractionResult] = field(default_factory=list)
    trigger_stats: dict[str, int] = field(default_factory=dict)

    def status_counts(self) -> dict[str, int]:
        """Count episodes per status, with every status present."""
        counts = Counter(r.status.value for r in self.results)
        return {s.value: counts.get(s.value, 0) for s in Status}

    @property
    def failed_episode_ids(self) -> list[str]:
        """Ids of the episodes that should be extracted again."""
        return [r.episode_id for r in self.results if r.status in RETRYABLE]
