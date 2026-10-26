"""Escalation thresholds and the triggers that judge a cheap extraction pass."""

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from ontogr_mem.data_models import UNKNOWN_TYPE_ID, Episode, ExtractedEntity
from ontogr_mem.entity_extraction.extractors.base import Trigger
from ontogr_mem.entity_extraction.text import text_of


@dataclass(frozen=True, slots=True)
class EscalationThresholds:
    """Limits that decide when a cheap pass counts as bad.

    The defaults are placeholders. Set them from a labelled sample for the model
    in use.
    """

    min_entities_per_1k_chars: float = 3.0
    max_unknown_share: float = 0.5
    min_mean_confidence: float = 0.5
    domain_terms: frozenset[str] = field(default_factory=frozenset)


def evaluate_triggers(
    episode: Episode,
    entities: Sequence[ExtractedEntity],
    thresholds: EscalationThresholds,
) -> tuple[Trigger, ...]:
    """Check merged entities against the thresholds and return the triggers hit.

    The expected entity count is a rate, so it is compared without rounding. A
    text that calls for 2.7 entities is short when only two were found, and any
    non-empty text calls for at least one entity.

    Args:
        episode: The episode the entities came from.
        entities: The merged entities of that episode.
        thresholds: The limits to judge them by.

    Returns:
        The triggers that fired, in the order they are checked.
    """
    text = text_of(episode)
    fired: list[Trigger] = []

    expected = len(text) / 1000 * thresholds.min_entities_per_1k_chars
    if len(entities) < expected:
        fired.append(Trigger.TOO_FEW_ENTITIES)

    if entities:
        unknown = sum(e.type_id == UNKNOWN_TYPE_ID for e in entities)
        if unknown / len(entities) > thresholds.max_unknown_share:
            fired.append(Trigger.HIGH_UNKNOWN_SHARE)
        mean = sum(e.confidence for e in entities) / len(entities)
        if mean < thresholds.min_mean_confidence:
            fired.append(Trigger.LOW_MEAN_CONFIDENCE)

    covered = [span for e in entities for span in e.spans]
    for term in thresholds.domain_terms:
        pattern = rf"(?<!\w){re.escape(term)}(?!\w)"
        for m in re.finditer(pattern, text, re.IGNORECASE):
            if not any(s < m.end() and m.start() < e for s, e in covered):
                fired.append(Trigger.DOMAIN_TERM_UNCOVERED)
                return tuple(fired)
    return tuple(fired)
