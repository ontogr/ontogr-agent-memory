"""Candidate ordering and the vote rule shared by both verification tiers.

Cosine and Jaccard sit on separate scales with separate floors, so a raw score
never orders candidates across signals. Each signal is measured by how far it
clears its own floor, and the best margin wins.
"""

from dataclasses import dataclass

from ontogr_mem.resolution.config import ResolverSettings


@dataclass(frozen=True, slots=True)
class PairSignals:
    """The similarity signals measured for one pair of entities.

    Cosine and Jaccard are on separate scales, so each keeps its own field and
    stays None when that signal did not propose the pair.

    Attributes:
        cosine: Cosine similarity from the vector search, or None.
        jaccard: Exact Jaccard similarity from blocking, or None.
    """

    cosine: float | None = None
    jaccard: float | None = None


@dataclass(frozen=True, slots=True)
class Candidate:
    """A candidate entity with the signals that proposed it.

    Attributes:
        entity_id: Id of the candidate entity.
        signals: The similarity signals that proposed the candidate.
    """

    entity_id: str
    signals: PairSignals = PairSignals()


def margin(score: float, floor: float) -> float:
    """Return how far ``score`` clears ``floor`` on a 0 to 1 scale.

    Args:
        score: Similarity value.
        floor: Minimum value the signal must clear.

    Returns:
        ``(score - floor) / (1 - floor)`` clamped at 0. A floor of 1 gives 0.
    """
    if floor >= 1.0:
        return 0.0
    return max(0.0, (score - floor) / (1.0 - floor))


def rank_key(candidate: Candidate, settings: ResolverSettings) -> tuple[float, str]:
    """Return the sort key of a candidate, best first.

    The first element is the negated best margin, so ascending order puts the
    strongest candidate first. Equal margins fall back to the smaller id.

    Args:
        candidate: Candidate to rank.
        settings: Thresholds that define each signal's floor.

    Returns:
        ``(-best margin, entity id)``.
    """
    return (-_best_margin(candidate, settings), candidate.entity_id)


def needs_two_votes(
    signals: PairSignals,
    settings: ResolverSettings,
    cross_family: bool,
) -> bool:
    """Return whether the pair needs a second, swapped verification vote.

    A cross-family pair always needs two votes. Otherwise a strong enough
    cosine or Jaccard value is accepted on one vote. Each signal is read
    against its own floor, so a Jaccard value is never weighed as a cosine.
    A pair that a rule proposed exactly is settled by the caller.

    Args:
        signals: The signals measured for the pair.
        settings: Thresholds for a single-vote answer.
        cross_family: True when the two entities share a merge family but not a
            type.

    Returns:
        True when a second vote is required.
    """
    if cross_family:
        return True
    if signals.cosine is not None and signals.cosine >= settings.single_vote_min:
        return False
    jaccard_strong = (
        signals.jaccard is not None
        and signals.jaccard >= settings.jaccard_single_vote_min
    )
    return not jaccard_strong


def _best_margin(candidate: Candidate, settings: ResolverSettings) -> float:
    best = 0.0
    if candidate.signals.cosine is not None:
        best = max(best, margin(candidate.signals.cosine, settings.ann_floor))
    if candidate.signals.jaccard is not None:
        best = max(best, margin(candidate.signals.jaccard, settings.jaccard_floor))
    return best
