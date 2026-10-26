"""Reciprocal rank fusion of the ranked lists of the retrieval lanes."""

from collections.abc import Hashable, Sequence
from typing import TypeVar

from ontogr_mem.retrieval.settings import RetrievalSettings


Key = TypeVar("Key", bound=Hashable)
"""A lane key. It is hashable, because it is a dict key, and orderable,
because ties are broken by the key."""


def fuse(
    lists: Sequence[Sequence[Key]],
    *,
    limit: int,
    settings: RetrievalSettings,
) -> list[tuple[Key, float]]:
    """Fuse the ranked lists of the lanes into one ranking.

    A lane list is de-duplicated first, so a key that repeats inside one lane
    counts once and keeps its first position. A key scores the sum of
    ``1 / (k + position)`` over the lanes that hold it, with the position
    counted from 1. The score of the result is that sum divided by the best
    sum of the same call, so the top key scores 1.0.

    Args:
        lists: The ranked lists of the lanes, best key first.
        limit: The number of results the caller wants. It sets the fusion
            constant.
        settings: The fusion constants.

    Returns:
        The fused keys and their scores, best first. Keys with the same score
        keep the order of their keys, smallest first.
    """
    constant = _fusion_constant(limit, settings)
    scores: dict[Key, float] = {}
    for lane in lists:
        seen: set[Key] = set()
        position = 0
        for key in lane:
            if key in seen:
                continue
            seen.add(key)
            position += 1
            scores[key] = scores.get(key, 0.0) + 1.0 / (constant + position)
    if not scores:
        return []
    best = max(scores.values())
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [(key, value / best) for key, value in ordered]


def _fusion_constant(limit: int, settings: RetrievalSettings) -> int:
    """Return the reciprocal rank fusion constant for the given limit.

    Args:
        limit: The number of results the caller wants.
        settings: The fusion constants.

    Returns:
        The constant, kept between the floor and the ceiling of the settings.
    """
    raw = settings.fusion_k_base + settings.fusion_k_per_limit * limit
    return min(max(raw, settings.fusion_k_min), settings.fusion_k_max)
