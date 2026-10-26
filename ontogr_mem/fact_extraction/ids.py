"""Deterministic ids for extracted facts.

A fact id is a pure function of the episode, the normalized text and the
ordinal of the fact among the facts with the same text. The same input always
gives the same id, across runs and machines.
"""

import uuid
from dataclasses import replace

from ontogr_mem.data_models.facts import ExtractedFact
from ontogr_mem.fact_extraction.text import normalize_text


# Fixed so that the same fact gets the same id across runs and machines.
FACT_NAMESPACE = uuid.UUID("6f1c2a0e-5a53-4a43-9f0e-3c1d6a0b7e11")


def fact_id(episode_id: str, text: str, ordinal: int) -> str:
    """Return a `uuid5` over the episode id, the normalized text and the ordinal."""
    return str(
        uuid.uuid5(FACT_NAMESPACE, f"{episode_id}|{normalize_text(text)}|{ordinal}")
    )


def assign_ids(episode_id: str, facts: list[ExtractedFact]) -> list[ExtractedFact]:
    """Give each fact its id; same-text facts are told apart by ordinal.

    The ordinal counts earlier facts in the episode with the same normalized text.
    """
    ordinals: dict[str, int] = {}
    out: list[ExtractedFact] = []
    for fact in facts:
        norm = normalize_text(fact.text)
        ordinal = ordinals.get(norm, 0)
        ordinals[norm] = ordinal + 1
        out.append(replace(fact, fact_id=fact_id(episode_id, fact.text, ordinal)))
    return out
