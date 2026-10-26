"""The pairs the verifier works on and the results it returns.

A pair carries the two entities to compare and the signals that brought them
together. An outcome carries the verdict of one pair, or the reason there is
none.
"""

from dataclasses import dataclass, field

from ontogr_mem.data_models.entities import Entity, VerdictKind, VerifyStatus


@dataclass(frozen=True, slots=True)
class PairInput:
    """One pair to verify.

    Attributes:
        a: First entity.
        b: Second entity.
        two_votes: Whether a YES must repeat with the sides swapped.
        cosine: Cosine similarity, if the pair came from an embedding search.
        jaccard: Jaccard similarity, if the pair came from blocking.
        a_aliases: Aliases of the first entity.
        b_aliases: Aliases of the second entity.
        a_quotes: Quotes that support the first entity.
        b_quotes: Quotes that support the second entity.
    """

    a: Entity
    b: Entity
    two_votes: bool = False
    cosine: float | None = None
    jaccard: float | None = None
    a_aliases: tuple[str, ...] = ()
    b_aliases: tuple[str, ...] = ()
    a_quotes: tuple[str, ...] = ()
    b_quotes: tuple[str, ...] = ()


@dataclass(slots=True)
class PairOutcome:
    """Verification result for one pair. ``verdict`` is None when nothing is known."""

    a_id: str
    b_id: str
    verdict: VerdictKind | None
    status: VerifyStatus
    passes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class VerifyStats:
    """Counters of one ``verify`` call."""

    calls: int = 0
    invalid_ids: int = 0
    reasked: int = 0
