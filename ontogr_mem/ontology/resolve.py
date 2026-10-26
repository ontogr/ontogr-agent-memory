"""Map a free-form type string to a catalog type.

Order: normalized exact on the label, then alias, then fuzzy. Two candidates
within a small margin are ``ambiguous`` and return no match. The index is built
once per catalog version.
"""

from dataclasses import dataclass
from difflib import SequenceMatcher
from enum import StrEnum

from ontogr_mem.ontology.models import Kind, Ontology
from ontogr_mem.ontology.names import normalize


class How(StrEnum):
    """How a string matched a type."""

    EXACT = "exact"
    ALIAS = "alias"
    FUZZY = "fuzzy"


@dataclass(frozen=True, slots=True)
class TypeMatch:
    """A resolved type and how it was found; a fuzzy match carries its score."""

    type_id: int
    how: How
    score: float


@dataclass(frozen=True, slots=True)
class Resolution:
    """Outcome of a lookup.

    ``match`` is None when nothing fits or the text is ambiguous. ``ambiguous`` lists
    the candidate type ids that were too close to choose between.
    """

    match: TypeMatch | None
    ambiguous: tuple[int, ...] = ()


class TypeResolver:
    """Resolver for one kind over one catalog version.

    The index is built once from labels and aliases. A merged type resolves to its live
    target.
    """

    def __init__(
        self,
        cat: Ontology,
        kind: Kind,
        cutoff: float = 0.8,
        margin: float = 0.05,
    ) -> None:
        """Build the index.

        Args:
            cat: The catalog to index.
            kind: Entity or relation types.
            cutoff: Lowest similarity ratio that counts as a fuzzy match.
            margin: Two candidates closer than this are ambiguous.
        """
        self.cutoff = cutoff
        self.margin = margin
        self._exact: dict[str, int] = {}
        self._alias: dict[str, int] = {}
        for t in cat.types(kind):
            live = cat.follow(kind, t.type_id)
            self._exact[normalize(t.label)] = live
            for al in t.aliases:
                self._alias.setdefault(normalize(al), live)
        self._keys = {**self._alias, **self._exact}

    def resolve(self, text: str) -> Resolution:
        """Resolve ``text`` to a type.

        The order is the normalized label, then an alias, then a fuzzy match at or
        above the cutoff. Two fuzzy candidates within the margin return no match and
        list both in ``ambiguous``.
        """
        key = normalize(text)
        if not key:
            return Resolution(None)
        if key in self._exact:
            return Resolution(TypeMatch(self._exact[key], How.EXACT, 1.0))
        if key in self._alias:
            return Resolution(TypeMatch(self._alias[key], How.ALIAS, 1.0))
        scored: dict[int, float] = {}
        for cand, type_id in self._keys.items():
            score = SequenceMatcher(None, key, cand).ratio()
            if score >= self.cutoff and score > scored.get(type_id, 0.0):
                scored[type_id] = score
        if not scored:
            return Resolution(None)
        ranked = sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))
        best_id, best = ranked[0]
        near = tuple(i for i, s in ranked if best - s <= self.margin)
        if len(near) > 1:
            return Resolution(None, near)
        return Resolution(TypeMatch(best_id, How.FUZZY, best))
