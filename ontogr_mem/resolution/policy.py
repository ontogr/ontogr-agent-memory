"""Per-type resolution policy, type relations and identity-attribute helpers.

``ResolutionPolicy`` says, for each type label, which merge family it belongs
to, whether a key hit may merge on its own (``AUTO_KEY``) or must be verified
first (``VERIFY_ALWAYS``), which attributes identify an entity, and whether legal
suffixes are stripped when looking for candidates. ``TypeRelations`` tells
whether two type labels are the same type or one is an ancestor of the other,
using the catalog the caller supplies.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from ontogr_mem.data_models.text import normalize_label
from ontogr_mem.data_models.type_catalog import UNKNOWN_LABEL
from ontogr_mem.resolution.keys import strict_key


PERSON = "person"
GENERIC_LABELS = frozenset({"entity", "thing", "concept", "other", "object"})

DEFAULT_FAMILIES: dict[str, str] = {
    **dict.fromkeys(
        ("person", "human", "employee", "patient", "author", "user", "customer"),
        PERSON,
    ),
    **dict.fromkeys(
        (
            "organization",
            "organisation",
            "company",
            "institution",
            "team",
            "university",
            "agency",
            "government",
        ),
        "organization",
    ),
    **dict.fromkeys(
        ("place", "location", "city", "country", "region", "address", "facility"),
        "place",
    ),
    **dict.fromkeys(("event", "meeting", "incident", "conference"), "event"),
    **dict.fromkeys(("time", "date", "period", "quarter"), "time"),
    **dict.fromkeys(
        ("work", "document", "book", "paper", "film", "software", "product"),
        "work",
    ),
}


class ResolutionMode(StrEnum):
    """How a key hit on a type is treated."""

    AUTO_KEY = "auto_key"
    VERIFY_ALWAYS = "verify_always"


class TypeRel(StrEnum):
    """Relation between the types of two entities.

    ``NESTED`` covers an ancestor, a descendant, ``Unknown`` and the generic
    labels; such pairs may match automatically. ``FAMILY`` pairs share a merge
    family and are verified only.
    """

    SAME = "same"
    NESTED = "nested"
    FAMILY = "family"
    DISTINCT = "distinct"


@dataclass(frozen=True, slots=True)
class TypePolicy:
    """Overrides for one type label.

    ``None`` means the value is derived from the family.

    Attributes:
        merge_family: Merge family of the type.
        mode: How a key hit on the type is treated.
        identity_attrs: Attribute names that identify an entity of the type.
        strip_suffixes: Whether candidates are also looked up by the
            suffix-stripped key.
    """

    merge_family: str | None = None
    mode: ResolutionMode | None = None
    identity_attrs: tuple[str, ...] = ()
    strip_suffixes: bool = False


class TypeRelations:
    """Ancestor links between type labels, built by the caller from the catalog."""

    def __init__(self, parents: Mapping[str, str | None] | None = None) -> None:
        """Create from a ``label -> parent label`` map.

        Labels use the canonical label normalization rules.
        """
        self._parents = {
            normalize_label(k): (normalize_label(v) if v else None)
            for k, v in (parents or {}).items()
        }

    def ancestors(self, label: str) -> list[str]:
        """Return the ancestors of ``label``, nearest first.

        A cycle ends the walk.
        """
        out: list[str] = []
        cur = self._parents.get(normalize_label(label))
        while cur is not None and cur not in out:
            out.append(cur)
            cur = self._parents.get(cur)
        return out

    def related(self, a: str, b: str) -> bool:
        """Return whether one label is an ancestor of the other."""
        a_c, b_c = normalize_label(a), normalize_label(b)
        return b_c in self.ancestors(a_c) or a_c in self.ancestors(b_c)

    def more_specific(self, a: str, b: str) -> str:
        """Return the descendant of two related labels.

        ``Unknown`` and the generic labels lose.
        """
        a_c, b_c = normalize_label(a), normalize_label(b)
        if _vague(a_c) and not _vague(b_c):
            return b
        if _vague(b_c) and not _vague(a_c):
            return a
        if a_c in self.ancestors(b_c):
            return b
        return a


def _vague(label: str) -> bool:
    normalized = normalize_label(label)
    return normalized == normalize_label(UNKNOWN_LABEL) or normalized in GENERIC_LABELS


class ResolutionPolicy:
    """Policy table with defaults derived from the merge-family table."""

    def __init__(
        self,
        types: Mapping[str, TypePolicy] | None = None,
        families: Mapping[str, str] | None = None,
    ) -> None:
        """Create the table; ``types`` and ``families`` are keyed by label."""
        self._types = {normalize_label(k): v for k, v in (types or {}).items()}
        fam = DEFAULT_FAMILIES if families is None else families
        self._families = {normalize_label(k): v for k, v in fam.items()}

    def family(self, label: str) -> str | None:
        """Return the merge family of a label, or None for a free label."""
        explicit = self._types.get(normalize_label(label))
        if explicit and explicit.merge_family:
            return explicit.merge_family
        return self._families.get(normalize_label(label))

    def mode(self, label: str) -> ResolutionMode:
        """Return how a key hit on this type is treated.

        An explicit per-type setting wins. Otherwise types in the person family
        default to ``VERIFY_ALWAYS``, because two real people can share a name
        and nothing in the extractor output tells them apart. Every other type
        defaults to ``AUTO_KEY``.
        """
        explicit = self._types.get(normalize_label(label))
        if explicit and explicit.mode:
            return explicit.mode
        if self.family(label) == PERSON:
            return ResolutionMode.VERIFY_ALWAYS
        return ResolutionMode.AUTO_KEY

    def identity_attrs(self, label: str) -> tuple[str, ...]:
        """Return the attribute names that identify an entity of this type."""
        explicit = self._types.get(normalize_label(label))
        return explicit.identity_attrs if explicit else ()

    def strip_suffixes(self, label: str) -> bool:
        """Return whether candidates are also looked up by the stripped key.

        The stripped key drops a leading article and one trailing legal suffix,
        so "Apple Inc" becomes "apple". It is off unless the type enables it, and
        a pair found only through it is verified, never merged automatically.
        """
        explicit = self._types.get(normalize_label(label))
        return bool(explicit and explicit.strip_suffixes)

    def relation(
        self, a: str, b: str, relations: TypeRelations | None = None
    ) -> TypeRel:
        """Return the relation between two type labels."""
        a_c, b_c = normalize_label(a), normalize_label(b)
        if a_c == b_c:
            return TypeRel.SAME
        if _vague(a_c) or _vague(b_c):
            return TypeRel.NESTED
        if relations is not None and relations.related(a_c, b_c):
            return TypeRel.NESTED
        fam = self.family(a)
        if fam is not None and fam == self.family(b):
            return TypeRel.FAMILY
        return TypeRel.DISTINCT


def identity_of(
    policy: ResolutionPolicy, label: str, attrs: dict[str, Any]
) -> dict[str, str]:
    """Return the normalized values of the identity attributes of ``label``."""
    out: dict[str, str] = {}
    for name in policy.identity_attrs(label):
        v = attrs.get(name)
        if v not in (None, ""):
            out[name] = strict_key(str(v))
    return out


def identity_conflict(a: dict[str, str], b: dict[str, str]) -> bool:
    """Return whether both sides carry the same identity attribute differently."""
    return any(a[k] != b[k] for k in a.keys() & b.keys())


def identity_agree(a: dict[str, str], b: dict[str, str]) -> bool:
    """Return whether an identity attribute is present on both sides and equal."""
    return any(a[k] == b[k] for k in a.keys() & b.keys())
