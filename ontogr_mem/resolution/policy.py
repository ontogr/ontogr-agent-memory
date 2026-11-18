"""Per-label resolution overrides and identity-attribute helpers."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from ontogr_mem.data_models.text import normalize_label
from ontogr_mem.ontology.links import TypeLinks
from ontogr_mem.resolution.keys import strict_key


class ResolutionMode(StrEnum):
    """How a key hit on a type is treated."""

    AUTO_KEY = "auto_key"
    VERIFY_ALWAYS = "verify_always"


@dataclass(frozen=True, slots=True)
class TypePolicy:
    """Overrides for one type label."""

    mode: ResolutionMode | None = None
    strip_suffixes: bool = False


class ResolutionPolicy:
    """Per-label overrides of how a key hit is treated."""

    def __init__(
        self,
        types: Mapping[str, TypePolicy] | None = None,
    ) -> None:
        """Create the table; ``types`` is keyed by label."""
        self._types = {normalize_label(k): v for k, v in (types or {}).items()}

    def mode(self, label: str) -> ResolutionMode:
        """Return how a key hit on this type is treated.

        An explicit per-type setting wins. Every other type is keyed on its name.
        """
        explicit = self._types.get(normalize_label(label))
        if explicit and explicit.mode:
            return explicit.mode
        return ResolutionMode.AUTO_KEY

    def strip_suffixes(self, label: str) -> bool:
        """Return whether candidates are also looked up by the stripped key.

        The stripped key drops a leading article and one trailing legal suffix,
        so "Apple Inc" becomes "apple". It is off unless the type enables it, and
        a pair found only through it is verified, never merged automatically.
        """
        explicit = self._types.get(normalize_label(label))
        return bool(explicit and explicit.strip_suffixes)


def identity_of(
    links: TypeLinks, type_id: int, attrs: Mapping[str, Any]
) -> dict[str, str]:
    """Return the normalized values of a type's identity attributes."""
    out: dict[str, str] = {}
    for name in links.identity_names(type_id):
        value = attrs.get(name)
        if value not in (None, ""):
            out[name] = strict_key(str(value))
    return out


def identity_conflict(a: dict[str, str], b: dict[str, str]) -> bool:
    """Return whether both sides carry the same identity attribute differently."""
    return any(a[k] != b[k] for k in a.keys() & b.keys())


def identity_agree(a: dict[str, str], b: dict[str, str]) -> bool:
    """Return whether an identity attribute is present on both sides and equal."""
    return any(a[k] == b[k] for k in a.keys() & b.keys())
