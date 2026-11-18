"""Identifier rules, reserved names and label normalization."""

import re

from ontogr_mem.data_models.text import normalize_label
from ontogr_mem.ontology.errors import InvalidChange


MAX_NAME_LEN = 64
_IDENT = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")

RESERVED_LABELS = frozenset(
    {"entity", "unknown", "episode", "chunk", "fact", "statement"}
)
"""Structural labels the stores use, in lower case.

Users cannot declare a type with one of these labels. Only the catalog itself owns
``Unknown`` (entity type id 0).
"""

RESERVED_IDENTITY_NAMES = frozenset({"uri", "actor_id", "role"})
"""Attribute names the resolver reads directly, so types cannot declare them."""


def normalize(text: str) -> str:
    """Fold text for matching: lower case, with spaces and underscores removed."""
    return normalize_label(text)


def check_identifier(name: str, *, kind: str) -> None:
    """Raise ``InvalidChange`` unless ``name`` is a legal type name.

    A legal name starts with a letter, holds letters, digits and single underscores, and
    is at most ``MAX_NAME_LEN`` characters. ``kind`` only labels the error message.
    """
    if len(name) > MAX_NAME_LEN:
        raise InvalidChange(
            f"{kind} name longer than {MAX_NAME_LEN}", item=name[:20] + "..."
        )
    if not _IDENT.match(name) or "__" in name:
        raise InvalidChange(
            f"illegal {kind} name",
            item=name,
            hint="use letters, digits and single underscores, starting with a letter",
        )
