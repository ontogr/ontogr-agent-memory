"""Identifier rules, reserved names and label normalization."""

import re

from ontogr_mem.data_models.text import normalize_label
from ontogr_mem.ontology.errors import InvalidChange


MAX_NAME_LEN = 64
_IDENT = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")

RESERVED_LABELS = frozenset(
    {"entity", "unknown", "episode", "chunk", "document", "fact", "statement"}
)
"""Structural labels the stores use, in lower case.

Users cannot declare a type with one of these labels. Only the catalog itself owns
``Unknown`` (entity type id 0).
"""

RESERVED_ATTRIBUTES = frozenset(
    {
        "uuid",
        "name",
        "labels",
        "created_at",
        "name_embedding",
        "summary",
        "attributes",
        "id",
        "type_id",
        "label",
        "entity_id",
        "episode_id",
        "embedding",
        "valid_from",
        "valid_to",
        "recorded_at",
        "expired_at",
        "description",
    }
)
"""Attribute names that the graph rows or this repo already use.

The union of Graphiti's ``EntityNode`` field names, the names FalkorDB's SDK keeps for
itself, and this repo's id and time fields. An attribute may not reuse any of them.
"""


def normalize(text: str) -> str:
    """Fold text for matching: lower case, with spaces and underscores removed."""
    return normalize_label(text)


def check_identifier(name: str, *, kind: str) -> None:
    """Raise ``InvalidChange`` unless ``name`` is a legal type or attribute name.

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
