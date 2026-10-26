"""Deterministic id builders."""

import hashlib
from uuid import UUID, uuid5


_SEPARATOR = "\x1f"


def stable_id(*parts: str, length: int = 32) -> str:
    """Build a deterministic hex id from parts.

    Args:
        *parts: Strings that identify the object. Order matters.
        length: Number of hex characters to keep.

    Returns:
        The first `length` hex characters of the SHA-256 of the parts joined
        with the unit separator character.

    Raises:
        ValueError: If a part contains the unit separator, which would let two
            different part lists produce the same id.
    """
    if any(_SEPARATOR in part for part in parts):
        msg = "id parts must not contain the unit separator character"
        raise ValueError(msg)
    digest = hashlib.sha256(_SEPARATOR.join(parts).encode()).hexdigest()
    return digest[:length]


def uuid5_id(namespace: UUID, *parts: str) -> str:
    """Build a deterministic UUID string from parts.

    Args:
        namespace: The UUID namespace.
        *parts: Strings that identify the object, joined with a bar.

    Returns:
        The UUIDv5 of the joined parts under the namespace.
    """
    return str(uuid5(namespace, "|".join(parts)))
