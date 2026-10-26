"""Property names reserved by graph-store internals."""

from collections.abc import Iterable, Mapping
from typing import Any

from ontogr_mem.graphstore.errors import ValidationError


TENANT = "_tenant"
LOCK = "_lock"
RESERVED = frozenset({TENANT, LOCK})


def reject_reserved(props: Iterable[str], owner: str) -> None:
    """Raise when caller properties use a reserved name.

    Raises:
        ValidationError: If `props` contains a reserved name.
    """
    clash = RESERVED.intersection(props)
    if clash:
        msg = f"{owner}: property {sorted(clash)[0]!r} is reserved"
        raise ValidationError(msg)


def strip_reserved(props: Mapping[str, Any]) -> dict[str, Any]:
    """Return properties without the names reserved by the graph store."""
    return {key: value for key, value in props.items() if key not in RESERVED}
