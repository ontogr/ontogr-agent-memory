"""Identifier validation for labels, relationship types and property keys.

Cypher cannot parameterise these, so every name passes one validator and is
backtick-quoted before interpolation.
"""

import re

from ontogr_mem.data_models import InvalidTenantError, validate_tenant_id
from ontogr_mem.graphstore.errors import ValidationError


_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")


def ident(name: str, kind: str = "identifier") -> str:
    """Return `name` if it is a safe identifier.

    Args:
        name: Label, relationship type or property key.
        kind: Word used in the error message.

    Returns:
        The same name.

    Raises:
        ValidationError: If `name` does not match `[A-Za-z_][A-Za-z0-9_]{0,63}`.
    """
    if not isinstance(name, str) or not _IDENT_RE.fullmatch(name):
        msg = f"invalid {kind}: {name!r}"
        raise ValidationError(msg)
    return name


def quote(name: str, kind: str = "identifier") -> str:
    """Validate `name` and return it backtick-quoted."""
    return f"`{ident(name, kind)}`"


def tenant_id(tenant: str) -> str:
    """Return `tenant` if it is a valid tenant id.

    Raises:
        ValidationError: If the id is empty, longer than 60 characters, or uses
            characters outside letters, digits, dash and underscore.
    """
    try:
        return validate_tenant_id(tenant)
    except (InvalidTenantError, TypeError) as exc:
        msg = f"invalid tenant id: {tenant!r}"
        raise ValidationError(msg) from exc
