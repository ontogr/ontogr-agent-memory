"""Tenant id validation."""

import re
from typing import Final

from ontogr_mem.errors import AgentMemoryError


TENANT_ID_PATTERN: Final = re.compile(r"[A-Za-z0-9_-]{1,60}")


class InvalidTenantError(AgentMemoryError, ValueError):
    """A tenant id is empty, too long, or uses characters outside the alphabet."""


def validate_tenant_id(value: str) -> str:
    """Check a tenant id.

    A tenant id has 1 to 60 characters from letters, digits, dash and underscore.

    Args:
        value: The tenant id to check.

    Returns:
        The same id.

    Raises:
        InvalidTenantError: If the id does not match the format.
    """
    if not TENANT_ID_PATTERN.fullmatch(value):
        msg = (
            f"invalid tenant id {value!r}: use 1 to 60 letters, digits, "
            "dashes or underscores"
        )
        raise InvalidTenantError(msg)
    return value
