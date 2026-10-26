"""Tenant ids for tests."""

from uuid import uuid4


def new_tenant() -> str:
    """Return a fresh tenant id that passes tenant id validation."""
    return "t" + uuid4().hex[:16]
