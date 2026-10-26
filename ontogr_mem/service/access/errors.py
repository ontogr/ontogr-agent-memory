"""Errors for cryptographic secrets used by service access controls."""

from ontogr_mem.errors import AgentMemoryError


class AccessError(AgentMemoryError):
    """Base error for service access controls."""


class InvalidKeySecretError(AccessError, ValueError):
    """An HMAC signing secret is too short for service confirmation tokens."""
