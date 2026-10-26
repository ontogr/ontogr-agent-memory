"""Trace id generation and validation helpers."""

import re
from uuid import uuid4


TRACE_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,64}")


def new_trace_id() -> str:
    """Return a fresh 32-character hexadecimal trace id."""
    return uuid4().hex


def accept_trace_id(candidate: str | None) -> str:
    """Keep a valid caller trace id or replace it with a fresh id."""
    if candidate is not None and TRACE_ID_PATTERN.fullmatch(candidate):
        return candidate
    return new_trace_id()
