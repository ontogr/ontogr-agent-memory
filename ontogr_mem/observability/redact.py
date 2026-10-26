"""Redact credential names and values before spans reach a tracer."""

import re
from collections.abc import Mapping


MAX_ATTRIBUTE_CHARS = 8000
SUMMARY_CHARS = 200
TRUNCATED_MARKER = "...[truncated]"

_SECRET_WORDS = frozenset(
    {
        "key",
        "keys",
        "apikey",
        "token",
        "secret",
        "password",
        "passwd",
        "authorization",
        "cookie",
    }
)
_WORD_BREAKS = re.compile(r"[\s._-]+|(?<=[a-z0-9])(?=[A-Z])")
_BEARER_TOKEN = re.compile(r"(\bBearer\s+)\S{3,}", re.IGNORECASE)
_MINTED_KEY = re.compile(r"mk_[A-Za-z0-9_-]+")
_URL_USERINFO = re.compile(r"([A-Za-z][A-Za-z0-9+.-]*://)[^/@\s]+@")


def is_secret_name(name: str) -> bool:
    """Return whether an attribute name contains a credential word."""
    words = _WORD_BREAKS.split(name)
    # Token counts are usage data, not credentials.
    return any(
        word.casefold() in _SECRET_WORDS
        and not (
            word.casefold() == "token"
            and index + 1 < len(words)
            and words[index + 1].casefold() == "count"
        )
        for index, word in enumerate(words)
    )


def redact_text(value: str) -> str:
    """Hide bearer tokens, minted keys and URL userinfo in text."""
    value = _BEARER_TOKEN.sub(r"\1***", value)
    value = _MINTED_KEY.sub("mk_***", value)
    return _URL_USERINFO.sub(r"\1***@", value)


def cap(value: str, limit: int = MAX_ATTRIBUTE_CHARS) -> str:
    """Limit a string to ``limit`` characters, including its truncation marker."""
    if len(value) <= limit:
        return value
    marker = TRUNCATED_MARKER[:limit]
    return value[: limit - len(marker)] + marker


def redact_value(name: str, value: object) -> object:
    """Redact one attribute value while preserving safe scalar types."""
    if is_secret_name(name):
        return "***"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return cap(redact_text(value if isinstance(value, str) else str(value)))


def redact_attributes(attributes: Mapping[str, object]) -> dict[str, object]:
    """Return a detached mapping with each attribute value redacted and capped."""
    return {name: redact_value(name, value) for name, value in attributes.items()}


def summarize_exception(error: BaseException) -> str:
    """Return a redacted, capped one-line summary of an exception."""
    lines = str(error).splitlines()
    first_line = lines[0] if lines else ""
    if not first_line:
        return type(error).__name__
    return cap(redact_text(f"{type(error).__name__}: {first_line}"), SUMMARY_CHARS)
