"""Text normalization shared by components that compare names."""

import re
import unicodedata


def normalize_label(label: str) -> str:
    """Return a case-folded label key without whitespace or underscores."""
    return re.sub(r"[\s_]+", "", label.strip().casefold())


def normalize_name(name: str) -> str:
    """Return a canonical name key with NFC, collapsed whitespace and case-folding."""
    normalized = unicodedata.normalize("NFC", name)
    return " ".join(normalized.split()).casefold()


def same_name(first: str, second: str) -> bool:
    """Return whether two names have the same normalized key."""
    return normalize_name(first) == normalize_name(second)
