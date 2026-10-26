"""Fact text identity and quote location."""

import re
import unicodedata

from ontogr_mem.data_models import Episode, Span
from ontogr_mem.data_models.text import normalize_name


def text_of(episode: Episode) -> str:
    """Return the episode text in NFC, or an empty string for structured content."""
    content = episode.content if isinstance(episode.content, str) else ""
    return unicodedata.normalize("NFC", content)


def language_of(episode: Episode) -> str:
    """Return `metadata["language"]`, or "en" when it is not set."""
    return str(episode.metadata.get("language", "en"))


def normalize_text(text: str) -> str:
    """Return NFC, collapsed whitespace and case-folded: a fact's identity."""
    return normalize_name(text)


def locate(text: str, needle: str) -> Span | None:
    """Return the first span of `needle` in `text`, or None.

    Case and whitespace are ignored. `needle` is normalized to NFC; `text` must
    already be NFC so the offsets index it.
    """
    spans = _find_spans(text, unicodedata.normalize("NFC", needle))
    return spans[0] if spans else None


def _find_spans(text: str, needle: str) -> list[Span]:
    """Return every case-insensitive, whitespace-flexible occurrence."""
    words = needle.split()
    if not words:
        return []
    pattern = r"\s+".join(re.escape(word) for word in words)
    return [match.span() for match in re.finditer(pattern, text, re.IGNORECASE)]
