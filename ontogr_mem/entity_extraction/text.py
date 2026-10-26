"""Accessors for the text and language of an episode."""

import re

from ontogr_mem.data_models import Episode, Span


def text_of(episode: Episode) -> str:
    """Return the episode text, or an empty string for structured content."""
    return episode.content if isinstance(episode.content, str) else ""


def language_of(episode: Episode) -> str:
    """Return `metadata["language"]`, or "en" when it is not set."""
    return str(episode.metadata.get("language", "en"))


def find_spans(text: str, needle: str) -> list[Span]:
    """Find every case-insensitive, whitespace-flexible occurrence of `needle`."""
    words = needle.split()
    if not words:
        return []
    pattern = r"\s+".join(re.escape(w) for w in words)
    return [m.span() for m in re.finditer(pattern, text, re.IGNORECASE)]
