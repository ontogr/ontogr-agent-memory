"""Lookup keys. Neither key is an identity; both are index values."""

import re
import unicodedata
from collections.abc import Iterable

from ontogr_mem.data_models.entities import Entity


_SPACE = re.compile(r"\s+")
_DIGITS = re.compile(r"\d+")
DEFAULT_LEGAL_SUFFIXES = frozenset(
    {
        "inc",
        "incorporated",
        "ltd",
        "limited",
        "llc",
        "gmbh",
        "corp",
        "corporation",
        "co",
        "plc",
        "sa",
        "ag",
        "bv",
        "oy",
        "ab",
        "srl",
        "pty",
    }
)
DEFAULT_ARTICLES = frozenset({"the", "a", "an"})


def entity_text(entity: Entity) -> str:
    """Return the text used to embed an entity."""
    description = entity.summary or (
        entity.descriptions[0] if entity.descriptions else ""
    )
    return f"{entity.canonical_name} | {entity.type_label} | {description[:200]}"


def strict_key(text: str) -> str:
    """Normalize a name for exact lookup.

    Applies NFKC and casefolding and collapses whitespace. Punctuation and
    diacritics are kept. This is the only key an automatic match may use.

    Args:
        text: Name to normalize.

    Returns:
        The normalized name.
    """
    return _SPACE.sub(" ", unicodedata.normalize("NFKC", text).casefold()).strip()


def _drop_marks_and_punct(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    kept = []
    for ch in decomposed:
        cat = unicodedata.category(ch)
        if cat.startswith("M"):
            continue
        kept.append(" " if cat[0] in "PSZC" else ch)
    return _SPACE.sub(" ", "".join(kept)).strip()


def loose_key(text: str) -> str:
    """Normalize a name for blocking.

    The strict key without diacritics and punctuation. A name with no letters or
    digits left keeps its strict key, so the key is never empty for a non-empty
    name.

    Args:
        text: Name to normalize.

    Returns:
        The blocking key.
    """
    strict = strict_key(text)
    loose = _drop_marks_and_punct(strict)
    return loose or strict


def stripped_key(
    text: str,
    *,
    suffixes: Iterable[str] = DEFAULT_LEGAL_SUFFIXES,
    articles: Iterable[str] = DEFAULT_ARTICLES,
) -> str:
    """Drop a leading article and one trailing legal suffix from the loose key.

    Pairs found only through this key are verified, never merged automatically.

    Args:
        text: Name to normalize.
        suffixes: Legal suffixes to drop.
        articles: Articles to drop.

    Returns:
        The stripped blocking key.
    """
    words = loose_key(text).split(" ")
    if len(words) > 1 and words[0] in set(articles):
        words = words[1:]
    if len(words) > 1 and words[-1] in set(suffixes):
        words = words[:-1]
    return " ".join(words)


def differ_only_in_digits(a: str, b: str) -> bool:
    """Return whether two names differ only in their digit runs.

    Embeddings conflate "Q3 2023" and "Q3 2024", so a digit-only difference is a
    hard veto on an automatic match.

    Args:
        a: First name.
        b: Second name.

    Returns:
        True when the names are equal once digit runs are masked but differ.
    """
    a_s, b_s = strict_key(a), strict_key(b)
    if a_s == b_s:
        return False
    return _DIGITS.sub("#", a_s) == _DIGITS.sub("#", b_s)
