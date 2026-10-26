"""Safe query tokens shared by keyword search and graph retrieval."""

import re


_TERM = re.compile(r"\w+(?:'\w+)*")


def terms(text: str) -> list[str]:
    """Return the first 32 distinct lowercase tokens, keeping internal apostrophes."""
    return list(dict.fromkeys(tokens(text)))[:32]


def tokens(text: str) -> list[str]:
    """Tokenize a document without dropping repetitions or truncating its text."""
    return _TERM.findall(text.lower())
