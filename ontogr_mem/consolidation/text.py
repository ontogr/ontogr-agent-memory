"""Small text helpers shared by the stages."""

import hashlib
import json
import re


_SENTENCE_END = re.compile(r"[.!?](?=\s|$)")


def truncate_at_sentence(text: str, limit: int) -> str:
    """Cut ``text`` to at most ``limit`` characters, at a sentence end if possible."""
    text = text.strip()
    if len(text) <= limit:
        return text
    head = text[:limit]
    ends = [m.end() for m in _SENTENCE_END.finditer(head)]
    return head[: ends[-1]] if ends else head.rstrip()


def quote(value: object) -> str:
    """JSON-quote a value so untrusted text cannot close a prompt block around it."""
    return json.dumps(value, ensure_ascii=False)


def short_hash(*parts: object) -> str:
    """Return a stable 12-character id for ``parts``. It is not a security token."""
    raw = json.dumps(parts, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]
