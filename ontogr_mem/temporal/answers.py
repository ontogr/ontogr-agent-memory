"""The answers the model returns and how they are read.

Ids must be exactly ``P1..Pn`` as asked. An invented or out-of-range id is
counted and dropped, so the model cannot answer for a pair it was not given. An
answer that does not fill the shape the prompt asked for is missing: it is
dropped, and the pair is asked again later.
"""

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TypedDict

from ontogr_mem.temporal.verdicts import VerdictKind


class RawAnswer(TypedDict, total=False):
    """One answer as the model sends it.

    Attributes:
        id: The pair the answer is for, ``P1``..``Pn``.
        verdict: The answer, one of the verdict names.
        confidence: How sure the model was, from 0 to 1.
        kind: Kind of disagreement the model named.
        rule: Short rule or reason label.
        reason: Plain explanation.
        quote_a: Words of the first statement that support the answer.
        quote_b: Words of the second statement that support the answer.
    """

    id: str
    verdict: str
    confidence: float
    kind: str
    rule: str
    reason: str
    quote_a: str
    quote_b: str


class RawReply(TypedDict):
    """The whole reply: one answer per pair.

    Attributes:
        pairs: The answers, one per pair the judge asked about.
    """

    pairs: list[RawAnswer]


@dataclass(frozen=True, slots=True)
class Answer:
    """One answer for one pair.

    Attributes:
        verdict: The answer the model gave.
        confidence: How sure the model was, from 0 to 1.
        kind: Kind of disagreement the model named.
        rule: Short rule or reason label.
        reason: Plain explanation.
        quote_a: Words of the first statement that support the answer.
        quote_b: Words of the second statement that support the answer.
    """

    verdict: VerdictKind
    confidence: float
    kind: str
    rule: str
    reason: str
    quote_a: str
    quote_b: str


def _loads(text: str) -> object:
    """Parse the model's JSON, tolerating a code fence or words around it.

    The Markdown fence is stripped first. If that is still not JSON, the outermost
    ``{...}`` span is tried. Anything else raises.
    """
    body = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        start, end = body.find("{"), body.rfind("}")
        if start < 0 or end <= start:
            raise
        return json.loads(body[start : end + 1])


def _pairs(data: object) -> list[object] | None:
    """Return the answers of a reply, or ``None`` when the reply has no shape."""
    if not isinstance(data, dict):
        return None
    pairs = data.get("pairs")
    if not isinstance(pairs, list):
        return None
    return pairs


def _text(value: object) -> str:
    """Return a text field of the reply, empty when it is not text."""
    return value if isinstance(value, str) else ""


def _raw_answer(item: Mapping[str, object]) -> RawAnswer | None:
    """Return the answer as the shape the prompt asked for, or ``None``.

    The verdict and the confidence are required: an answer without them says
    nothing about the pair, so it is a missing answer.
    """
    verdict = item.get("verdict")
    confidence = item.get("confidence")
    if not isinstance(verdict, str):
        return None
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return None
    return RawAnswer(
        id=_text(item.get("id")),
        verdict=verdict,
        confidence=float(confidence),
        kind=_text(item.get("kind")),
        rule=_text(item.get("rule")),
        reason=_text(item.get("reason")),
        quote_a=_text(item.get("quote_a")),
        quote_b=_text(item.get("quote_b")),
    )


def parse_answers(text: str, count: int) -> tuple[dict[int, Answer], int, bool]:
    """Return the answers by 0-based pair index, the invalid-id count and parse ok.

    Ids must be exactly ``P1..Pn`` as asked. An invented or out-of-range id is
    counted and dropped, so the model cannot answer for a pair it was not given.
    An answer that does not fill the shape the prompt asked for is dropped
    without a verdict.
    """
    try:
        data = _loads(text)
    except ValueError:
        return {}, 0, False
    items = _pairs(data)
    if items is None:
        return {}, 0, False
    answers: dict[int, Answer] = {}
    invalid = 0
    for item in items:
        if not isinstance(item, dict):
            invalid += 1
            continue
        raw_id = item.get("id")
        if not isinstance(raw_id, str):
            invalid += 1
            continue
        match = re.fullmatch(r"P(\d+)", raw_id)
        index = int(match.group(1)) - 1 if match else -1
        if not 0 <= index < count:
            invalid += 1
            continue
        raw = _raw_answer(item)
        if raw is None:
            continue
        try:
            verdict = VerdictKind(raw["verdict"].strip().lower())
        except ValueError:
            continue
        answers[index] = Answer(
            verdict=verdict,
            confidence=min(1.0, max(0.0, raw["confidence"])),
            kind=raw["kind"],
            rule=raw["rule"],
            reason=raw["reason"],
            quote_a=raw["quote_a"],
            quote_b=raw["quote_b"],
        )
    return answers, invalid, True


def unswap(answer: Answer) -> Answer:
    """Put the quotes back in their own order after a swapped pass."""
    return replace(answer, quote_a=answer.quote_b, quote_b=answer.quote_a)
