"""Strict parsing of fact-pass replies.

`parse_facts` collects every problem in a reply before raising `SchemaError`, so
the repair attempt can show all of them. It checks shape and enum values only;
refs, quotes and times are checked in binding.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, TypeVar

from ontogr_mem.data_models.facts import FactKind, Modality, Polarity, Precision
from ontogr_mem.llm.json_reply import SchemaError, load_json


E = TypeVar("E", bound=StrEnum)


@dataclass(frozen=True, slots=True)
class RawFact:
    """A fact as the model wrote it, before binding.

    Times are strings and refs are unchecked ids.
    """

    text: str
    entity_refs: tuple[str, ...]
    relation_type_id: int | None
    fact_kind: FactKind
    polarity: Polarity
    modality: Modality
    evidence_quote: str
    relation_label: str | None = None
    valid_from: str | None = None
    valid_to: str | None = None
    precision: Precision | None = None
    time_expression: str | None = None
    confidence: float | None = None


def _str(item: dict[str, Any], key: str, where: str, errors: list[str]) -> str:
    """Read a required string field."""
    value = item.get(key)
    if not isinstance(value, str):
        errors.append(f"{where}.{key} must be a string")
        return ""
    return value


def _opt_str(
    item: dict[str, Any], key: str, where: str, errors: list[str]
) -> str | None:
    """Read a nullable string field."""
    value = item.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        errors.append(f"{where}.{key} must be a string or null")
        return None
    return value


def _enum(
    item: dict[str, Any], key: str, cls: type[E], where: str, errors: list[str]
) -> E | None:
    """Read an enum field, accepting only its values."""
    value = item.get(key)
    try:
        if not isinstance(value, str):
            raise ValueError(value)
        return cls(value)
    except ValueError:
        allowed = " | ".join(m.value for m in cls)
        errors.append(f"{where}.{key} must be one of: {allowed}")
        return None


def _refs(item: dict[str, Any], where: str, errors: list[str]) -> tuple[str, ...]:
    """Read the entity refs as a tuple of strings."""
    value = item.get("entity_refs")
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        errors.append(f"{where}.entity_refs must be an array of strings")
        return ()
    return tuple(value)


def _type_id(item: dict[str, Any], where: str, errors: list[str]) -> int | None:
    """Read the dense relation type id."""
    value = item.get("relation_type_id")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        errors.append(f"{where}.relation_type_id must be an integer or null")
        return None
    return value


def _confidence(item: dict[str, Any], where: str, errors: list[str]) -> float | None:
    """Read the model's confidence; out-of-range values are schema errors."""
    value = item.get("confidence")
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not 0 <= value <= 1
    ):
        errors.append(f"{where}.confidence must be a number in 0..1 or null")
        return None
    return float(value)


def parse_facts(text: str) -> list[RawFact]:
    """Parse and validate a reply.

    Args:
        text: The model's reply.

    Returns:
        The facts that passed every check, in reply order.

    Raises:
        SchemaError: With every problem found in the reply.
    """
    data = load_json(text)
    if not isinstance(data, dict):
        raise SchemaError(["top level must be a JSON object"])
    items = data.get("facts")
    if not isinstance(items, list):
        raise SchemaError(["missing required array 'facts'"])

    errors: list[str] = []
    out: list[RawFact] = []
    for i, item in enumerate(items):
        where = f"facts[{i}]"
        if not isinstance(item, dict):
            errors.append(f"{where} must be an object")
            continue
        # Error count before this item, to tell whether this item added any.
        before = len(errors)
        kind = _enum(item, "fact_kind", FactKind, where, errors)
        polarity = _enum(item, "polarity", Polarity, where, errors)
        modality = _enum(item, "modality", Modality, where, errors)
        precision = (
            None
            if item.get("precision") is None
            else _enum(item, "precision", Precision, where, errors)
        )
        fact_text = _str(item, "text", where, errors)
        quote = _str(item, "evidence_quote", where, errors)
        label = _opt_str(item, "relation_label", where, errors)
        refs = _refs(item, where, errors)
        type_id = _type_id(item, where, errors)
        confidence = _confidence(item, where, errors)
        raw_from = _opt_str(item, "valid_from", where, errors)
        raw_to = _opt_str(item, "valid_to", where, errors)
        expression = _opt_str(item, "time_expression", where, errors)
        if len(errors) > before or kind is None or polarity is None or modality is None:
            continue
        out.append(
            RawFact(
                text=fact_text,
                entity_refs=refs,
                relation_type_id=type_id,
                relation_label=label,
                fact_kind=kind,
                polarity=polarity,
                modality=modality,
                evidence_quote=quote,
                valid_from=raw_from,
                valid_to=raw_to,
                precision=precision,
                time_expression=expression,
                confidence=confidence,
            )
        )
    if errors:
        raise SchemaError(errors)
    return out
