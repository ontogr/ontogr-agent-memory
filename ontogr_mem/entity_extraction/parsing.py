"""Strict parsing of model replies, with a lenient JSON fallback."""

from dataclasses import dataclass
from typing import Any

from ontogr_mem.llm.json_reply import SchemaError, load_json


@dataclass(frozen=True, slots=True)
class RawEntity:
    """An entity the model proposed, as written."""

    name: str
    type_id: int
    type_label: str | None = None
    description: str = ""
    confidence: float | None = None


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _text(item: dict[str, Any], key: str, where: str, errors: list[str]) -> str:
    value = item.get(key)
    if not isinstance(value, str):
        errors.append(f"{where}.{key} must be a string")
        return ""
    return value


def _optional_text(
    item: dict[str, Any], key: str, where: str, errors: list[str]
) -> str | None:
    value = item.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        errors.append(f"{where}.{key} must be a string or null")
        return None
    return value


def parse_entities(text: str) -> list[RawEntity]:
    """Parse and check an entity reply.

    Raises:
        SchemaError: With every problem found in the reply.
    """
    data = load_json(text)
    if not isinstance(data, dict):
        raise SchemaError(["top level must be a JSON object"])
    errors: list[str] = []
    items = data.get("entities")
    if items is None:
        errors.append("missing required key 'entities'")
        items = []
    elif not isinstance(items, list):
        errors.append("'entities' must be an array")
        items = []
    out: list[RawEntity] = []
    for i, item in enumerate(items):
        where = f"entities[{i}]"
        if not isinstance(item, dict):
            errors.append(f"{where} must be an object")
            continue
        if not _is_int(item.get("type_id")):
            errors.append(f"{where}.type_id must be an integer")
        confidence = item.get("confidence")
        if confidence is not None and (
            isinstance(confidence, bool)
            or not isinstance(confidence, int | float)
            or not 0 <= confidence <= 1
        ):
            errors.append(f"{where}.confidence must be a number in 0..1 or null")
            confidence = None
        out.append(
            RawEntity(
                name=_text(item, "name", where, errors),
                type_id=item["type_id"] if _is_int(item.get("type_id")) else -999,
                type_label=_optional_text(item, "type_label", where, errors),
                description=_optional_text(item, "description", where, errors) or "",
                confidence=float(confidence) if confidence is not None else None,
            )
        )
    if errors:
        raise SchemaError(errors)
    return out
