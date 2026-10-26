"""Property conversion at the store boundary."""

import json
import math
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any
from uuid import UUID

from ontogr_mem.data_models import to_micros
from ontogr_mem.graphstore.errors import ValidationError
from ontogr_mem.graphstore.ident import ident


_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_PRIMITIVES = (str, int, float, bool)


def micros(value: datetime) -> int:
    """Convert a timezone-aware datetime to UTC epoch microseconds.

    Raises:
        ValidationError: If the datetime has no timezone.
    """
    try:
        return to_micros(value)
    except ValueError as exc:
        msg = "naive datetime rejected; attach a timezone"
        raise ValidationError(msg) from exc


def clean_string(value: str) -> str:
    """Strip NUL and control characters except tab, newline and carriage return."""
    return _CONTROL_RE.sub("", value)


def _scalar(key: str, value: Any) -> Any:
    if isinstance(value, datetime):
        return micros(value)
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, str):
        return clean_string(value)
    if isinstance(value, float) and not math.isfinite(value):
        msg = f"property {key!r}: non-finite float"
        raise ValidationError(msg)
    if isinstance(value, _PRIMITIVES):
        return value
    msg = f"property {key!r}: unsupported type {type(value).__name__}"
    raise ValidationError(msg)


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return micros(value)
    if isinstance(value, UUID):
        return str(value)
    msg = f"not JSON serialisable: {type(value).__name__}"
    raise TypeError(msg)


def clean_properties(props: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of `props` that the graph databases accept.

    `None` is omitted, datetimes become epoch microseconds, UUIDs become strings,
    strings lose control characters, a mapping becomes a JSON string under
    `<key>_json`, and a list holding anything but primitives is rejected.

    Raises:
        ValidationError: On an unsupported value or an unsafe key.
    """
    out: dict[str, Any] = {}
    for key, value in props.items():
        ident(key, "property key")
        if value is None:
            continue
        if isinstance(value, Mapping):
            try:
                encoded = json.dumps(
                    value, default=_json_default, sort_keys=True, allow_nan=False
                )
            except ValueError as exc:
                msg = f"property {key!r}: {exc}"
                raise ValidationError(msg) from exc
            out[f"{key}_json"] = encoded
        elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
            out[key] = [_scalar(key, item) for item in value]
        else:
            out[key] = _scalar(key, value)
    return out
