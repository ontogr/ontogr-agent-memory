"""Read a JSON reply from the model.

Small models wrap their answer in a Markdown fence and sometimes put a line of
prose around it. The loader strips the fence and falls back to the outermost
braces of the reply.
"""

import json
import re
from typing import Any


class SchemaError(ValueError):
    """The reply does not match the schema.

    Attributes:
        errors: Every problem found, to feed back to the model.
    """

    def __init__(self, errors: list[str]) -> None:
        """Keep the error list."""
        super().__init__("; ".join(errors))
        self.errors = errors


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def load_json(text: str) -> Any:
    """Parse JSON, falling back to the outermost braces of the reply.

    The Markdown fence is stripped whatever its case, so a reply fenced with
    ```JSON reads like one fenced with ```json.

    Args:
        text: The model's reply.

    Returns:
        The parsed JSON.

    Raises:
        SchemaError: If no JSON object can be read.
    """
    stripped = _FENCE.sub("", text.strip())
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as first:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start == -1 or end <= start:
            raise SchemaError([f"not valid JSON: {first.msg}"]) from first
        try:
            return json.loads(stripped[start : end + 1])
        except json.JSONDecodeError as second:
            raise SchemaError([f"not valid JSON: {second.msg}"]) from second
