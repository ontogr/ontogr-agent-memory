"""Vector helpers in plain Python; the dimensions are small."""

import math
from collections.abc import Sequence

from ontogr_mem.vectorstore.errors import InvalidVectorError
from ontogr_mem.vectorstore.models import FailReason


def norm(vec: Sequence[float]) -> float:
    """Return the Euclidean norm."""
    return math.sqrt(math.fsum(x * x for x in vec))


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    """Return the dot product."""
    return math.fsum(x * y for x, y in zip(a, b, strict=True))


def classify_vector(
    raw: Sequence[float], dim: int
) -> tuple[tuple[float, ...] | None, FailReason | None, str]:
    """Validate a provider vector and scale it to unit length.

    Returns:
        `(unit_vector, None, "")` on success, else `(None, reason, detail)`.
    """
    if len(raw) == 0:
        return None, FailReason.EMPTY_VECTOR, "provider returned an empty vector"
    if len(raw) != dim:
        return None, FailReason.BAD_DIMENSION, f"got {len(raw)}, expected {dim}"
    if not all(math.isfinite(x) for x in raw):
        return None, FailReason.NON_FINITE, "NaN or infinity in vector"
    length = norm(raw)
    if length == 0.0:
        return None, FailReason.ZERO_NORM, "zero-norm vector"
    return tuple(float(x) / length for x in raw), None, ""


def to_unit(vec: Sequence[float], dim: int) -> tuple[float, ...]:
    """Scale a query vector to unit length.

    Raises:
        InvalidVectorError: If the vector cannot be used.
    """
    unit, reason, detail = classify_vector(vec, dim)
    if unit is None:
        msg = f"{reason}: {detail}" if reason else detail
        raise InvalidVectorError(msg)
    return unit


def mean_unit(vectors: Sequence[Sequence[float]]) -> tuple[float, ...] | None:
    """Average unit vectors and rescale; None when they cancel out."""
    total = [math.fsum(v[i] for v in vectors) for i in range(len(vectors[0]))]
    length = norm(total)
    if length == 0.0:
        return None
    return tuple(x / length for x in total)


def to_score(cosine: float) -> float:
    """Map a cosine similarity in [-1, 1] to a score in [0, 1]."""
    return min(1.0, max(0.0, (1.0 + cosine) / 2.0))
