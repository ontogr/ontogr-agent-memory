"""Which fact versions a reader sees, and which a writer can still retire.

Two rules live here, and they are not the same rule.

``in_current_view`` is the reader's rule. The current view is what the system
believes now. An overridden version stays in it, flagged, because a person or a
trust gap preferred another fact over it. A corrected version is out of it,
because a correction judged the statement wrong, and a withdrawn version is a
tombstone.

``is_withdrawable`` is the writer's rule. A version the writer still holds and
may retire with a tombstone. A corrected version is out of the reader's view,
but the writer still holds it, so the writer can retire it when its evidence
goes away.
"""

from ontogr_mem.data_models.fact_versions import Fact, FactStatus


_OUT_OF_VIEW = frozenset({FactStatus.WITHDRAWN, FactStatus.CORRECTED})
"""The statuses a reader does not see."""


def in_current_view(status: FactStatus) -> bool:
    """Return whether a reader shows a fact version of this status."""
    return status not in _OUT_OF_VIEW


def is_withdrawable(fact: Fact) -> bool:
    """Return whether the writer still holds this fact version and can retire it.

    A version a later version replaced, or one that is already a tombstone, is
    spent. A corrected version is out of the reader's view, but the writer
    still holds it.

    Args:
        fact: The fact version.

    Returns:
        True when the writer may still replace it with a tombstone.
    """
    return fact.expired_at is None and fact.status is not FactStatus.WITHDRAWN
