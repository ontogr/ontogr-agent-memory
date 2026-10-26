"""Cover the service guard that verifies update reconciliation closed a fact."""

from datetime import UTC, datetime

import pytest

from ontogr_mem.data_models.fact_versions import FactStatus
from ontogr_mem.service.errors import InternalError
from ontogr_mem.service.writes import ensure_closed
from ontogr_mem.temporal.query import FactView


NOW = datetime(2025, 1, 1, tzinfo=UTC)


def _fact_view(status: FactStatus) -> FactView:
    """Build a minimal fact view for the closed-state guard."""
    return FactView(
        fact_id="fact-1",
        version=1,
        slot_key="slot-1",
        subject_id="entity-1",
        object_id="entity-2",
        relation_type_id=1,
        valid_from=NOW,
        valid_to=None,
        valid_from_precision=None,
        valid_to_precision=None,
        recorded_at=NOW,
        expired_at=None,
        status=status,
        evidence=(),
    )


class TestEnsureClosed:
    """A successful update must no longer expose its old fact as active."""

    def test_raises_when_the_old_fact_remains_active(self):
        """An unmatched stated end is an internal reconciliation failure."""
        with pytest.raises(InternalError, match="fact is still active"):
            ensure_closed(_fact_view(FactStatus.ACTIVE))

    def test_accepts_a_closed_fact(self):
        """The guard accepts an old fact after its temporal version closes."""
        assert ensure_closed(_fact_view(FactStatus.CLOSED)) is None
