"""Tests for the value types of the vector store."""

import pytest

from ontogr_mem.data_models import FactStatus
from ontogr_mem.vectorstore import Validity


class TestValidity:
    """Behavior of `Validity`."""

    def test_the_status_is_the_canonical_fact_status(self) -> None:
        """A status is the shared enum member, not a loose string."""
        assert Validity("active", 0, 0).status is FactStatus.ACTIVE

    def test_an_unknown_status_is_refused(self) -> None:
        """A typo cannot reach the query that filters on the status."""
        with pytest.raises(ValueError):
            Validity("activ", 0, 0)
