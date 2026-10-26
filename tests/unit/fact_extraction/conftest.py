"""Shared fixtures for fact extraction tests."""

import pytest

from ontogr_mem.data_models import TypeCatalog
from tests.support.facts import fact_catalog


@pytest.fixture
def catalog() -> TypeCatalog:
    """Return a catalog with two relation types and three entity types."""
    return fact_catalog()
