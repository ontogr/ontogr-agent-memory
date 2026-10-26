"""Tests for tenant id validation."""

import pytest

from ontogr_mem.data_models import InvalidTenantError, validate_tenant_id
from ontogr_mem.errors import AgentMemoryError


class TestValidateTenantId:
    """Behavior of validate_tenant_id."""

    @pytest.mark.parametrize("value", ["a", "A-b_9", "x" * 60])
    def test_returns_valid_ids_unchanged(self, value):
        """Letters, digits, dash and underscore up to 60 characters pass."""
        assert validate_tenant_id(value) == value

    @pytest.mark.parametrize("value", ["", "x" * 61, "a.b", "a b", "a/b", "é"])
    def test_rejects_invalid_ids(self, value):
        """Empty, too long and out-of-alphabet ids raise."""
        with pytest.raises(InvalidTenantError) as excinfo:
            validate_tenant_id(value)

        assert isinstance(excinfo.value, AgentMemoryError)
        assert isinstance(excinfo.value, ValueError)
