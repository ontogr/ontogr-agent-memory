"""Tests for identifier and tenant validation."""

import pytest

from ontogr_mem.graphstore.errors import ValidationError
from ontogr_mem.graphstore.ident import ident, quote, tenant_id


class TestIdent:
    """Behavior of ident, quote and tenant_id."""

    def test_quotes_plain_names(self):
        """Valid names come back backtick-quoted."""
        assert quote("Entity") == "`Entity`"
        assert quote("__Meta") == "`__Meta`"

    @pytest.mark.parametrize(
        "name", ["", "1abc", "a-b", "a b", "a`b", "x" * 65, "a;DROP", "é"]
    )
    def test_rejects_unsafe_names(self, name):
        """Names that could break out of a query are refused."""
        with pytest.raises(ValidationError):
            ident(name)

    def test_accepts_the_longest_name(self):
        """A 64 character name is the limit."""
        assert ident("x" * 64) == "x" * 64

    @pytest.mark.parametrize("tenant", ["", "x" * 61, "has space", "a.b", "é"])
    def test_rejects_invalid_tenants(self, tenant):
        """Tenant ids follow the 60 character rule."""
        with pytest.raises(ValidationError):
            tenant_id(tenant)

    def test_accepts_valid_tenant(self):
        """Letters, digits, dash and underscore pass."""
        assert tenant_id("acme-1_x") == "acme-1_x"
