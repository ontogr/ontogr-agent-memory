"""Tests for stable id helpers."""

from uuid import UUID

import pytest

from ontogr_mem.data_models import stable_id, uuid5_id


class TestStableId:
    """Behavior of stable_id."""

    def test_returns_fixed_hex_for_known_parts(self):
        """The id is the first 32 hex characters of the SHA-256 of the parts."""
        assert stable_id("a", "b") == "f04cdced9736a69da6103f08a4daaf8c"

    def test_length_truncates_the_digest(self):
        """A shorter length keeps the same prefix."""
        assert stable_id("a", "b", length=16) == "f04cdced9736a69d"

    def test_rejects_parts_containing_the_separator(self):
        """A part with the separator would collide with a split part list."""
        with pytest.raises(ValueError, match="separator"):
            stable_id("a\x1fb")


class TestUuid5Id:
    """Behavior of uuid5_id."""

    def test_returns_fixed_uuid_for_known_parts(self):
        """Parts are joined with a bar and hashed under the namespace."""
        namespace = UUID("12345678-1234-5678-1234-567812345678")

        assert uuid5_id(namespace, "ep", "e0") == "21ffe044-12d3-592e-9489-7166f4ee5887"
