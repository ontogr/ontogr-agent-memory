"""Cursor tests cover round-trip values and malformed public inputs."""

from datetime import UTC, datetime

import pytest

from ontogr_mem.service.reads import decode_episode_cursor, encode_episode_cursor


class TestEpisodeCursor:
    """The episode cursor is an opaque, URL-safe keyset position."""

    def test_round_trips_timestamp_and_episode_id(self):
        """A valid cursor preserves a precise timestamp and id."""
        position = (datetime(2026, 10, 23, 12, 0, 0, 123456, tzinfo=UTC), "ep-1")

        assert decode_episode_cursor(encode_episode_cursor(position)) == position

    @pytest.mark.parametrize("cursor", [None, ""])
    def test_empty_cursor_means_first_page(self, cursor):
        """An omitted or empty cursor has no keyset boundary."""
        assert decode_episode_cursor(cursor) is None

    @pytest.mark.parametrize("cursor", ["!", "bm90LXRpbWV8ZXA=", "MjAyNi0wMS0wMXx"])
    def test_malformed_cursor_is_rejected(self, cursor):
        """A cursor without a valid timestamp and episode id is rejected."""
        with pytest.raises(ValueError, match="cursor"):
            decode_episode_cursor(cursor)
