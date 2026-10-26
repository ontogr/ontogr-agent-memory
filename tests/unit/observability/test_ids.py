"""Trace id tests pin the accepted format used in records and log lines."""

from ontogr_mem.observability.ids import accept_trace_id


class TestTraceIds:
    """Trace ids accept safe caller values and replace unsafe values."""

    def test_accepts_a_valid_trace_id_without_changing_it(self):
        """A valid trace id is returned exactly as supplied."""
        assert accept_trace_id("abc-123_x.y") == "abc-123_x.y"

    def test_replaces_missing_empty_and_malformed_trace_ids(self):
        """Invalid ids become fresh 32-character hexadecimal values."""
        candidates = [None, "", "bad id", "line\nbreak", "a" * 65, "a" * 10_000]

        replacements = [accept_trace_id(candidate) for candidate in candidates]

        assert all(len(value) == 32 for value in replacements)
        assert all(int(value, 16) >= 0 for value in replacements)
        assert len(set(replacements)) == len(replacements)
