"""Redaction tests cover credential-shaped names, values and size limits."""

import pytest

from ontogr_mem.observability.redact import (
    MAX_ATTRIBUTE_CHARS,
    SUMMARY_CHARS,
    TRUNCATED_MARKER,
    cap,
    is_secret_name,
    redact_text,
    redact_value,
    summarize_exception,
)


class TestRedaction:
    """Redaction preserves useful typed values without exposing secrets."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("api_key", True),
            ("apiKey", True),
            ("APIKey", True),
            ("X-Api-Key", True),
            ("idempotency_key", True),
            ("Authorization", True),
            ("session.cookie", True),
            ("db_password", True),
            ("tokens_in", False),
            ("budget_tokens", False),
            ("keyword", False),
            ("token_count", False),
            ("request.token_count", False),
            ("tenant", False),
            ("count", False),
        ],
    )
    def test_secret_names_are_matched_as_words(self, name, expected):
        """Only credential words cause name-based redaction."""
        assert is_secret_name(name) is expected

    @pytest.mark.parametrize("value", ["raw", 42, 3.5, True, None, object()])
    def test_secret_values_are_hidden_regardless_of_type(self, value):
        """A credential-named attribute never exposes its supplied value."""
        assert redact_value("api_key", value) == "***"

    @pytest.mark.parametrize("value", [42, 3.5, True, None])
    def test_safe_scalar_types_are_preserved(self, value):
        """OpenTelemetry-compatible scalar values keep their original type."""
        assert redact_value("count", value) is value

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("Bearer abc.def", "Bearer ***"),
            ("key=mk_abcdefghijklmnopqrstuvwxyz1234", "key=mk_***"),
            ("redis://user:pw@host:6379", "redis://***@host:6379"),
            ("https://token@host/x", "https://***@host/x"),
            ("the word bearer is harmless", "the word bearer is harmless"),
        ],
    )
    def test_text_credentials_are_redacted(self, value, expected):
        """Bearer tokens, minted keys and URL userinfo are hidden."""
        assert redact_text(value) == expected

    def test_value_is_capped_without_exceeding_the_limit(self):
        """Truncation keeps its marker within the configured character limit."""
        exact = "a" * MAX_ATTRIBUTE_CHARS
        over = "b" * (MAX_ATTRIBUTE_CHARS + 1)
        huge = "c" * 100_000

        assert cap(exact) == exact
        assert len(cap(over)) == MAX_ATTRIBUTE_CHARS
        assert cap(over).endswith(TRUNCATED_MARKER)
        assert len(cap(huge)) == MAX_ATTRIBUTE_CHARS
        assert cap(huge).endswith(TRUNCATED_MARKER)

    @pytest.mark.parametrize(
        ("error", "expected"),
        [
            (ValueError("boom\nbody"), "ValueError: boom"),
            (ValueError(""), "ValueError"),
            (ValueError("Bearer secret"), "ValueError: Bearer ***"),
        ],
    )
    def test_exception_summary_keeps_only_redacted_first_line(self, error, expected):
        """An exception summary cannot expose later lines or bearer credentials."""
        assert summarize_exception(error) == expected

    def test_exception_summary_is_limited_to_200_characters(self):
        """Long exception text is cut to the summary limit."""
        summary = summarize_exception(ValueError("x" * 500))

        assert len(summary) <= SUMMARY_CHARS
