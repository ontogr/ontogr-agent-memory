"""Retrieval query tokenization keeps names and removes server query syntax."""

import pytest

from ontogr_mem.retrieval.text import terms


class TestTerms:
    """Only safe distinct terms reach either server's query parser."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Berlin BERLIN Zürich", ["berlin", "zürich"]),
            ("O'Brien state-of-the-art", ["o'brien", "state", "of", "the", "art"]),
            (
                'a AND (b x:y "unterminated -minus',
                ["a", "and", "b", "x", "y", "unterminated", "minus"],
            ),
            ("() : *", []),
            ("", []),
        ],
    )
    def test_safe_distinct_terms(self, text, expected):
        """Punctuation cannot introduce operators into the server query."""
        assert terms(text) == expected

    def test_caps_terms_in_input_order(self):
        """A long query keeps only its first 32 distinct terms."""
        assert terms(" ".join(str(i) for i in range(40))) == [str(i) for i in range(32)]
