"""Tests for episode text helpers."""

import pytest

from ontogr_mem.entity_extraction.text import find_spans, language_of
from tests.support.extraction import text_episode


class TestFindSpans:
    """Behavior of find_spans."""

    def test_finds_every_case_insensitive_occurrence(self):
        """All occurrences are returned with their offsets."""
        assert find_spans("Acme hired acme staff", "ACME") == [(0, 4), (11, 15)]

    def test_matches_across_flexible_whitespace(self):
        """A name written over a line break still matches."""
        assert find_spans("New\n  York", "new york") == [(0, 10)]

    @pytest.mark.parametrize("needle", ["", "   "])
    def test_empty_needle_finds_nothing(self, needle):
        """A blank name has no spans."""
        assert find_spans("anything", needle) == []


class TestLanguageOf:
    """Behavior of language_of."""

    def test_defaults_to_english(self):
        """An episode without a language is English."""
        assert language_of(text_episode("hi")) == "en"

    def test_reads_metadata_language(self):
        """The metadata value is used when present."""
        assert language_of(text_episode("hi", metadata={"language": "de"})) == "de"
