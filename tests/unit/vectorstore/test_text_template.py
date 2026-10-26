"""Tests for text normalisation, templates and hashing."""

import re

import pytest

from ontogr_mem.vectorstore import VectorKind
from ontogr_mem.vectorstore.text import (
    TEMPLATES,
    Template,
    edge_text,
    normalise_text,
    text_hash,
)


class TestNormaliseText:
    """Behavior of `normalise_text`."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("  a \n\t b  ", "a b"),
            ("Acme\nCorp", "Acme Corp"),
            ("a\r\n\r\nb", "a b"),
            ("  ", ""),
            ("e\u0301", "\u00e9"),
        ],
    )
    def test_returns_nfc_with_single_spaces(self, raw: str, expected: str) -> None:
        """Whitespace runs, including newlines, become one space."""
        assert normalise_text(raw) == expected


class TestTemplates:
    """Behavior of the `Template` renderings."""

    @pytest.mark.parametrize("kind", [VectorKind.NODE, VectorKind.EDGE])
    def test_property_text_is_normalised(self, kind: VectorKind) -> None:
        """Node and edge text lose whitespace runs."""
        assert TEMPLATES[kind].render("  Acme \n\t Corp ") == "Acme Corp"

    def test_episode_text_is_embedded_as_given(self) -> None:
        """Episode text keeps its whitespace exactly."""
        assert TEMPLATES[VectorKind.EPISODE].render("a  b\n") == "a  b\n"

    def test_episode_text_is_not_nfc_normalised(self) -> None:
        """The exact template leaves a decomposed character alone."""
        assert TEMPLATES[VectorKind.EPISODE].render("e\u0301") == "e\u0301"


class TestTextHash:
    """Behavior of `text_hash`."""

    def test_returns_a_stable_sha256_hex_digest(self) -> None:
        """The same template and text always hash the same."""
        template = TEMPLATES[VectorKind.NODE]

        digest = text_hash(template, "alice")

        assert digest == text_hash(template, "alice")
        assert re.fullmatch(r"[0-9a-f]{64}", digest) is not None

    @pytest.mark.parametrize(
        ("template", "rendered"),
        [
            (Template("edge.fact", 1), "alice"),
            (Template("node.name", 2), "alice"),
            (TEMPLATES[VectorKind.NODE], "alicf"),
        ],
        ids=["template-id", "template-version", "text"],
    )
    def test_digest_changes_with_template_or_text(
        self, template: Template, rendered: str
    ) -> None:
        """The id, the version and the text all take part in the digest."""
        baseline = text_hash(TEMPLATES[VectorKind.NODE], "alice")

        assert text_hash(template, rendered) != baseline


class TestEdgeText:
    """Behavior of `edge_text`."""

    def test_normalises_a_given_sentence(self) -> None:
        """A statement is preferred and normalised."""
        got = edge_text(" Alice works\nat Acme ", "A", "works_at", "B")

        assert got == "Alice works at Acme"

    @pytest.mark.parametrize("statement", ["", "   ", "\n\t"])
    def test_falls_back_to_subject_relation_object(self, statement: str) -> None:
        """A blank statement becomes the normalised triple."""
        got = edge_text(statement, " Alice ", "works\nat", "Acme ")

        assert got == "Alice works at Acme"
