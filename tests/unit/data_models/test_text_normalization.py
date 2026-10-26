"""Shared normalization rules for label and name identity."""

from ontogr_mem.data_models.text import normalize_label, normalize_name, same_name


class TestLabelNormalization:
    """Labels ignore case, whitespace and underscores."""

    def test_removes_whitespace_and_underscores_after_casefolding(self) -> None:
        """Label separators do not change the canonical label key."""
        assert normalize_label("  Straße _ Works  ") == "strasseworks"


class TestNameNormalization:
    """Names share case, Unicode and whitespace rules."""

    def test_casefolds_and_collapses_whitespace(self) -> None:
        """Case and repeated whitespace do not change a name key."""
        assert normalize_name("  Straße\tSystems  ") == "strasse systems"

    def test_uses_canonical_unicode_equivalence(self) -> None:
        """Composed and decomposed forms of the same name share a key."""
        assert normalize_name("Café") == normalize_name("Cafe\u0301")

    def test_same_name_uses_the_normalized_keys(self) -> None:
        """Names with different case and spacing compare equal."""
        assert same_name("  Acme   Corp ", "ACME corp")
        assert not same_name("Acme", "Acme Labs")

    def test_name_normalization_keeps_word_separators(self) -> None:
        """Name keys retain a separator between words."""
        assert normalize_name("A B") == "a b"
