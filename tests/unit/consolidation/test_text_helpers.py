"""Text helpers: sentence-aware truncation, JSON quoting and stable short ids."""

from ontogr_mem.consolidation.text import quote, short_hash, truncate_at_sentence


def test_short_text_is_returned_stripped() -> None:
    """Short text is returned stripped."""
    assert truncate_at_sentence("  one two.  ", 50) == "one two."


def test_long_text_is_cut_at_the_last_sentence_end_inside_the_limit() -> None:
    """Long text is cut at the last sentence end inside the limit."""
    text = "First sentence. Second sentence here. Third one runs long on and on."

    assert truncate_at_sentence(text, 40) == "First sentence. Second sentence here."


def test_long_text_without_a_sentence_end_is_cut_and_right_stripped() -> None:
    """Text with no sentence end is cut and right-stripped."""
    assert truncate_at_sentence("word word word word", 9) == "word word"


def test_quote_keeps_untrusted_text_inside_one_json_string() -> None:
    """Quoting keeps untrusted text inside one JSON string."""
    assert quote('say "hi"\n') == '"say \\"hi\\"\\n"'


def test_short_hash_is_stable_and_twelve_characters() -> None:
    """The short hash is stable and twelve characters long."""
    first = short_hash("a", 1)

    assert first == short_hash("a", 1)
    assert len(first) == 12
    assert first != short_hash("a", 2)
