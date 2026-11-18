"""Input and output path rules for the extraction mode comparison."""

import pytest

from tests.eval.extraction_modes import load_episodes, validate_output_path


def test_load_episodes_skips_blank_lines_and_applies_defaults(tmp_path):
    """Episode defaults follow the physical JSONL line number."""
    source = tmp_path / "episodes.jsonl"
    source.write_text('\n{"text":"hello"}\n{"text":"bye","id":"custom"}\n')

    episodes = load_episodes(source)

    assert len(episodes) == 2
    assert episodes[0].id == "ep2"
    assert episodes[0].kind == "message"
    assert episodes[0].actor == "user"
    assert episodes[1].id == "custom"


@pytest.mark.parametrize("payload", ["{}", '{"text": 2}', "[]"])
def test_load_episodes_rejects_invalid_rows(tmp_path, payload):
    """Malformed rows fail with their source line number."""
    source = tmp_path / "episodes.jsonl"
    source.write_text(payload + "\n")

    with pytest.raises(ValueError, match="line 1"):
        load_episodes(source)


def test_output_path_must_be_outside_repository(tmp_path):
    """Results cannot be written into the source checkout."""
    with pytest.raises(ValueError, match="outside the repository"):
        validate_output_path(tmp_path / "nested/result.json", repository_root=tmp_path)

    destination = tmp_path.parent / "result.json"
    assert validate_output_path(destination, repository_root=tmp_path) == destination
