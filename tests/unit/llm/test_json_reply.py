"""The shared loader reads a JSON reply from the model."""

import pytest

from ontogr_mem.llm.json_reply import SchemaError, load_json


class TestLoadJson:
    """A reply is read whatever the model wrapped it in."""

    @pytest.mark.parametrize("fence", ["```json", "```JSON", "```"])
    def test_a_fenced_reply_is_read(self, fence):
        """A Markdown fence around the JSON is stripped, whatever its case."""
        assert load_json(f'{fence}\n{{"summary": "ok"}}\n```') == {"summary": "ok"}

    def test_prose_around_the_json_is_ignored(self):
        """The outermost braces of the reply are read when it is not plain JSON."""
        assert load_json('Here you go: {"summary": "ok"} enjoy') == {"summary": "ok"}

    def test_a_reply_without_json_raises_the_schema_error(self):
        """A reply that holds no JSON raises the schema error with the reason."""
        with pytest.raises(SchemaError) as excinfo:
            load_json("not json at all")

        assert excinfo.value.errors
