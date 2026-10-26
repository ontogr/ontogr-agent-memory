"""Tests for the memory (attribute) extractor."""

from ontogr_mem.entity_extraction.extractors import (
    GLiNERExtractor,
    MemoryEntityExtractor,
    RawSpan,
)
from ontogr_mem.llm import LLMCallError
from tests.support.extraction import FakeGliner, catalog, gate, text_episode
from tests.support.llm import ScriptedLLM


TEXT = "Ana is my manager and she dislikes meetings."


def base():
    """Build a base extractor that finds Ana."""
    return GLiNERExtractor(model=FakeGliner([RawSpan(0, 3, "Ana", "Person", 0.9)]))


class TestMemoryEntityExtractor:
    """Behavior of MemoryEntityExtractor."""

    async def test_keeps_only_allowed_attributes_with_evidence_in_the_text(self):
        """Evidence missing from the text, or an unlisted name, drops the attribute."""
        llm = ScriptedLLM(
            [
                {
                    "entities": [
                        {
                            "id": "e0",
                            "description": "  The user's manager.  ",
                            "attributes": {
                                "role": {"value": "manager", "evidence": "my manager"},
                                "stance": {"value": "angry", "evidence": "furious"},
                                "mood": {"value": "x", "evidence": "Ana"},
                            },
                        }
                    ]
                }
            ]
        )

        result = await MemoryEntityExtractor(base(), gate(llm)).extract(
            text_episode(TEXT), catalog()
        )

        (entity,) = result.entities
        assert entity.description == "The user's manager."
        assert entity.attributes == {"role": "manager"}
        assert result.usage.buckets["enrich/small"].calls == 1

    async def test_failed_enrichment_keeps_the_plain_result(self):
        """A model failure does not lose the entities."""

        def fail(request):
            raise LLMCallError("down")

        result = await MemoryEntityExtractor(
            base(), gate(ScriptedLLM(answer=fail))
        ).extract(text_episode(TEXT), catalog())

        assert [e.name for e in result.entities] == ["Ana"]
        assert result.entities[0].attributes == {}

    async def test_malformed_reply_keeps_the_plain_result(self):
        """A reply that is not the expected JSON changes nothing."""
        result = await MemoryEntityExtractor(
            base(), gate(ScriptedLLM(["oops"]))
        ).extract(text_episode(TEXT), catalog())

        assert result.entities[0].description == ""
