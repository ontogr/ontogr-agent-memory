"""Tests for the model entity extractor."""

import pytest

from ontogr_mem.data_models import UNKNOWN_TYPE_ID, Method
from ontogr_mem.entity_extraction.errors import ParseFailedError
from ontogr_mem.entity_extraction.extractors import LLMExtractor
from ontogr_mem.entity_extraction.prompts import NER_SYSTEM, build_ner_prompt
from ontogr_mem.entity_extraction.settings import EntityExtractionSettings
from ontogr_mem.llm import InputTooLargeError, estimate_tokens
from tests.support.extraction import catalog, entities_reply, gate, text_episode
from tests.support.llm import ScriptedLLM


TEXT = "Ana moved the Orders table to Postgres."


def reply(*entities):
    """Build a scripted reply."""
    return entities_reply(*entities)


class TestLLMExtractor:
    """Behavior of LLMExtractor."""

    async def test_binds_entities_to_spans_and_types(self):
        """Spans come from the text, not from the model."""
        llm = ScriptedLLM(
            [
                reply(
                    {"name": "Ana", "type_id": 0, "confidence": 0.9},
                    {"name": "Postgres", "type_id": 1, "confidence": 0.8},
                )
            ]
        )

        result = await LLMExtractor(gate(llm)).extract(text_episode(TEXT), catalog())

        assert [(e.name, e.type_id, e.spans, e.method) for e in result.entities] == [
            ("Ana", 0, ((0, 3),), Method.LLM),
            ("Postgres", 1, ((30, 38),), Method.LLM),
        ]
        assert result.usage.buckets["entities/small"].calls == 1

    async def test_discards_entities_that_are_not_in_the_text_or_have_a_bad_id(self):
        """A name missing from the episode and an id outside the catalog are dropped."""
        llm = ScriptedLLM(
            [
                reply(
                    {"name": "Bob", "type_id": 0, "confidence": 0.9},
                    {"name": "Ana", "type_id": 9, "confidence": 0.9},
                    {"name": "Postgres", "type_id": 1, "confidence": 0.9},
                )
            ]
        )

        result = await LLMExtractor(gate(llm)).extract(text_episode(TEXT), catalog())

        assert [e.name for e in result.entities] == ["Postgres"]

    async def test_keeps_the_type_label_only_when_no_catalog_type_fits(self):
        """A named type is kept for an Unknown entity and ignored for a typed one."""
        llm = ScriptedLLM(
            [
                reply(
                    {"name": "Orders", "type_id": -1, "type_label": "Table"},
                    {"name": "Ana", "type_id": 0, "type_label": "Human"},
                )
            ]
        )

        result = await LLMExtractor(gate(llm)).extract(text_episode(TEXT), catalog())

        assert [(e.name, e.type_id, e.type_label) for e in result.entities] == [
            ("Orders", UNKNOWN_TYPE_ID, "Table"),
            ("Ana", UNKNOWN_TYPE_ID, None),
        ]

    async def test_low_confidence_entity_is_typed_unknown(self):
        """Below the threshold the model's type is not trusted."""
        llm = ScriptedLLM([reply({"name": "Ana", "type_id": 0, "confidence": 0.6})])

        result = await LLMExtractor(gate(llm)).extract(text_episode(TEXT), catalog())

        assert result.entities[0].type_id == UNKNOWN_TYPE_ID

    async def test_repairs_an_invalid_reply(self):
        """The second call carries the errors and no schema."""
        llm = ScriptedLLM(["not json", reply({"name": "Ana", "type_id": 0})])

        result = await LLMExtractor(gate(llm)).extract(text_episode(TEXT), catalog())

        assert [e.name for e in result.entities] == ["Ana"]
        assert llm.requests[0].schema is not None
        assert llm.requests[1].schema is None
        assert "validation_errors" in llm.requests[1].prompt

    async def test_raises_parse_failed_when_repairs_run_out(self):
        """The repair budget is one extra call by default."""
        llm = ScriptedLLM(["bad", "still bad"])

        with pytest.raises(ParseFailedError):
            await LLMExtractor(gate(llm)).extract(text_episode(TEXT), catalog())
        assert len(llm.requests) == 2


class TestPreviousEpisodes:
    """Earlier episodes are context, never evidence."""

    async def test_previous_episodes_are_in_the_prompt_oldest_first(self):
        """The block lists earlier episodes in the order given."""
        llm = ScriptedLLM([reply()])
        previous = [
            text_episode("First earlier note.", "p1", minutes=-2),
            text_episode("Second earlier note.", "p2", minutes=-1),
        ]

        await LLMExtractor(gate(llm)).extract(text_episode(TEXT), catalog(), previous)

        prompt = llm.requests[0].prompt
        assert prompt.index("First earlier note.") < prompt.index(
            "Second earlier note."
        )
        assert prompt.index("Second earlier note.") < prompt.index(TEXT)
        assert prompt.count("<episode_text>") == 1

    async def test_no_block_without_previous_episodes(self):
        """An empty context adds no previous_episodes block."""
        llm = ScriptedLLM([reply()])

        await LLMExtractor(gate(llm)).extract(text_episode(TEXT), catalog())

        assert "<previous_episodes>" not in llm.requests[0].prompt

    async def test_a_name_found_only_in_a_previous_episode_is_not_an_entity(self):
        """Spans are searched in the current episode only."""
        llm = ScriptedLLM([reply({"name": "Carol", "type_id": 0, "confidence": 0.9})])
        previous = [text_episode("Carol joined.", "p1", minutes=-1)]

        result = await LLMExtractor(gate(llm)).extract(
            text_episode(TEXT), catalog(), previous
        )

        assert result.entities == []

    async def test_oldest_previous_episode_is_dropped_first_when_the_prompt_is_large(
        self,
    ):
        """The current episode is never cut; context shrinks from the oldest end."""
        episode = text_episode(TEXT)
        bare = estimate_tokens(NER_SYSTEM) + estimate_tokens(
            build_ner_prompt(episode, catalog())
        )
        previous = [
            text_episode(f"{tag} " + "x" * 300, f"p{tag}", minutes=-3 + i)
            for i, tag in enumerate("ABC")
        ]
        llm = ScriptedLLM([reply()])

        await LLMExtractor(gate(llm, bare + 150)).extract(episode, catalog(), previous)

        prompt = llm.requests[0].prompt
        assert "C xxx" in prompt
        assert "A xxx" not in prompt
        assert "B xxx" not in prompt
        assert TEXT in prompt

    async def test_raises_when_the_episode_alone_is_too_large(self):
        """With no context left the input cap error is raised."""
        llm = ScriptedLLM()
        previous = [text_episode("early", "p1", minutes=-1)]

        with pytest.raises(InputTooLargeError):
            await LLMExtractor(gate(llm, 10)).extract(
                text_episode(TEXT), catalog(), previous
            )
        assert llm.requests == []


class TestSettings:
    """Settings change extractor behavior."""

    async def test_threshold_comes_from_settings(self):
        """A lower threshold keeps the model's type."""
        settings = EntityExtractionSettings(threshold=0.5)
        llm = ScriptedLLM([reply({"name": "Ana", "type_id": 0, "confidence": 0.6})])

        result = await LLMExtractor(gate(llm), settings).extract(
            text_episode(TEXT), catalog()
        )

        assert result.entities[0].type_id == 0
