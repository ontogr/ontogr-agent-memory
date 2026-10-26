"""Tests for the composite and fallback extractors."""

import pytest

from ontogr_mem.data_models import UNKNOWN_TYPE_ID, Method
from ontogr_mem.entity_extraction.errors import ExtractionError
from ontogr_mem.entity_extraction.extractors import (
    CompositeExtractor,
    EscalationThresholds,
    FallbackExtractor,
    GLiNERExtractor,
    LLMExtractor,
    RawSpan,
    SpacyExtractor,
    Trigger,
)
from tests.support.extraction import (
    FakeGliner,
    catalog,
    entities_reply,
    gate,
    text_episode,
)
from tests.support.llm import ScriptedLLM


class FakeEnt:
    """A spaCy entity."""

    def __init__(self, text, label, start):
        """Create the entity."""
        self.text, self.label_ = text, label
        self.start_char, self.end_char = start, start + len(text)


class FakeNlp:
    """A spaCy pipeline that returns fixed entities."""

    def __init__(self, ents):
        """Keep the entities."""
        self.ents = ents

    def __call__(self, text):
        """Return a doc with the entities."""
        return self


def gliner(*hits):
    """Build a GLiNER extractor over fixed hits."""
    return GLiNERExtractor(model=FakeGliner(hits))


class Failing:
    """An extractor that always raises."""

    languages = None

    def __init__(self, error):
        """Keep the error."""
        self.error = error

    async def extract(self, episode, types, previous=()):
        """Raise the error."""
        raise self.error


class TestCompositeExtractor:
    """Behavior of CompositeExtractor."""

    async def test_earlier_member_wins_the_type_and_fills_gaps_from_later_ones(self):
        """An Unknown type from the first member is filled by the second."""
        text = "Ana joined Acme."
        first = gliner(RawSpan(0, 3, "Ana", "Person", 0.6))
        second = SpacyExtractor(nlp=FakeNlp([FakeEnt("Ana", "PERSON", 0)]))

        result = await CompositeExtractor([first, second]).extract(
            text_episode(text), catalog()
        )

        (entity,) = result.entities
        assert (entity.name, entity.type_id, entity.method) == ("Ana", 0, Method.GLINER)

    async def test_longer_span_with_a_content_word_replaces_the_fragment(self):
        """`Paris Observatory` replaces `Paris`."""
        text = "We visited Paris Observatory."
        first = gliner(RawSpan(11, 16, "Paris", "Company", 0.9))
        second = gliner(RawSpan(11, 28, "Paris Observatory", "Company", 0.9))

        result = await CompositeExtractor([first, second]).extract(
            text_episode(text), catalog()
        )

        assert [e.name for e in result.entities] == ["Paris Observatory"]

    async def test_article_only_extension_keeps_the_earlier_entity(self):
        """`the Paris` does not replace `Paris`."""
        text = "We visited the Paris today."
        first = gliner(RawSpan(15, 20, "Paris", "Company", 0.9))
        second = gliner(RawSpan(11, 20, "the Paris", "Company", 0.9))

        result = await CompositeExtractor([first, second]).extract(
            text_episode(text), catalog()
        )

        assert [e.name for e in result.entities] == ["Paris"]

    async def test_a_partial_overlap_keeps_both_entities(self):
        """`Dr. Ana` and `Ana Smith` overlap without containing each other."""
        text = "Dr. Ana Smith"
        first = gliner(RawSpan(0, 7, "Dr. Ana", "Person", 0.9))
        second = gliner(RawSpan(4, 13, "Ana Smith", "Person", 0.9))

        result = await CompositeExtractor([first, second]).extract(
            text_episode(text), catalog()
        )

        assert [e.name for e in result.entities] == ["Dr. Ana", "Ana Smith"]

    async def test_skips_a_member_that_fails_unexpectedly(self):
        """Other members still contribute."""
        good = gliner(RawSpan(0, 3, "Ana", "Person", 0.9))

        result = await CompositeExtractor(
            [Failing(RuntimeError("boom")), good]
        ).extract(text_episode("Ana"), catalog())

        assert [e.name for e in result.entities] == ["Ana"]

    async def test_counted_failures_propagate(self):
        """An extraction error is not hidden by the other members."""
        good = gliner(RawSpan(0, 3, "Ana", "Person", 0.9))

        with pytest.raises(ExtractionError):
            await CompositeExtractor([Failing(ExtractionError("bad")), good]).extract(
                text_episode("Ana"), catalog()
            )

    async def test_raises_the_first_error_when_every_member_fails(self):
        """No member left means the episode fails."""
        with pytest.raises(RuntimeError, match="boom"):
            await CompositeExtractor([Failing(RuntimeError("boom"))]).extract(
                text_episode("Ana"), catalog()
            )

    async def test_reports_triggers_for_a_weak_result(self):
        """No entities in a long text trips the density trigger."""
        text = "word " * 1000
        thresholds = EscalationThresholds(domain_terms=frozenset({"word"}))

        result = await CompositeExtractor([gliner()], thresholds=thresholds).extract(
            text_episode(text), catalog()
        )

        assert result.triggers == (
            Trigger.TOO_FEW_ENTITIES,
            Trigger.DOMAIN_TERM_UNCOVERED,
        )

    async def test_too_few_entities_fires_for_a_text_length_that_is_not_round(self):
        """A 900 character text calls for 2.7 entities, so two are too few."""
        text = "Ana Bo Cy " + "x" * 890
        cheap = gliner(
            RawSpan(0, 3, "Ana", "Person", 0.9),
            RawSpan(4, 6, "Bo", "Person", 0.9),
        )

        result = await CompositeExtractor([cheap]).extract(
            text_episode(text), catalog()
        )

        assert result.triggers == (Trigger.TOO_FEW_ENTITIES,)


class TestFallbackExtractor:
    """Behavior of FallbackExtractor."""

    TEXT = "Ana joined Acme and Postgres."

    def llm(self, *entities):
        """Build a model extractor with a scripted reply."""
        return LLMExtractor(gate(ScriptedLLM([entities_reply(*entities)])))

    async def test_does_not_call_the_model_when_the_cheap_pass_is_good(self):
        """No trigger means one pass only."""
        llm = ScriptedLLM()
        cheap = gliner(RawSpan(0, 3, "Ana", "Person", 0.9))

        result = await FallbackExtractor(cheap, LLMExtractor(gate(llm))).extract(
            text_episode("Ana"), catalog()
        )

        assert [e.name for e in result.entities] == ["Ana"]
        assert llm.requests == []

    async def test_the_model_entity_wins_and_untouched_cheap_entities_survive(self):
        """The model is merged first, so it keeps the type for the same name."""
        cheap = gliner(
            RawSpan(0, 3, "Ana", "Person", 0.6),
            RawSpan(20, 28, "Postgres", "Database", 0.9),
        )
        thresholds = EscalationThresholds(max_unknown_share=0.4)
        model = self.llm({"name": "Ana", "type_id": 0, "confidence": 0.95})

        result = await FallbackExtractor(cheap, model, thresholds=thresholds).extract(
            text_episode(self.TEXT), catalog()
        )

        assert [(e.name, e.type_id, e.method) for e in result.entities] == [
            ("Ana", 0, Method.LLM),
            ("Postgres", 1, Method.GLINER),
        ]
        assert result.triggers == ()

    async def test_a_partially_overlapping_cheap_entity_survives_the_merge(self):
        """The cheap entity the model only partly covers is kept."""
        cheap = gliner(RawSpan(0, 7, "Dr. Ana", "Person", 0.9))
        model = self.llm({"name": "Ana Smith", "type_id": 0, "confidence": 0.9})
        thresholds = EscalationThresholds(min_entities_per_1k_chars=1000)

        result = await FallbackExtractor(cheap, model, thresholds=thresholds).extract(
            text_episode("Dr. Ana Smith"), catalog()
        )

        assert [(e.name, e.method) for e in result.entities] == [
            ("Ana Smith", Method.LLM),
            ("Dr. Ana", Method.GLINER),
        ]

    async def test_the_model_runs_when_the_cheap_pass_finds_too_few_entities(self):
        """A 900 character text needs three entities, so two escalate."""
        text = "Ana Bo Cy " + "x" * 890
        cheap = gliner(
            RawSpan(0, 3, "Ana", "Person", 0.9),
            RawSpan(4, 6, "Bo", "Person", 0.9),
        )
        model = self.llm({"name": "Cy", "type_id": 0, "confidence": 0.9})

        result = await FallbackExtractor(cheap, model).extract(
            text_episode(text), catalog()
        )

        assert [e.name for e in result.entities] == ["Cy", "Ana", "Bo"]

    async def test_only_on_limits_which_triggers_escalate(self):
        """A trigger outside `only_on` does not call the model."""
        llm = ScriptedLLM()
        cheap = gliner(RawSpan(0, 3, "Ana", "Person", 0.6))
        extractor = FallbackExtractor(
            cheap,
            LLMExtractor(gate(llm)),
            thresholds=EscalationThresholds(max_unknown_share=0.4),
            only_on=frozenset({Trigger.TOO_FEW_ENTITIES}),
        )

        result = await extractor.extract(text_episode("Ana"), catalog())

        assert llm.requests == []
        assert result.entities[0].type_id == UNKNOWN_TYPE_ID
        assert result.triggers == (Trigger.HIGH_UNKNOWN_SHARE,)
