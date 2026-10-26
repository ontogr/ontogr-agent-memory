"""Tests for the span-based local extractors."""

from ontogr_mem.data_models import UNKNOWN_TYPE_ID, Method
from ontogr_mem.entity_extraction.extractors import (
    GLiNERExtractor,
    RawSpan,
    SpacyExtractor,
)
from tests.support.extraction import FakeGliner, catalog, text_episode


TEXT = "Ana uses Postgres. Ana likes Postgres."


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


class TestGLiNERExtractor:
    """Behavior of GLiNERExtractor over a fake model."""

    async def test_types_by_score_and_groups_repeat_mentions(self):
        """Scores at the threshold are typed, lower ones Unknown, repeats grouped."""
        model = FakeGliner(
            [
                RawSpan(0, 3, "Ana", "Person", 0.9),
                RawSpan(19, 22, "Ana", "Person", 0.95),
                RawSpan(9, 17, "Postgres", "Database", 0.6),
            ]
        )
        extractor = GLiNERExtractor(model=model)

        result = await extractor.extract(text_episode(TEXT), catalog())

        by_name = {e.name: e for e in result.entities}
        assert by_name["Ana"].type_id == 0
        assert by_name["Ana"].spans == ((0, 3), (19, 22))
        assert by_name["Ana"].confidence == 0.95
        assert by_name["Ana"].method is Method.GLINER
        assert by_name["Postgres"].type_id == UNKNOWN_TYPE_ID

    async def test_ignores_scores_below_the_floor(self):
        """A weak hit is not an entity."""
        model = FakeGliner([RawSpan(0, 3, "Ana", "Person", 0.2)])

        result = await GLiNERExtractor(model=model).extract(
            text_episode(TEXT), catalog()
        )

        assert result.entities == []

    async def test_discards_a_span_that_does_not_match_the_text(self):
        """Offsets pointing at other text are discarded."""
        model = FakeGliner([RawSpan(0, 3, "Bob", "Person", 0.99)])

        result = await GLiNERExtractor(model=model).extract(
            text_episode(TEXT), catalog()
        )

        assert result.entities == []


class TestSpacyExtractor:
    """Behavior of SpacyExtractor over a fake pipeline."""

    async def test_maps_labels_and_skips_unmapped_ones(self):
        """Person maps to the catalog; a label the catalog lacks is skipped."""
        nlp = FakeNlp([FakeEnt("Ana", "PERSON", 0), FakeEnt("Paris", "GPE", 30)])

        result = await SpacyExtractor(nlp=nlp).extract(
            text_episode(TEXT + " Paris"), catalog()
        )

        assert [(e.name, e.type_id, e.method) for e in result.entities] == [
            ("Ana", 0, Method.SPACY)
        ]
