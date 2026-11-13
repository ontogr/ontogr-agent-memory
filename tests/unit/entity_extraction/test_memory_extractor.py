"""Behavior of the optional memory entity enrichment pass."""

from collections.abc import Sequence
from dataclasses import replace

from pydantic_ai.exceptions import ModelHTTPError

from ontogr_mem.data_models import Episode, ExtractedEntity, Method, TypeCatalog
from ontogr_mem.entity_extraction.extractors import (
    EntityExtractor,
    Extracted,
    GLiNERExtractor,
    MemoryEntityExtractor,
    RawSpan,
)
from tests.support.extraction import (
    FakeGliner,
    catalog,
    entity_models,
    text_episode,
)
from tests.support.scripted_model import scripted_model, scripted_role_models


TEXT = "Ana is my manager and she dislikes meetings."


def base(hits: list[RawSpan] | None = None) -> GLiNERExtractor:
    """Build a base extractor that finds Ana unless hits are supplied."""
    return GLiNERExtractor(
        model=FakeGliner(
            hits if hits is not None else [RawSpan(0, 3, "Ana", "Person", 0.9)]
        )
    )


class ExistingEntityExtractor(EntityExtractor):
    """Return one entity that already has details from an earlier pass."""

    def __init__(self, entity: ExtractedEntity) -> None:
        """Store the entity returned by the extractor."""
        self.entity = entity

    async def extract(
        self,
        episode: Episode,
        types: TypeCatalog,
        previous: Sequence[Episode] = (),
    ) -> Extracted:
        """Return the prepared entity without using episode context."""
        return Extracted([self.entity])


class TestMemoryEntityExtractor:
    """The enrichment pass only adds grounded details to base entities."""

    async def test_keeps_only_allowed_attributes_with_episode_evidence(self):
        """Unsupported names, ids and quotes do not change the base entity."""
        models, scripted = entity_models(
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
                        },
                        {
                            "id": "unknown",
                            "description": "Ignored entity.",
                            "attributes": {},
                        },
                    ]
                }
            ]
        )

        result = await MemoryEntityExtractor(base(), models).extract(
            text_episode(TEXT), catalog()
        )

        (entity,) = result.entities
        assert entity.description == "The user's manager."
        assert entity.attributes == {"role": "manager"}
        assert (entity.name, entity.type_id, entity.confidence, entity.spans) == (
            "Ana",
            0,
            0.9,
            ((0, 3),),
        )
        assert entity.method is Method.GLINER
        assert result.usage.buckets["enrich/entity_extraction"].calls == 1
        assert len(scripted.calls) == 1

    async def test_keeps_scalar_attribute_values_without_coercion(self):
        """String, integer, float and boolean attribute values stay scalar."""
        models, _ = entity_models(
            [
                {
                    "entities": [
                        {
                            "id": "e0",
                            "attributes": {
                                "role": {"value": "manager", "evidence": "manager"},
                                "count": {"value": 2, "evidence": "meetings"},
                                "score": {"value": 0.5, "evidence": "meetings"},
                                "active": {"value": False, "evidence": "dislikes"},
                            },
                        }
                    ]
                }
            ]
        )

        result = await MemoryEntityExtractor(
            base(),
            models,
            allowed_attributes={"Person": ("role", "count", "score", "active")},
        ).extract(text_episode(TEXT), catalog())

        attributes = result.entities[0].attributes
        assert attributes == {
            "role": "manager",
            "count": 2,
            "score": 0.5,
            "active": False,
        }
        assert type(attributes["count"]) is int
        assert type(attributes["score"]) is float
        assert type(attributes["active"]) is bool

    async def test_caps_description(self):
        """A long description is stripped and capped at 300 characters."""
        models, _ = entity_models(
            [
                {
                    "entities": [
                        {
                            "id": "e0",
                            "description": "  " + "x" * 350 + "  ",
                            "attributes": {
                                "role": {"value": "manager", "evidence": "my manager"}
                            },
                        }
                    ]
                }
            ]
        )
        result = await MemoryEntityExtractor(base(), models).extract(
            text_episode(TEXT), catalog()
        )

        assert result.entities[0].description == "x" * 300
        assert result.entities[0].attributes == {"role": "manager"}

    async def test_keeps_the_base_description_when_enrichment_has_none(self):
        """An empty enrichment description does not replace the base text."""
        episode = text_episode(TEXT)
        types = catalog()
        original = (await base().extract(episode, types)).entities[0]
        original = replace(
            original,
            description="Team lead",
            attributes={"source": "base"},
        )
        models, _ = entity_models(
            [
                {
                    "entities": [
                        {
                            "id": "e0",
                            "description": "  ",
                            "attributes": {
                                "role": {"value": "manager", "evidence": "my manager"}
                            },
                        }
                    ]
                }
            ]
        )

        result = await MemoryEntityExtractor(
            ExistingEntityExtractor(original), models
        ).extract(episode, types)

        assert result.entities[0].description == "Team lead"
        assert result.entities[0].attributes == {
            "source": "base",
            "role": "manager",
        }

    async def test_uses_the_entity_extraction_role_override(self):
        """Enrichment uses its role-specific model instead of another role."""

        def fail(call):
            raise AssertionError("the shared model should not be called")

        shared, _ = scripted_model(answer=fail)
        extraction, _ = scripted_model(
            [
                {
                    "entities": [
                        {
                            "id": "e0",
                            "attributes": {
                                "role": {"value": "manager", "evidence": "my manager"}
                            },
                        }
                    ]
                }
            ]
        )
        models = scripted_role_models(shared, entity_extraction=extraction)

        result = await MemoryEntityExtractor(base(), models).extract(
            text_episode(TEXT), catalog()
        )

        assert result.entities[0].attributes == {"role": "manager"}

    async def test_returns_the_plain_entities_after_transport_failure(self):
        """A provider error does not discard base entities."""
        models, scripted = entity_models(
            [ModelHTTPError(503, "model", {"error": "unavailable"})]
        )

        result = await MemoryEntityExtractor(base(), models).extract(
            text_episode(TEXT), catalog()
        )

        assert [entity.name for entity in result.entities] == ["Ana"]
        assert result.entities[0].attributes == {}
        assert len(scripted.calls) == 1

    async def test_reasks_a_malformed_reply_and_counts_all_requests(self):
        """A non-scalar value rejects the reply and a valid re-ask is applied."""
        models, scripted = entity_models(
            [
                {
                    "entities": [
                        {
                            "id": "e0",
                            "attributes": {
                                "role": {
                                    "value": ["manager"],
                                    "evidence": "my manager",
                                }
                            },
                        }
                    ]
                },
                {
                    "entities": [
                        {
                            "id": "e0",
                            "attributes": {
                                "role": {"value": "manager", "evidence": "my manager"}
                            },
                        }
                    ]
                },
            ]
        )

        result = await MemoryEntityExtractor(base(), models).extract(
            text_episode(TEXT), catalog()
        )

        assert result.entities[0].attributes == {"role": "manager"}
        assert len(scripted.calls) == 2
        assert result.usage.buckets["enrich/entity_extraction"].calls == 2

    async def test_keeps_a_later_batch_when_its_output_retries_exhaust(self):
        """A failed later batch keeps its entity and counts all failed requests."""
        hits = [
            RawSpan(0, 3, "Ana", "Person", 0.9),
            RawSpan(35, 43, "meetings", "Person", 0.8),
        ]
        models, scripted = entity_models(
            [
                {
                    "entities": [
                        {
                            "id": "e0",
                            "attributes": {
                                "role": {"value": "manager", "evidence": "my manager"}
                            },
                        }
                    ]
                },
                "bad",
                "bad",
                "bad",
            ]
        )

        result = await MemoryEntityExtractor(
            base(hits), models, max_entities_per_call=1
        ).extract(text_episode(TEXT), catalog())

        assert [entity.name for entity in result.entities] == ["Ana", "meetings"]
        assert result.entities[0].attributes == {"role": "manager"}
        assert result.entities[1].attributes == {}
        assert len(scripted.calls) == 4
        assert result.usage.buckets["enrich/entity_extraction"].calls == 4
