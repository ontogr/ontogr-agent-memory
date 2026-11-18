"""Behavior of the single-call entity and fact extractor."""

from pydantic_ai.exceptions import ModelHTTPError

from ontogr_mem.combined_extraction import CombinedExtractor
from ontogr_mem.data_models import DropReason, PreviousEpisodes, Status, TypeView
from ontogr_mem.fact_extraction import FactDropReason
from tests.support.facts import fact_catalog, make_episode
from tests.support.scripted_model import scripted_model, scripted_role_models


def entity(ref: str, name: str, type_id: int) -> dict[str, object]:
    """Build one combined entity response item."""
    labels = {0: "Person", 1: "Database", 2: "Company"}
    return {
        "ref": ref,
        "name": name,
        "type_id": type_id,
        "type_label": labels[type_id],
        "description": name,
        "confidence": 0.9,
    }


def combined_reply(entities=(), facts=()):
    """Build one complete combined response."""
    return {"entities": list(entities), "facts": list(facts)}


def fact(entity_refs=(), evidence_quote="I switched to Postgres"):
    """Build a fact response item."""
    return {
        "text": "Ana switched to Postgres",
        "entity_refs": list(entity_refs),
        "relation_type_id": None,
        "relation_label": None,
        "fact_kind": "event",
        "polarity": "affirmed",
        "modality": "actual",
        "evidence_quote": evidence_quote,
    }


def prepare(reply, episode=None):
    """Run one combined response against the fact test ontology."""
    model, scripted = scripted_model([reply])
    extractor = CombinedExtractor(scripted_role_models(model))
    return extractor, scripted, episode or make_episode(), TypeView(fact_catalog())


class TestCombinedExtractor:
    """Behavior visible to a text-write caller."""

    async def test_entities_and_facts_share_one_call_and_bind_to_local_ids(self):
        """Bind entity references from the shared model response."""
        extractor, scripted, episode, catalog = prepare(
            combined_reply(
                [entity("x", "Acme", 2), entity("y", "Postgres", 1)],
                [fact(["x", "y"], "because Acme needs transactions")],
            )
        )

        entities, facts = await extractor.extract(episode, catalog, None)

        assert entities.status is Status.OK
        assert [item.name for item in entities.entities] == ["Acme", "Postgres", "Ana"]
        assert facts.status is Status.OK
        assert facts.facts[0].entity_refs == tuple(
            item.local_id for item in entities.entities[:2]
        )
        assert facts.usage.buckets["combined/fact_extraction"].calls == 1
        assert len(scripted.calls) == 1

    async def test_same_name_refs_with_distinct_types_bind_separately(self):
        """Same-name refs with different effective types get distinct local ids."""
        episode = make_episode("Jordan switched to Postgres because Jordan needs it")
        extractor, _, episode, catalog = prepare(
            combined_reply(
                [entity("person", "Jordan", 0), entity("company", "Jordan", 2)],
                [fact(["person", "company"], "Jordan switched to Postgres")],
            ),
            episode,
        )

        entities, facts = await extractor.extract(episode, catalog, None)

        jordans = [item for item in entities.entities if item.name == "Jordan"]
        assert [(item.local_id, item.type_id) for item in jordans] == [
            ("e0", 0),
            ("e1", 2),
        ]
        assert facts.facts[0].entity_refs == ("e0", "e1")

    async def test_rejected_entities_are_reported_as_drops(self):
        """Report every entity the binder rejects with a public drop reason."""
        bad_type = {"ref": "bad", "name": "Atlas", "type_id": 9, "confidence": 0.9}
        extractor, _, episode, catalog = prepare(
            combined_reply(
                [
                    bad_type,
                    entity("blank", "", 1),
                    entity("missing", "Atlantis", 2),
                    entity("kept", "Postgres", 1),
                ],
                [],
            )
        )

        entities, facts = await extractor.extract(episode, catalog, None)

        assert [(d.reason, d.item["name"]) for d in entities.dropped] == [
            (DropReason.TYPE_OUT_OF_RANGE, "Atlas"),
            (DropReason.MISSING_FIELD, ""),
            (DropReason.MISSING_FIELD, "Atlantis"),
        ]
        assert [item.name for item in entities.entities] == ["Postgres", "Ana"]
        assert facts.status is Status.EMPTY

    async def test_dropped_entity_ref_does_not_collide_with_renumbered_entity(self):
        """Keep surviving entity references aligned after a dropped entity."""
        extractor, _, episode, catalog = prepare(
            combined_reply(
                [entity("lost", "Atlantis", 2), entity("kept", "Postgres", 1)],
                [fact(["lost"]), fact(["kept"])],
            )
        )

        entities, facts = await extractor.extract(episode, catalog, None)

        assert [item.name for item in entities.entities] == ["Postgres", "Ana"]
        assert len(facts.facts) == 1
        assert facts.facts[0].entity_refs == (entities.entities[0].local_id,)
        assert any(
            item.reason is FactDropReason.UNKNOWN_ENTITY_REF for item in facts.dropped
        )

    async def test_message_speaker_is_available_as_actor_reference(self):
        """Allow facts to refer to the message speaker."""
        extractor, _, episode, catalog = prepare(
            combined_reply([], [fact(["actor"], "I switched")])
        )

        entities, facts = await extractor.extract(episode, catalog, None)

        assert any(item.name == "Ana" for item in entities.entities)
        assert facts.facts[0].entity_refs == ("actor",)

    async def test_duplicate_refs_are_reasked_as_a_whole(self):
        """Retry the response when entity references are duplicated."""
        first = combined_reply(
            [entity("same", "Ana", 0), entity("same", "Postgres", 1)]
        )
        second = combined_reply([entity("x", "Ana", 0)], [])
        model, scripted = scripted_model([first, second])
        extractor = CombinedExtractor(scripted_role_models(model))

        await extractor.extract(make_episode(), TypeView(fact_catalog()), None)

        assert len(scripted.calls) == 2

    async def test_provider_failure_sets_both_results_to_error(self):
        """Report model failures on both extraction results."""
        model, scripted = scripted_model(
            [ModelHTTPError(503, "model", {"error": "down"})]
        )
        extractor = CombinedExtractor(scripted_role_models(model))

        entities, facts = await extractor.extract(
            make_episode(), TypeView(fact_catalog()), None
        )

        assert entities.status is Status.ERROR
        assert facts.status is Status.ERROR
        assert len(scripted.calls) == 1
        assert facts.usage.buckets["combined/fact_extraction"].calls == 0

    async def test_blank_text_does_not_call_model(self):
        """Skip model requests for blank episode text."""
        model, scripted = scripted_model()
        extractor = CombinedExtractor(scripted_role_models(model))

        entities, facts = await extractor.extract(
            make_episode("  "), TypeView(fact_catalog()), None
        )

        assert entities.status is facts.status is Status.EMPTY
        assert scripted.calls == []

    async def test_previous_provider_selects_context_for_prompt_and_binding(self):
        """Use the provider's bounded history for prompting and grounding."""
        earlier = make_episode("Ana switched to Postgres", episode_id="earlier")

        class Previous:
            def __init__(self):
                self.limits: list[int] = []

            async def before(self, episode, limit):
                self.limits.append(limit)
                return [earlier]

        provider: PreviousEpisodes = Previous()
        extractor, scripted, episode, catalog = prepare(
            combined_reply([], [fact(evidence_quote="Ana switched to Postgres")])
        )

        _, facts = await extractor.extract(episode, catalog, provider)

        assert provider.limits == [3]
        assert "Ana switched to Postgres" in scripted.calls[0].prompt
        assert facts.dropped[0].detail == "quote found only in a previous episode"
