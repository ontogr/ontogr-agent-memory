"""Tests that stored ontology type fields survive resolution round trips."""

from dataclasses import replace

import pytest

from ontogr_mem.data_models import ExtractedEntity
from ontogr_mem.data_models.entities import Entity
from ontogr_mem.ontology.declaration import EntityDeclaration, OntologyDeclaration
from ontogr_mem.ontology.declare import build_target
from ontogr_mem.ontology.links import TypeLinks
from ontogr_mem.ontology.models import Ontology
from ontogr_mem.resolution.errors import MergeRefusedError
from ontogr_mem.resolution.policy import ResolutionMode, ResolutionPolicy, TypePolicy
from ontogr_mem.resolution.resolver_store import ResolverStore
from tests.integration.resolution.resolver_helpers import (
    count_active,
    ex,
    make,
    stamp,
)
from tests.support.type_world import WORLD
from tests.unit.resolution.helpers import oracle_models


def typed(
    extracted: ExtractedEntity, label: str, proposed: str = ""
) -> ExtractedEntity:
    """Stamp an extracted entity with a type label and a proposed type."""
    return replace(stamp(extracted, label), proposed_type=proposed)


def custom_world(*declarations: EntityDeclaration) -> tuple[dict[str, int], TypeLinks]:
    """Build a world with declaration-level merges and renames."""
    declaration = OntologyDeclaration(entities=list(declarations))
    ontology = build_target(Ontology.genesis(), declaration)
    ids = {entity.label: entity.type_id for entity in ontology.entities}
    return ids, TypeLinks.from_ontology(ontology)


async def named(store: ResolverStore, name: str) -> list[Entity]:
    """Return the active entities with a canonical name."""
    return [
        item for item in await store.active_entities() if item.canonical_name == name
    ]


class TestStoredEntityTypes:
    """Stored entity type fields across resolution, merge and unmerge."""

    @pytest.mark.asyncio
    async def test_matching_entity_keeps_stored_type_fields(self, graph_backend):
        """A matching mention keeps the id, label and proposed type."""
        resolver_store = (await graph_backend.open_stores()).resolver_store
        resolver = await make(resolver_store)
        await resolver.resolve(
            [typed(ex("e1", "Acme"), "Unknown", "startup")], WORLD.links
        )

        result = await resolver.resolve(
            [typed(ex("e2", "Acme", episode_id="ep2"), "Unknown", "other")],
            WORLD.links,
        )

        entity = result.entities[0]
        assert (entity.ontology_type_id, entity.type_label, entity.proposed_type) == (
            WORLD.ids["Unknown"],
            "Unknown",
            "startup",
        )

    @pytest.mark.asyncio
    async def test_specific_mention_types_stored_parent(self, graph_backend):
        """A Contract matched by a PurchaseOrder mention takes the child type."""
        resolver_store = (await graph_backend.open_stores()).resolver_store
        resolver = await make(resolver_store)
        await resolver.resolve([typed(ex("e1", "Acme"), "Contract")], WORLD.links)

        result = await resolver.resolve(
            [typed(ex("e2", "Acme", episode_id="ep2"), "PurchaseOrder")],
            WORLD.links,
        )

        entity = result.entities[0]
        assert (entity.ontology_type_id, entity.type_label, entity.proposed_type) == (
            WORLD.ids["PurchaseOrder"],
            "PurchaseOrder",
            "",
        )

    @pytest.mark.asyncio
    async def test_parent_mention_keeps_stored_child_type(self, graph_backend):
        """A PurchaseOrder matched by a Contract mention stays a PurchaseOrder."""
        resolver_store = (await graph_backend.open_stores()).resolver_store
        resolver = await make(resolver_store)
        await resolver.resolve([typed(ex("e1", "Acme"), "PurchaseOrder")], WORLD.links)

        result = await resolver.resolve(
            [typed(ex("e2", "Acme", episode_id="ep2"), "Contract")], WORLD.links
        )

        entity = result.entities[0]
        assert (entity.ontology_type_id, entity.type_label, entity.proposed_type) == (
            WORLD.ids["PurchaseOrder"],
            "PurchaseOrder",
            "",
        )

    @pytest.mark.asyncio
    async def test_typed_mention_promotes_unknown_and_clears_proposed(
        self, graph_backend
    ):
        """A typed mention gives an Unknown entity the type and clears the proposal."""
        resolver_store = (await graph_backend.open_stores()).resolver_store
        resolver = await make(resolver_store)
        await resolver.resolve(
            [typed(ex("e1", "Berlin"), "Unknown", "business")], WORLD.links
        )

        result = await resolver.resolve(
            [typed(ex("e2", "Berlin", episode_id="ep2"), "City")], WORLD.links
        )

        entity = result.entities[0]
        assert (entity.ontology_type_id, entity.type_label, entity.proposed_type) == (
            WORLD.ids["City"],
            "City",
            "",
        )

    @pytest.mark.asyncio
    async def test_unknown_mention_keeps_stored_typed_entity(self, graph_backend):
        """An Unknown mention does not overwrite a stored typed entity."""
        resolver_store = (await graph_backend.open_stores()).resolver_store
        resolver = await make(resolver_store)
        await resolver.resolve([typed(ex("e1", "Berlin"), "City")], WORLD.links)

        result = await resolver.resolve(
            [typed(ex("e2", "Berlin", episode_id="ep2"), "Unknown", "business")],
            WORLD.links,
        )

        entity = result.entities[0]
        assert (entity.ontology_type_id, entity.type_label, entity.proposed_type) == (
            WORLD.ids["City"],
            "City",
            "",
        )

    @pytest.mark.asyncio
    async def test_unrelated_types_with_one_name_stay_separate(self, graph_backend):
        """Company and City with one name become two entities."""
        resolver_store = (await graph_backend.open_stores()).resolver_store
        resolver = await make(resolver_store)

        await resolver.resolve(
            [ex("1", "Hydra", "Company"), ex("2", "Hydra", "City")], WORLD.links
        )

        assert await count_active(resolver_store) == 2

    @pytest.mark.asyncio
    async def test_child_type_with_one_name_becomes_one_entity(self, graph_backend):
        """Person and Employee with one name become one Employee entity."""
        resolver_store = (await graph_backend.open_stores()).resolver_store
        resolver = await make(resolver_store)

        await resolver.resolve(
            [ex("1", "Ada", "Person"), ex("2", "Ada", "Employee")], WORLD.links
        )

        entities = await named(resolver_store, "Ada")
        assert len(entities) == 1
        assert entities[0].type_label == "Employee"
        assert entities[0].ontology_type_id == WORLD.ids["Employee"]

    @pytest.mark.asyncio
    async def test_merged_away_type_id_takes_the_target_label(self, graph_backend):
        """A mention under a merged id matches the target and refreshes its label."""
        ids, links = custom_world(
            EntityDeclaration(label="Company"),
            EntityDeclaration(label="Firm", merged_into="Company"),
        )
        resolver_store = (await graph_backend.open_stores()).resolver_store
        resolver = await make(resolver_store)
        stored = replace(
            ex("e1", "Acme"), type_label="Firm", ontology_type_id=ids["Firm"]
        )
        mention = replace(
            ex("e2", "Acme", episode_id="ep2"),
            type_label="Company",
            ontology_type_id=ids["Company"],
        )
        await resolver.resolve([stored], links)

        result = await resolver.resolve([mention], links)

        entity = result.entities[0]
        assert entity.ontology_type_id == ids["Company"]
        assert entity.type_label == "Company"

    @pytest.mark.asyncio
    async def test_renamed_type_refreshes_the_label_at_the_next_mention(
        self, graph_backend
    ):
        """A type renamed in the file shows its new label on the entity."""
        ids, links = custom_world(
            EntityDeclaration(label="Company", renamed_from=["Employer"])
        )
        resolver_store = (await graph_backend.open_stores()).resolver_store
        resolver = await make(resolver_store)
        stored = replace(
            ex("e1", "Acme"),
            type_label="Employer",
            ontology_type_id=ids["Company"],
        )
        mention = replace(
            ex("e2", "Acme", episode_id="ep2"),
            type_label="Company",
            ontology_type_id=ids["Company"],
        )
        await resolver.resolve([stored], links)

        result = await resolver.resolve([mention], links)

        entity = result.entities[0]
        assert entity.ontology_type_id == ids["Company"]
        assert entity.type_label == "Company"

    @pytest.mark.asyncio
    async def test_known_email_then_blank_then_known_then_new_email(
        self, graph_backend
    ):
        """A stored value wins until a conflicting value creates a namesake."""
        resolver_store = (await graph_backend.open_stores()).resolver_store
        resolver = await make(resolver_store)

        first = await resolver.resolve(
            [typed(ex("1", "Ana", "Person", email="a@x.org"), "Person")], WORLD.links
        )
        second = await resolver.resolve(
            [ex("1", "Ana", "Person", episode_id="ep2")], WORLD.links
        )
        third = await resolver.resolve(
            [ex("1", "Ana", "Person", episode_id="ep3", email="a@x.org")],
            WORLD.links,
        )
        fourth = await resolver.resolve(
            [ex("1", "Ana", "Person", episode_id="ep4", email="b@x.org")],
            WORLD.links,
        )

        assert len(await named(resolver_store, "Ana")) == 2
        assert second.entities[0].entity_id == first.entities[0].entity_id
        assert third.entities[0].entity_id == first.entities[0].entity_id
        assert fourth.entities[0].entity_id != first.entities[0].entity_id
        assert first.entities[0].identity_attrs == {"email": "a@x.org"}

    @pytest.mark.asyncio
    async def test_blank_then_two_different_emails(self, graph_backend):
        """A blank first mention takes the first stated email."""
        resolver_store = (await graph_backend.open_stores()).resolver_store
        resolver = await make(resolver_store)

        first = await resolver.resolve(
            [ex("1", "Ana", "Person", episode_id="ep1")], WORLD.links
        )
        second = await resolver.resolve(
            [ex("1", "Ana", "Person", episode_id="ep2", email="a@x.org")],
            WORLD.links,
        )
        third = await resolver.resolve(
            [ex("1", "Ana", "Person", episode_id="ep3", email="b@x.org")],
            WORLD.links,
        )

        assert second.entities[0].entity_id == first.entities[0].entity_id
        assert third.entities[0].entity_id != first.entities[0].entity_id
        entities = {item.entity_id: item for item in await named(resolver_store, "Ana")}
        assert entities[first.entities[0].entity_id].identity_attrs == {
            "email": "a@x.org"
        }
        assert entities[third.entities[0].entity_id].identity_attrs == {
            "email": "b@x.org"
        }

    @pytest.mark.asyncio
    async def test_user_merge_of_same_type_succeeds(self, graph_backend):
        """A user merges two entities of the same type."""
        resolver_store = (await graph_backend.open_stores()).resolver_store
        resolver = await make(resolver_store)
        await resolver.resolve([ex("1", "Acme", "Company")], WORLD.links)
        await resolver.resolve(
            [ex("1", "Globex", "Company", episode_id="ep2")], WORLD.links
        )
        entities = await named(resolver_store, "Acme") + await named(
            resolver_store, "Globex"
        )
        ids = [item.entity_id for item in entities]

        record = await resolver.merge(WORLD.links, ids[0], ids[1], actor="user")

        survivor = (await resolver_store.get_entities([record.survivor_id]))[0]
        absorbed = (await resolver_store.get_entities([record.absorbed_id]))[0]
        assert survivor.status.value == "active"
        assert absorbed.status.value == "merged"

    @pytest.mark.asyncio
    async def test_non_user_merge_of_distinct_types_is_refused(self, graph_backend):
        """A non-user merge of a Company and a City is refused."""
        resolver_store = (await graph_backend.open_stores()).resolver_store
        resolver = await make(resolver_store)
        await resolver.resolve([ex("1", "Hydra", "Company")], WORLD.links)
        await resolver.resolve(
            [ex("1", "Hydra", "City", episode_id="ep2")], WORLD.links
        )
        entities = await named(resolver_store, "Hydra")
        ids = [item.entity_id for item in entities]

        with pytest.raises(MergeRefusedError) as raised:
            await resolver.merge(WORLD.links, ids[0], ids[1], actor="resolver")

        assert raised.value.reason == "different_type"

    @pytest.mark.asyncio
    async def test_merge_and_unmerge_restore_type_fields(self, graph_backend):
        """A child/parent merge picks the child and unmerge restores the survivor."""
        resolver_store = (await graph_backend.open_stores()).resolver_store
        resolver = await make(resolver_store)
        await resolver.resolve([ex("1", "Acme", "Contract")], WORLD.links)
        await resolver.resolve(
            [ex("1", "Acme Holdings", "PurchaseOrder", episode_id="ep2")],
            WORLD.links,
        )
        contract = (await named(resolver_store, "Acme"))[0]
        order = (await named(resolver_store, "Acme Holdings"))[0]

        record = await resolver.merge(
            WORLD.links, contract.entity_id, order.entity_id, actor="user"
        )
        survivor = (await resolver_store.get_entities([record.survivor_id]))[0]
        assert survivor.ontology_type_id == WORLD.ids["PurchaseOrder"]
        assert survivor.type_label == "PurchaseOrder"

        await resolver.unmerge(record.merge_id)
        restored = (await resolver_store.get_entities([record.survivor_id]))[0]
        assert (
            restored.ontology_type_id,
            restored.type_label,
            restored.proposed_type,
        ) == (
            WORLD.ids["Contract"],
            "Contract",
            "",
        )

    @pytest.mark.asyncio
    async def test_verify_always_sends_a_repeated_person_to_the_verifier(
        self, graph_backend
    ):
        """A VERIFY_ALWAYS override sends a repeated person to the verifier."""
        policy = ResolutionPolicy(
            types={"Person": TypePolicy(mode=ResolutionMode.VERIFY_ALWAYS)}
        )
        models, scripted = oracle_models(lambda _a, _b: True)
        resolver_store = (await graph_backend.open_stores()).resolver_store
        resolver = await make(resolver_store, policy=policy, models=models, inline=True)

        first = await resolver.resolve([ex("1", "Ada", "Person")], WORLD.links)
        second = await resolver.resolve(
            [ex("1", "Ada", "Person", episode_id="ep2")], WORLD.links
        )

        assert scripted.calls
        assert second.entities[0].entity_id == first.entities[0].entity_id

    @pytest.mark.asyncio
    async def test_default_policy_merges_a_repeated_person_without_a_model(
        self, graph_backend
    ):
        """With no override a repeated person is merged without a model call."""
        models, scripted = oracle_models(lambda _a, _b: True)
        resolver_store = (await graph_backend.open_stores()).resolver_store
        resolver = await make(resolver_store, models=models, inline=True)

        first = await resolver.resolve([ex("1", "Ada", "Person")], WORLD.links)
        second = await resolver.resolve(
            [ex("1", "Ada", "Person", episode_id="ep2")], WORLD.links
        )

        assert scripted.calls == []
        assert second.entities[0].entity_id == first.entities[0].entity_id
        assert await count_active(resolver_store) == 1
