"""Paged entity reads attach aliases from bounded per-entity store reads."""

from datetime import UTC, datetime
from types import SimpleNamespace

from ontogr_mem.data_models.entities import Alias, Entity
from ontogr_mem.service.contracts import ListEntities
from ontogr_mem.service.reads import list_entities


NOW = datetime(2026, 11, 18, tzinfo=UTC)


def _entity(entity_id: str, name: str) -> Entity:
    """Build an active entity for the listing fixture."""
    return Entity(
        entity_id=entity_id,
        type_label="Person",
        ontology_type_id=1,
        canonical_name=name,
        strict_key=name.lower(),
        loose_key=name.lower(),
        created_at=NOW,
        updated_at=NOW,
    )


def _alias(entity_id: str, text: str) -> Alias:
    """Build an alias for the listing fixture."""
    return Alias(
        alias_id=f"{entity_id}:{text}",
        entity_id=entity_id,
        text=text,
        strict_key=text.lower(),
        loose_key=text.lower(),
    )


class BoundedResolverStore:
    """Resolver read port that serves aliases only for requested entities."""

    def __init__(self, entities: list[Entity], aliases: dict[str, list[Alias]]):
        """Hold the id-ordered entities and their aliases by entity id."""
        self._entities = entities
        self._aliases = aliases

    async def list_active_entities(self, *, type_ids, after, limit):
        """Return the fixture entities sliced by the requested page limit."""
        return self._entities[:limit]

    async def aliases_of(self, entity_id):
        """Return the stored aliases for one entity."""
        return self._aliases.get(entity_id, [])

    async def all_aliases(self):
        """Refuse the tenant-wide scan that a bounded paged read must not use."""
        raise AssertionError("paged entity reads must not scan all aliases")


class EmptyOntology:
    """Ontology read port that returns an unexamined snapshot."""

    async def catalog(self):
        """Return a catalog value the fixture does not inspect."""
        return SimpleNamespace()


class TestListEntitiesAliasScope:
    """A page carries only the aliases of the entities it returns."""

    async def test_loads_aliases_for_page_entities_only(self):
        """The page keeps aliases in store order and still advances its cursor."""
        entities = [
            _entity("e1", "Ana"),
            _entity("e2", "Bo"),
            _entity("e3", "Cy"),
        ]
        aliases = {
            "e1": [_alias("e1", "Annie"), _alias("e1", "A")],
            "e2": [_alias("e2", "Bobby")],
        }
        store = BoundedResolverStore(entities, aliases)
        runtime = SimpleNamespace(
            stores=SimpleNamespace(resolver_store=store),
            ontology=EmptyOntology(),
        )
        request = ListEntities(tenant="t1", limit=2)

        page = await list_entities("t1", runtime, request)

        assert [item.entity_id for item in page.entities] == ["e1", "e2"]
        assert [item.aliases for item in page.entities] == [
            ["Annie", "A"],
            ["Bobby"],
        ]
        assert page.next_cursor is not None
