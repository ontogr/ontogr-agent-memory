"""Catalog reads over active entities with and without a stored type filter."""

from datetime import UTC, datetime

from ontogr_mem.data_models.entities import Entity
from ontogr_mem.graphstore.cypher import TenantScope
from ontogr_mem.graphstore.executor import QuerySpec, Rows
from ontogr_mem.graphstore.resolver.entities import ResolverEntities


NOW = datetime(2026, 10, 23, 12, 0, tzinfo=UTC)


def _entity(entity_id: str, type_id: int) -> Entity:
    """Build an active entity with a stored type id."""
    return Entity(
        entity_id=entity_id,
        type_label="",
        ontology_type_id=type_id,
        canonical_name=entity_id,
        strict_key=entity_id,
        loose_key=entity_id,
        created_at=NOW,
        updated_at=NOW,
    )


class _ScalarPageRunner:
    """Serve one catalog page and count store queries."""

    def __init__(self, entities: list[Entity]) -> None:
        """Keep the entities this runner will return."""
        self._entities = entities
        self.query_count = 0

    async def run(self, spec: QuerySpec) -> Rows:
        """Return rows past the cursor, filtered by type when the query sets one."""
        self.query_count += 1
        raw_types = spec.params.get("types")
        type_ids = None if raw_types is None else set(raw_types)
        after = spec.params.get("after") or ""
        size = spec.params["limit"]
        rows = [
            {"value": entity.model_dump_json()}
            for entity in self._entities
            if entity.entity_id > after
            and (type_ids is None or entity.ontology_type_id in type_ids)
        ]
        return Rows(rows=rows[:size], stats={})


class TestListActiveEntities:
    """Page active entities using their stored type scalar."""

    async def test_filters_and_pages_rows_with_one_query_per_page(self) -> None:
        """Skip other types and page by id without rescanning the tenant."""
        entities = [_entity("a", 2), _entity("b", 3), _entity("c", 2), _entity("d", 2)]
        runner = _ScalarPageRunner(entities)
        store = ResolverEntities(runner, TenantScope())

        first = await store.list_active_entities(
            type_ids=frozenset({2}), after=None, limit=2
        )
        second = await store.list_active_entities(
            type_ids=frozenset({2}), after=first[-1].entity_id, limit=2
        )

        assert [item.entity_id for item in first] == ["a", "c"]
        assert [item.entity_id for item in second] == ["d"]
        assert runner.query_count == 2

    async def test_pages_every_type_when_no_type_filter_is_set(self) -> None:
        """An omitted type filter pages every stored type by id."""
        entities = [_entity("a", 2), _entity("b", 3), _entity("c", 2), _entity("d", 2)]
        runner = _ScalarPageRunner(entities)
        store = ResolverEntities(runner, TenantScope())

        first = await store.list_active_entities(type_ids=None, after=None, limit=2)
        second = await store.list_active_entities(
            type_ids=None, after=first[-1].entity_id, limit=2
        )

        assert [item.entity_id for item in first] == ["a", "b"]
        assert [item.entity_id for item in second] == ["c", "d"]
        assert runner.query_count == 2
