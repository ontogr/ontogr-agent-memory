"""Catalog reads over active entities filtered by their stored type scalar."""

from datetime import UTC, datetime

from ontogr_mem.data_models.entities import Entity
from ontogr_mem.graphstore.cypher import TenantScope
from ontogr_mem.graphstore.executor import QuerySpec, Rows
from ontogr_mem.graphstore.resolver.entities import ResolverEntities


NOW = datetime(2026, 10, 23, 12, 0, tzinfo=UTC)


class _ScalarPageRunner:
    """Serve one scalar-filtered catalog page and count store queries."""

    def __init__(self, entities: list[Entity]) -> None:
        """Keep the entities this runner will return."""
        self._entities = entities
        self.query_count = 0

    async def run(self, spec: QuerySpec) -> Rows:
        """Return rows matching the query's stored type id and id cursor."""
        self.query_count += 1
        type_ids = set(spec.params.get("types") or ())
        after = spec.params.get("after") or ""
        size = spec.params["limit"]
        rows = [
            {"value": entity.model_dump_json()}
            for entity in self._entities
            if entity.entity_id > after and entity.ontology_type_id in type_ids
        ]
        return Rows(rows=rows[:size], stats={})


class TestListActiveEntities:
    """Page active entities using their stored type scalar."""

    async def test_filters_and_pages_rows_with_one_query_per_page(self) -> None:
        """Skip other types and page by id without rescanning the tenant."""

        def make(entity_id: str, type_id: int) -> Entity:
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

        entities = [make("a", 2), make("b", 3), make("c", 2), make("d", 2)]
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
