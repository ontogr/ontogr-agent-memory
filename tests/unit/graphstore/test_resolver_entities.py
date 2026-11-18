"""Catalog reads over entity rows that predate the type-id scalar."""

from datetime import UTC, datetime

from ontogr_mem.data_models.entities import Entity
from ontogr_mem.graphstore.cypher import TenantScope
from ontogr_mem.graphstore.executor import QuerySpec, Rows
from ontogr_mem.graphstore.resolver.entities import ResolverEntities


class _JsonOnlyRunner:
    """Answer catalog reads with rows whose type id lives only in the JSON."""

    def __init__(self, entities: list[Entity]) -> None:
        """Keep the entities this runner will return."""
        self._entities = entities

    async def run(self, spec: QuerySpec) -> Rows:
        """Return the active rows after the cursor, up to the requested size."""
        after = spec.params.get("after") or ""
        size = spec.params["limit"]
        rows = [
            {"value": entity.model_dump_json()}
            for entity in self._entities
            if entity.entity_id > after
        ]
        return Rows(rows=rows[:size], stats={})


class TestListActiveEntities:
    """Filter and page active entities using their stored JSON type id."""

    async def test_filters_and_pages_rows_without_a_type_scalar(self) -> None:
        """Skip other types and continue paging past them in small windows."""
        now = datetime.now(UTC)

        def make(entity_id: str, type_id: int) -> Entity:
            return Entity(
                entity_id=entity_id,
                type_label="",
                ontology_type_id=type_id,
                canonical_name=entity_id,
                strict_key=entity_id,
                loose_key=entity_id,
                created_at=now,
                updated_at=now,
            )

        entities = [make("a", 2), make("b", 3), make("c", 2), make("d", 2)]
        store = ResolverEntities(_JsonOnlyRunner(entities), TenantScope())

        first = await store.list_active_entities(
            type_ids=frozenset({2}), after=None, limit=2
        )
        second = await store.list_active_entities(
            type_ids=frozenset({2}), after=first[-1].entity_id, limit=2
        )

        assert [item.entity_id for item in first] == ["a", "c"]
        assert [item.entity_id for item in second] == ["d"]
