"""Contract tests for the resolver store: the in-memory twin and the adapter."""

from dataclasses import replace
from typing import Any, cast

import pytest

from ontogr_mem.data_models.entities import EntityStatus, MergeState
from ontogr_mem.graphstore.cypher import PROPERTY_SCOPE
from ontogr_mem.graphstore.executor import QuerySpec, Rows
from ontogr_mem.graphstore.memory.resolver_store import MemoryResolverStore
from ontogr_mem.graphstore.resolver_store import GraphResolverStore
from ontogr_mem.resolution.resolver_store import (
    MergeRows,
    ResolverStore,
    snapshot_with_intent,
)
from tests.contracts.resolver_store import (
    ResolverStoreContract,
    entity,
    mention,
    merge_plan,
)


class TestMemoryResolverStore(ResolverStoreContract):
    """`ResolverStore` contract on the in-memory twin."""

    @pytest.fixture
    def store(self) -> MemoryResolverStore:
        """Return a store bound to a fresh tenant."""
        return MemoryResolverStore()

    @pytest.fixture
    def other_store(self) -> MemoryResolverStore:
        """Return a second store bound to a different tenant."""
        return MemoryResolverStore()

    async def seed_fact_edge(
        self, store: ResolverStore, entity_id: str, *, live: bool
    ) -> None:
        """Write one live or withdrawn fact edge for the entity."""
        twin = cast(MemoryResolverStore, store)
        twin.seed_fact_edge(
            entity_id,
            "fact-object",
            "RELATED_TO",
            {"status": "active" if live else "withdrawn"},
        )

    async def fact_edge_count(
        self, store: ResolverStore, entity_id: str, *, live: bool
    ) -> int:
        """Count the live or withdrawn fact edges of the entity."""
        return cast(MemoryResolverStore, store).fact_edge_count(entity_id, live=live)


class TestMemoryResolverStoreRecovery:
    """Recovery of a merge that was interrupted on the in-memory twin."""

    async def test_recovery_finishes_an_interrupted_merge(self):
        """The rows of an interrupted merge are written again and settled."""
        store = MemoryResolverStore()
        survivor = entity("e1", name="Acme")
        absorbed = entity("e2", name="Acme Inc")
        await store.write_entity_batch([survivor, absorbed], [])
        moved = mention("e2", "m1")
        await store.write_mentions("ep1", [moved])
        plan = merge_plan("mg1", survivor, absorbed, mentions=(moved,))
        store.seed_merge_record(
            plan.record.model_copy(
                update={
                    "state": MergeState.APPLYING,
                    "snapshot": snapshot_with_intent(
                        plan.record.snapshot, MergeRows.from_merge_plan(plan)
                    ),
                }
            )
        )

        finished = await store.recover_merges()

        assert [record.merge_id for record in finished] == ["mg1"]
        settled = await store.merge_record("mg1")
        assert settled is not None
        assert settled.state is MergeState.APPLIED
        merged = (await store.get_entities(["e2"]))[0]
        assert merged.status is EntityStatus.MERGED
        assert merged.merged_into == "e1"
        assert [m.mention_id for m in await store.mentions_of("e1")] == [
            moved.mention_id
        ]
        assert await store.mentions_of("e2") == []

    async def test_recovery_leaves_a_merge_whose_rows_did_not_move(self):
        """A merge that did not move its rows is not declared applied."""
        store = MemoryResolverStore()
        survivor = entity("e1", name="Acme")
        absorbed = entity("e2", name="Acme Inc")
        await store.write_entity_batch([survivor, absorbed], [])
        moved = mention("e2", "m1")
        await store.write_mentions("ep1", [moved])
        plan = merge_plan("mg1", survivor, absorbed, mentions=(moved,))
        rows = MergeRows.from_merge_plan(plan)
        stuck = replace(rows, mentions=(moved.model_copy(update={"entity_id": "e2"}),))
        store.seed_merge_record(
            plan.record.model_copy(
                update={
                    "state": MergeState.APPLYING,
                    "snapshot": snapshot_with_intent(plan.record.snapshot, stuck),
                }
            )
        )

        finished = await store.recover_merges()

        assert finished == []
        record = await store.merge_record("mg1")
        assert record is not None
        assert record.state is MergeState.APPLYING

    async def test_recovery_leaves_a_record_that_keeps_no_rows(self):
        """A record without the rows of its merge cannot be finished."""
        store = MemoryResolverStore()
        survivor = entity("e1")
        absorbed = entity("e2")
        await store.write_entity_batch([survivor, absorbed], [])
        plan = merge_plan("mg1", survivor, absorbed)
        store.seed_merge_record(
            plan.record.model_copy(update={"state": MergeState.APPLYING})
        )

        finished = await store.recover_merges()

        assert finished == []
        record = await store.merge_record("mg1")
        assert record is not None
        assert record.state is MergeState.APPLYING


class _FakeRunner:
    """A runner that answers the statements of the resolver adapter.

    Attributes:
        statements: The tag and text of every statement the store ran.
    """

    scope = PROPERTY_SCOPE

    def __init__(self, responses: dict[str, list[dict[str, Any]]]) -> None:
        """Answer the statements with the rows of their tag."""
        self._responses = responses
        self.statements: list[tuple[str, str]] = []

    async def run(self, spec: QuerySpec) -> Rows:
        """Return the rows of the statement's tag."""
        self.statements.append((spec.tag, spec.cypher))
        return Rows(list(self._responses.get(spec.tag, [])), {})


class TestGraphResolverStoreRecovery:
    """Recovery of an interrupted merge on the graph adapter."""

    def _runner(self, record: str, **rows: list[dict[str, Any]]) -> _FakeRunner:
        responses = {
            "resolver_merge_applying": [{"value": record}],
            "resolver_get_entities": [],
            "resolver_mentions_of": [],
            "resolver_aliases_of": [],
        }
        responses.update(rows)
        return _FakeRunner(responses)

    async def test_recovery_writes_the_rows_again_and_settles_the_record(self):
        """The rows of an interrupted merge are written again and settled."""
        survivor = entity("e1", name="Acme")
        absorbed = entity("e2", name="Acme Inc")
        plan = merge_plan("mg1", survivor, absorbed)
        record = plan.record.model_copy(
            update={
                "state": MergeState.APPLYING,
                "snapshot": snapshot_with_intent(
                    plan.record.snapshot, MergeRows.from_merge_plan(plan)
                ),
            }
        )
        runner = self._runner(
            record.model_dump_json(),
            resolver_get_entities=[{"value": plan.absorbed.model_dump_json()}],
        )
        store = GraphResolverStore(runner)

        finished = await store.recover_merges()

        assert [settled.merge_id for settled in finished] == ["mg1"]
        assert finished[0].state is MergeState.APPLIED
        tags = [tag for tag, _ in runner.statements]
        assert "resolver_entities_write" in tags
        assert "resolver_merge_write" in tags

    async def test_recovery_leaves_a_merge_whose_rows_did_not_move(self):
        """A merge whose rows are still on the absorbed entity is not settled."""
        survivor = entity("e1", name="Acme")
        absorbed = entity("e2", name="Acme Inc")
        moved = mention("e2", "m1")
        plan = merge_plan("mg1", survivor, absorbed, mentions=(moved,))
        record = plan.record.model_copy(
            update={
                "state": MergeState.APPLYING,
                "snapshot": snapshot_with_intent(
                    plan.record.snapshot, MergeRows.from_merge_plan(plan)
                ),
            }
        )
        runner = self._runner(
            record.model_dump_json(),
            resolver_get_entities=[{"value": plan.absorbed.model_dump_json()}],
            resolver_mentions_of=[{"value": moved.model_dump_json()}],
        )
        store = GraphResolverStore(runner)

        finished = await store.recover_merges()

        assert finished == []
        assert "resolver_merge_write" not in [tag for tag, _ in runner.statements]

    async def test_recovery_leaves_a_record_that_keeps_no_rows(self):
        """A record without the rows of its merge is not settled."""
        survivor = entity("e1")
        absorbed = entity("e2")
        plan = merge_plan("mg1", survivor, absorbed)
        record = plan.record.model_copy(update={"state": MergeState.APPLYING})
        runner = self._runner(record.model_dump_json())
        store = GraphResolverStore(runner)

        finished = await store.recover_merges()

        assert finished == []
        assert "resolver_merge_write" not in [tag for tag, _ in runner.statements]
