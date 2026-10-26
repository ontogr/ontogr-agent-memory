"""Contract every graph store backend must satisfy.

A backend test module subclasses `GraphStoreContract` and provides the fixtures
named in the class body. The class has no `Test` prefix, so pytest collects only
its subclasses.
"""

import asyncio
from datetime import UTC, datetime
from typing import Any

import pytest

from ontogr_mem.graphstore import (
    ConstraintViolationError,
    Edge,
    EdgeSelector,
    GraphHandle,
    GraphStore,
    Guard,
    Node,
    NodeSelector,
    QuerySpec,
    SchemaDriftError,
    StoreQueryError,
    ValidationError,
)
from tests.support.graph_schema import DRIFT_PART, TEST_PART
from tests.support.tenants import new_tenant


def entities(*ids, **props):
    """Build `Entity` nodes with a name each."""
    return [Node(i, "Entity", {"name": f"name {i}", **props}) for i in ids]


class GraphStoreContract:
    """Behavior shared by every `GraphStore`.

    Subclasses define these fixtures: `store`, `handle` (opened with
    `TEST_PART`), `store_factory` (an async callable that returns a new store of
    the same backend), `unreachable_store`, and the async methods
    `drop_range_index` and `drop_unique_constraint`.
    """

    async def drop_range_index(self, store: GraphStore, handle: GraphHandle) -> None:
        """Drop the `Drift.name` range index behind the store's back."""
        raise NotImplementedError

    async def drop_unique_constraint(
        self, store: GraphStore, handle: GraphHandle
    ) -> None:
        """Drop the `Drift.email` unique constraint behind the store's back."""
        raise NotImplementedError

    def create_cypher(self, node_id: str) -> str:
        """Return a statement that creates one `Entity` with `node_id`."""
        return f"CREATE (:Entity {{id: '{node_id}'}})"

    def count_cypher(self) -> str:
        """Return a statement that counts the `Entity` nodes visible to a runner."""
        return "MATCH (n:Entity) RETURN count(n) AS c"

    def guard_props_cypher(self, node_id: str) -> str:
        """Return a statement that reads the guard node's stored properties."""
        return f"MATCH (n:Slot {{id: '{node_id}'}}) RETURN properties(n) AS props"

    async def guard_props(
        self, store: GraphStore, handle: GraphHandle, node_id: str
    ) -> dict[str, Any]:
        """Return the guard node's stored properties, reserved names included."""
        rows = await store.runner(handle).run(
            QuerySpec(
                self.guard_props_cypher(node_id),
                read_only=True,
                idempotent=True,
                tag="guard_props",
            )
        )
        return dict(rows.rows[0]["props"]) if rows.rows else {}

    # Tenancy and schema

    async def test_reopen_returns_the_same_handle_and_keeps_data(self, store, handle):
        """Opening a provisioned tenant again changes nothing."""
        await store.upsert_nodes(handle, entities("a"))

        again = await store.open(handle.tenant, [TEST_PART])

        assert again == handle
        assert [r.id for r in await store.get_nodes(handle, "Entity", ["a"])] == ["a"]
        assert (await store.stats(handle)).drift == ()

    async def test_concurrent_opens_of_a_new_tenant_all_succeed(
        self, store, store_factory
    ):
        """Separate clients provisioning the same new tenant do not fail."""
        other = await store_factory()
        tenant = new_tenant()
        try:
            handles = await asyncio.gather(
                store.open(tenant, [TEST_PART]),
                other.open(tenant, [TEST_PART]),
                store.open(tenant, [TEST_PART]),
            )
            assert len({h.graph for h in handles}) == 1
            assert (await store.stats(handles[0])).drift == ()
        finally:
            await store.erase_tenant(tenant)
            await other.close()

    @pytest.mark.parametrize("tenant", ["", "x" * 61, "a b", "a.b"])
    async def test_open_rejects_invalid_tenant(self, store, tenant):
        """A bad tenant id fails before any query."""
        with pytest.raises(ValidationError):
            await store.open(tenant)

    async def test_dropped_index_is_reported_and_reprovision_repairs_it(self, store):
        """Drift shows in stats, verify raises, provision repairs."""
        handle = await store.open(new_tenant(), [DRIFT_PART])
        try:
            await store.verify(handle)
            await self.drop_range_index(store, handle)

            drift = (await store.stats(handle)).drift
            assert any("Drift.name" in line for line in drift)
            with pytest.raises(SchemaDriftError):
                await store.verify(handle)

            await store.provision(handle)

            assert (await store.stats(handle)).drift == ()
            await store.verify(handle)
        finally:
            await store.erase_tenant(handle.tenant)

    async def test_dropped_constraint_is_reported_and_reprovision_repairs_it(
        self, store
    ):
        """A missing unique constraint counts as drift too."""
        handle = await store.open(new_tenant(), [DRIFT_PART])
        try:
            await self.drop_unique_constraint(store, handle)

            drift = (await store.stats(handle)).drift
            assert any("Drift.email" in line for line in drift)

            await store.provision(handle)

            assert (await store.stats(handle)).drift == ()
        finally:
            await store.erase_tenant(handle.tenant)

    async def test_erase_removes_the_graph_and_a_second_erase_is_a_no_op(self, store):
        """Erasing twice is safe and a reopened tenant starts empty."""
        tenant = new_tenant()
        handle = await store.open(tenant, [TEST_PART])
        await store.upsert_nodes(handle, entities("a"))

        await store.erase_tenant(tenant)
        await store.erase_tenant(tenant)

        reopened = await store.open(tenant, [TEST_PART])
        assert await store.get_nodes(reopened, "Entity", ["a"]) == []
        await store.erase_tenant(tenant)

    async def test_health_reports_ok_with_latency(self, store):
        """A reachable server reports ok and a closed breaker."""
        health = await store.health()

        assert health.ok
        assert health.breaker == "closed"
        assert health.latency_ms >= 0

    async def test_health_of_a_stopped_server_reports_not_ok(self, unreachable_store):
        """An unreachable server is reported, not raised."""
        health = await unreachable_store.health()

        assert not health.ok
        assert health.detail

    # Nodes

    async def test_upsert_nodes_is_idempotent(self, store, handle):
        """Writing the same nodes again creates nothing new."""
        first = await store.upsert_nodes(handle, entities("a", "b"))
        second = await store.upsert_nodes(handle, entities("a", "b"))

        assert (first.complete, first.created) == (True, 2)
        assert (second.complete, second.created) == (True, 0)
        assert (await store.stats(handle)).labels["Entity"] == 2

    @pytest.mark.parametrize("name", ["_tenant", "_lock"])
    async def test_reserved_properties_are_refused_on_every_backend(
        self, store, handle, name
    ):
        """Backend-neutral writes reject properties reserved by the store."""
        with pytest.raises(ValidationError):
            await store.upsert_nodes(handle, [Node("reserved", "Entity", {name: "x"})])

    async def test_upsert_merges_properties_and_keeps_existing_ones(
        self, store, handle
    ):
        """A second write adds and overwrites only the properties it names."""
        await store.upsert_nodes(
            handle, [Node("a", "Entity", {"name": "A", "summary": "s"})]
        )
        await store.upsert_nodes(handle, [Node("a", "Entity", {"name": "A2"})])

        (row,) = await store.get_nodes(handle, "Entity", ["a"])

        assert dict(row.props) == {"id": "a", "name": "A2", "summary": "s"}

    async def test_replace_drops_properties_that_are_not_given(self, store, handle):
        """`replace=True` makes the node hold exactly the given properties."""
        await store.upsert_nodes(
            handle, [Node("a", "Entity", {"name": "A", "summary": "s"})]
        )
        await store.upsert_nodes(
            handle, [Node("a", "Entity", {"name": "A2"})], replace=True
        )

        (row,) = await store.get_nodes(handle, "Entity", ["a"])

        assert dict(row.props) == {"id": "a", "name": "A2"}

    async def test_labels_may_grow_on_one_node(self, store, handle):
        """Extra labels are added to the same node, never to a copy."""
        await store.upsert_nodes(handle, [Node("a", "Entity", {"name": "A"})])
        await store.upsert_nodes(handle, [Node("a", "Entity", {}, ("Person",))])
        await store.upsert_nodes(handle, [Node("a", "Entity", {}, ("Org", "Person"))])

        (row,) = await store.get_nodes(handle, "Entity", ["a"])

        assert set(row.labels) == {"Entity", "Person", "Org"}
        assert (await store.stats(handle)).labels["Entity"] == 1

    async def test_node_without_id_is_refused(self, store, handle):
        """An empty id never reaches the server."""
        with pytest.raises(ValidationError):
            await store.upsert_nodes(handle, [Node("", "Entity", {"name": "x"})])

    async def test_unique_id_constraint_refuses_a_duplicate_written_out_of_band(
        self, store, handle
    ):
        """The server, not only the client, keeps `id` unique per label."""
        runner = store.runner(handle)
        create = QuerySpec(self.create_cypher("dup"))
        await runner.run(create)

        with pytest.raises(ConstraintViolationError):
            await runner.run(create)

    async def test_same_id_in_two_tenants_never_crosses(self, store, handle):
        """Tenants share nothing, not even ids."""
        other = await store.open(new_tenant(), [TEST_PART])
        try:
            await store.upsert_nodes(handle, [Node("a", "Entity", {"name": "mine"})])
            await store.upsert_nodes(other, [Node("a", "Entity", {"name": "theirs"})])

            (mine,) = await store.get_nodes(handle, "Entity", ["a"])
            (theirs,) = await store.get_nodes(other, "Entity", ["a"])

            assert mine.props["name"] == "mine"
            assert theirs.props["name"] == "theirs"
            await store.upsert_nodes(other, entities("only-theirs"))
            assert await store.get_nodes(handle, "Entity", ["only-theirs"]) == []
        finally:
            await store.erase_tenant(other.tenant)

    async def test_large_batch_succeeds(self, store, handle):
        """A 1,000 node write is split into chunks and all of it lands."""
        result = await store.upsert_nodes(
            handle, [Node(f"n{i:04d}", "Item", {"rank": i}) for i in range(1000)]
        )

        assert result.complete
        assert result.created == 1000
        assert (await store.stats(handle)).labels["Item"] == 1000

    async def test_one_rejected_item_does_not_lose_the_others(self, store, handle):
        """The server rejects one item; the rest are written and it is reported."""
        await store.upsert_nodes(handle, [Node("first", "Entity", {"email": "x@y"})])
        batch = [
            Node("a", "Entity", {"email": "a@y"}),
            Node("b", "Entity", {"email": "x@y"}),
            Node("c", "Entity", {"email": "c@y"}),
        ]

        result = await store.upsert_nodes(handle, batch)

        assert [e.index for e in result.errors] == [1]
        assert result.matched == 2
        assert result.fell_back_to_single
        got = {r.id for r in await store.get_nodes(handle, "Entity", ["a", "b", "c"])}
        assert got == {"a", "c"}

    async def test_property_cleaning(self, store, handle):
        """Mappings become JSON, nulls are omitted, naive datetimes are refused."""
        moment = datetime(2026, 1, 2, 3, 4, 5, 6, tzinfo=UTC)
        await store.upsert_nodes(
            handle,
            [
                Node(
                    "a",
                    "Entity",
                    {"meta": {"k": 1}, "gone": None, "at": moment, "tags": ["x"]},
                )
            ],
        )

        (row,) = await store.get_nodes(handle, "Entity", ["a"])

        assert row.props["meta_json"] == '{"k": 1}'
        assert "gone" not in row.props
        assert row.props["at"] == 1767323045000006
        assert row.props["tags"] == ["x"]
        with pytest.raises(ValidationError):
            await store.upsert_nodes(
                handle, [Node("b", "Entity", {"at": datetime(2026, 1, 1)})]
            )
        assert await store.get_nodes(handle, "Entity", ["b"]) == []

    # Edges

    async def test_edge_with_missing_endpoint_is_reported_and_creates_no_ghost(
        self, store, handle
    ):
        """The good edge is written, the bad one is listed by index."""
        await store.upsert_nodes(handle, entities("a", "b"))

        result = await store.upsert_edges(
            handle,
            [
                Edge("Entity", "a", "Entity", "b", "MENTIONS"),
                Edge("Entity", "a", "Entity", "zzz", "MENTIONS"),
            ],
        )

        assert (result.attempted, result.matched, result.missing) == (2, 1, (1,))
        assert not result.complete
        stats = await store.stats(handle)
        assert stats.labels["Entity"] == 2
        assert stats.edge_count == 1

    async def test_edge_identity_includes_declared_keys(self, store, handle):
        """Same endpoints and type are one edge unless the declared key differs."""
        await store.upsert_nodes(handle, entities("a", "b"))

        def fact(key):
            return Edge(
                "Entity", "a", "Entity", "b", "FACT", {"fact_key": key, "fact": key}
            )

        await store.upsert_edges(handle, [fact("k1")])
        again = await store.upsert_edges(handle, [fact("k1")])
        assert again.created == 0
        await store.upsert_edges(handle, [fact("k2")])
        for _ in range(2):
            await store.upsert_edges(
                handle, [Edge("Entity", "a", "Entity", "b", "MENTIONS")]
            )

        stats = await store.stats(handle)
        assert dict(stats.relationship_types) == {"FACT": 2, "MENTIONS": 1}

    # Deletes

    async def test_delete_ignores_missing_items_and_counts_what_was_removed(
        self, store, handle
    ):
        """Counts reflect real removals only."""
        await store.upsert_nodes(handle, entities("a", "b"))
        await store.upsert_edges(
            handle, [Edge("Entity", "a", "Entity", "b", "MENTIONS")]
        )

        assert await store.delete(handle, NodeSelector("Entity", ["nope"])) == 0
        edges = EdgeSelector("MENTIONS", "Entity", "Entity", [("a", "b"), ("x", "y")])
        assert await store.delete(handle, edges) == 1
        assert await store.delete(handle, NodeSelector("Entity", ["a", "missing"])) == 1
        rows = await store.get_nodes(handle, "Entity", ["a", "b"])
        assert [r.id for r in rows] == ["b"]

    async def test_delete_without_detach_refuses_a_connected_node(self, store, handle):
        """Nothing is removed when a selected node still has edges."""
        await store.upsert_nodes(handle, entities("a", "b"))
        await store.upsert_edges(
            handle, [Edge("Entity", "a", "Entity", "b", "MENTIONS")]
        )

        with pytest.raises(StoreQueryError):
            await store.delete(handle, NodeSelector("Entity", ["a"]), detach=False)

        assert (await store.stats(handle)).edge_count == 1
        assert len(await store.get_nodes(handle, "Entity", ["a"])) == 1

    async def test_delete_with_detach_removes_the_edges_too(self, store, handle):
        """The default delete takes attached edges with the node."""
        await store.upsert_nodes(handle, entities("a", "b"))
        await store.upsert_edges(
            handle, [Edge("Entity", "a", "Entity", "b", "MENTIONS")]
        )

        await store.delete(handle, NodeSelector("Entity", ["a"]))

        assert (await store.stats(handle)).edge_count == 0

    # Guarded write

    @staticmethod
    def slot_guard(revision=0):
        """Return a guard that moves the `s` slot from `revision` to the next."""
        return Guard(
            "Slot",
            "s",
            expect={"revision": revision},
            set={"revision": revision + 1},
            create_with={"revision": 0},
        )

    async def test_guarded_write_applies_nodes_edges_and_guard_update(
        self, store, handle
    ):
        """A holding guard writes everything and returns True."""
        await store.upsert_nodes(handle, [Node("s", "Slot", {"revision": 0})])
        await store.upsert_nodes(handle, entities("a"))
        nodes = [Node("l1", "Ledger", {"note": "first"})]
        edges = [Edge("Entity", "a", "Ledger", "l1", "MENTIONS")]

        applied = await store.guarded_write(handle, self.slot_guard(), nodes, edges)

        assert applied is True
        (slot,) = await store.get_nodes(handle, "Slot", ["s"])
        assert slot.props["revision"] == 1
        assert [r.id for r in await store.get_nodes(handle, "Ledger", ["l1"])] == ["l1"]
        assert (await store.stats(handle)).relationship_types["MENTIONS"] == 1

    async def test_failing_guard_writes_nothing(self, store, handle):
        """A stale expectation returns False and leaves no trace."""
        await store.upsert_nodes(handle, [Node("s", "Slot", {"revision": 5})])
        await store.upsert_nodes(handle, entities("a"))

        applied = await store.guarded_write(
            handle,
            self.slot_guard(revision=0),
            [Node("l1", "Ledger", {})],
            [Edge("Entity", "a", "Ledger", "l1", "MENTIONS")],
        )

        assert applied is False
        assert await store.get_nodes(handle, "Ledger", ["l1"]) == []
        (slot,) = await store.get_nodes(handle, "Slot", ["s"])
        assert slot.props["revision"] == 5
        assert (await store.stats(handle)).edge_count == 0

    async def test_refused_guard_leaves_the_guard_node_byte_identical(
        self, store, handle
    ):
        """A stale expectation changes nothing about the guard node."""
        await store.upsert_nodes(handle, [Node("s", "Slot", {"revision": 5})])
        before = await self.guard_props(store, handle, "s")

        applied = await store.guarded_write(
            handle, self.slot_guard(revision=0), [Node("l1", "Ledger", {})]
        )

        assert applied is False
        assert await self.guard_props(store, handle, "s") == before
        assert await store.get_nodes(handle, "Ledger", ["l1"]) == []

    async def test_refused_guard_with_create_with_leaves_no_guard_node(
        self, store, handle
    ):
        """A guard that would not hold does not create the guard node."""
        guard = Guard(
            "Slot",
            "new",
            expect={"revision": 3},
            set={"revision": 4},
            create_with={"revision": 0},
        )

        applied = await store.guarded_write(handle, guard, [Node("l1", "Ledger", {})])

        assert applied is False
        assert await store.get_nodes(handle, "Slot", ["new"]) == []
        assert await store.get_nodes(handle, "Ledger", ["l1"]) == []

    async def test_guard_write_changes_only_the_expected_properties(
        self, store, handle
    ):
        """A holding guard writes the expected properties and nothing else."""
        await store.upsert_nodes(handle, [Node("s", "Slot", {"revision": 0})])
        before = await self.guard_props(store, handle, "s")

        applied = await store.guarded_write(handle, self.slot_guard())

        assert applied is True
        assert await self.guard_props(store, handle, "s") == {
            **before,
            "revision": 1,
        }

    async def test_guard_without_create_with_fails_when_the_node_is_missing(
        self, store, handle
    ):
        """Without `create_with` a missing guard node means the guard fails."""
        guard = Guard("Slot", "missing", expect={"revision": 0}, set={"revision": 1})

        applied = await store.guarded_write(handle, guard, [Node("l1", "Ledger", {})])

        assert applied is False
        assert await store.get_nodes(handle, "Ledger", ["l1"]) == []

    async def test_create_with_creates_the_missing_guard_node_then_applies(
        self, store, handle
    ):
        """A first caller creates the node and passes the guard."""
        applied = await store.guarded_write(
            handle, self.slot_guard(), [Node("l1", "Ledger", {})]
        )

        assert applied is True
        (slot,) = await store.get_nodes(handle, "Slot", ["s"])
        assert slot.props["revision"] == 1

    async def test_guard_expectation_must_be_a_scalar(self, store, handle):
        """A mapping expectation is refused, not compared against the wrong key."""
        guard = Guard("Slot", "s", expect={"meta": {"a": 1}}, set={"revision": 1})

        with pytest.raises(ValidationError):
            await store.guarded_write(handle, guard, [Node("l1", "Ledger", {})])

        assert await store.get_nodes(handle, "Ledger", ["l1"]) == []

    async def test_guard_expecting_an_absent_property(self, store, handle):
        """`None` in `expect` means the property must not exist."""
        await store.upsert_nodes(handle, [Node("s", "Slot", {})])
        guard = Guard("Slot", "s", expect={"owner": None}, set={"owner": "me"})

        first = await store.guarded_write(handle, guard)
        second = await store.guarded_write(handle, guard)

        assert (first, second) == (True, False)

    async def test_thirty_callers_with_one_expected_revision_have_one_winner(
        self, store, handle
    ):
        """Exactly one concurrent caller passes the guard and writes its ledger."""
        await store.upsert_nodes(handle, [Node("s", "Slot", {"revision": 0})])

        results = await asyncio.gather(
            *[
                store.guarded_write(
                    handle, self.slot_guard(), [Node(f"ledger-{i}", "Ledger", {})]
                )
                for i in range(30)
            ]
        )

        assert sum(results) == 1
        stats = await store.stats(handle)
        assert stats.labels["Ledger"] == 1
        (slot,) = await store.get_nodes(handle, "Slot", ["s"])
        assert slot.props["revision"] == 1

    async def test_thirty_tenants_each_have_one_winner_and_no_cross_effects(
        self, store
    ):
        """Concurrent guarded writes on many tenants stay inside each tenant."""
        handles = await asyncio.gather(
            *[store.open(new_tenant(), [TEST_PART]) for _ in range(30)]
        )
        try:
            results = await asyncio.gather(
                *[
                    store.guarded_write(
                        h,
                        self.slot_guard(),
                        [Node(f"ledger-{h.tenant}-{k}", "Ledger", {})],
                    )
                    for h in handles
                    for k in range(3)
                ]
            )

            assert sum(results) == 30
            for handle in handles:
                ledgers = await store.get_nodes(
                    handle, "Ledger", [f"ledger-{handle.tenant}-{k}" for k in range(3)]
                )
                assert len(ledgers) == 1
                assert (await store.stats(handle)).labels["Ledger"] == 1
        finally:
            await asyncio.gather(*[store.erase_tenant(h.tenant) for h in handles])

    # Scoped runner

    async def test_runner_returns_rows_and_write_counters(self, store, handle):
        """A write reports what it created and a read returns rows."""
        runner = store.runner(handle)

        written = await runner.run(QuerySpec(self.create_cypher("r1")))
        read = await runner.run(
            QuerySpec("RETURN 1 AS x", read_only=True, idempotent=True)
        )

        assert written.stats["nodes_created"] == 1
        assert read.rows == [{"x": 1}]

    async def test_runner_read_only_spec_rejects_writes(self, store, handle):
        """A write cannot run on the read-only path."""
        runner = store.runner(handle)

        with pytest.raises(StoreQueryError):
            await runner.run(QuerySpec(self.create_cypher("r2"), read_only=True))

        assert await store.get_nodes(handle, "Entity", ["r2"]) == []

    async def test_runner_is_bound_to_its_tenant(self, store, handle):
        """Cypher from one tenant's runner cannot see another tenant's data."""
        other = await store.open(new_tenant(), [TEST_PART])
        try:
            await store.upsert_nodes(other, entities("theirs"))

            rows = await store.runner(handle).run(
                QuerySpec(self.count_cypher(), read_only=True, idempotent=True)
            )

            assert rows.scalar("c") == 0
        finally:
            await store.erase_tenant(other.tenant)
