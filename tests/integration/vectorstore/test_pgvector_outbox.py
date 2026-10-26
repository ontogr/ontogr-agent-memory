"""The outbox contract, reconciler and commit hook against Postgres."""

import asyncio
from collections.abc import AsyncGenerator, Sequence
from types import SimpleNamespace

import pytest

from ontogr_mem.vectorstore import (
    EmbedItem,
    PermanentEmbeddingError,
    SearchFilters,
    TemporalFilter,
    Validity,
    VectorKind,
    VectorStore,
)
from ontogr_mem.vectorstore.pgvector import (
    OutboxOp,
    PgBackend,
    PgVectorSettings,
    ReconcileConfig,
    Reconciler,
    delete_entry,
    drop_tenant,
    enqueue_commit,
    entries_for_commit,
    episode_ready,
    upsert_entry,
    validity_entry,
)
from tests.contracts.outbox import OutboxContract
from tests.contracts.vector_store import edge_item, node_item
from tests.support.embedder import MockEmbedder, gateway
from tests.support.pgvector import PgEnv, open_pg_env
from tests.support.vectors import A, fixed_embedder


NODE, EDGE = VectorKind.NODE, VectorKind.EDGE
ACTIVE = Validity("active", 100, 100)
CLOSED = Validity("closed", 100, 100, valid_to_us=200)


class OnceFailingEmbedder(MockEmbedder):
    """Fails the first batch, then behaves like `MockEmbedder`."""

    def __init__(self) -> None:
        """Create the embedder."""
        super().__init__()
        self.failed = False

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Raise a permanent error once, then embed."""
        if not self.failed:
            self.failed = True
            msg = "401"
            raise PermanentEmbeddingError(msg)
        return await super().embed_batch(texts)


class BrokenBackend(PgBackend):
    """A backend whose writes always fail."""

    async def write(self, kind, rows):  # type: ignore[override]
        """Fail the way an unreachable store would."""
        msg = "store down"
        raise RuntimeError(msg)


async def drain_all(reconciler: Reconciler) -> None:
    """Drain until the queue is empty."""
    for _ in range(10):
        report = await reconciler.drain()
        if report.claimed == 0:
            return


def commit(fact_id: str = "f1", **overrides: object) -> SimpleNamespace:
    """Return a commit payload for a fact whose text is `a`."""
    base: dict[str, object] = {
        "fact_id": fact_id,
        "is_new": True,
        "subject_id": "s1",
        "object_id": "o1",
        "validity": ACTIVE,
        "statement": "a",
        "subject_name": "",
        "relation_name": "",
        "object_name": "",
        "relation_type_id": None,
        "episode_id": None,
        "deleted": False,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class TestPgOutbox(OutboxContract):
    """`PgOutbox` satisfies the outbox contract."""

    @pytest.fixture
    async def make_outbox(self, pg_pool) -> AsyncGenerator[object]:
        """Return an async factory for outboxes on fresh tenants."""
        tenants: list[str] = []

        async def make():
            env = await open_pg_env(pg_pool)
            tenants.append(env.tenant)
            return env.outbox

        yield make
        for tenant in tenants:
            await drop_tenant(pg_pool, tenant)


class TestReconciler:
    """The three crash points, sweeps and the episode gate on Postgres."""

    @pytest.fixture
    async def make_env(self, pg_pool) -> AsyncGenerator[object]:
        """Return an async factory for fresh tenant environments."""
        tenants: list[str] = []

        async def make(
            embedder: MockEmbedder | None = None,
            settings: PgVectorSettings | None = None,
        ) -> PgEnv:
            env = await open_pg_env(pg_pool, embedder, settings=settings)
            tenants.append(env.tenant)
            return env

        yield make
        for tenant in tenants:
            await drop_tenant(pg_pool, tenant)

    async def test_crash_before_graph_commit_leaves_no_vector(self, make_env):
        """An old entry whose graph object never appeared is dropped."""
        env = await make_env()
        await env.outbox.enqueue([upsert_entry(NODE, EmbedItem("ghost", "a"))])
        reconciler = Reconciler(
            env.store, env.outbox, env.graph, ReconcileConfig(grace_s=0)
        )

        report = await reconciler.drain()

        assert report.dropped_missing == 1
        assert await env.outbox.depth() == 0
        assert await env.store.exists(NODE, ["ghost"]) == set()

    async def test_young_entry_is_deferred_until_the_graph_lands(self, make_env):
        """Within the grace window the entry waits for the graph commit."""
        env = await make_env()
        await env.outbox.enqueue([upsert_entry(NODE, EmbedItem("n1", "a"))])
        reconciler = Reconciler(
            env.store, env.outbox, env.graph, ReconcileConfig(grace_s=3600)
        )

        first = await reconciler.drain()
        assert (first.deferred, first.dropped_missing) == (1, 0)
        assert await env.outbox.depth() == 1

        env.graph.add(NODE, EmbedItem("n1", "a"))
        second = await reconciler.drain()

        assert second.upserted == 1
        assert await env.outbox.depth() == 0
        assert await env.store.exists(NODE, ["n1"]) == {"n1"}

    async def test_crash_between_graph_and_vector_is_closed_by_the_drain(
        self, make_env
    ):
        """A committed graph object with no vector is filled in."""
        env = await make_env()
        item = EmbedItem("n1", "a")
        env.graph.add(NODE, item)
        await env.outbox.enqueue([upsert_entry(NODE, item)])
        assert await env.store.exists(NODE, ["n1"]) == set()

        report = await Reconciler(env.store, env.outbox, env.graph).drain()

        assert report.upserted == 1
        assert await env.outbox.depth() == 0
        assert [hit.id for hit in await env.store.search(NODE, A, k=1)] == ["n1"]

    async def test_redelivery_is_harmless(self, make_env):
        """Draining the same entry twice writes the row once."""
        env = await make_env()
        item = EmbedItem("n1", "a")
        env.graph.add(NODE, item)
        reconciler = Reconciler(env.store, env.outbox, env.graph)
        await env.outbox.enqueue([upsert_entry(NODE, item)])
        await reconciler.drain()

        await env.outbox.enqueue([upsert_entry(NODE, item)])
        report = await reconciler.drain()

        assert (report.upserted, report.unchanged) == (0, 1)
        assert await env.outbox.depth() == 0

    async def test_failed_embedding_keeps_the_entry_and_the_gate_closed(self, make_env):
        """A provider failure leaves the episode incomplete until a retry works."""
        env = await make_env()
        env.store.gateway = gateway(OnceFailingEmbedder())
        item = EmbedItem("n1", "a")
        env.graph.add(NODE, item)
        await env.outbox.enqueue([upsert_entry(NODE, item, episode_id="ep1")])
        reconciler = Reconciler(env.store, env.outbox, env.graph)

        first = await reconciler.drain()

        assert first.failed and first.failed[0][0] == "n1"
        assert not await episode_ready(env.outbox, "ep1")
        assert await env.store.exists(NODE, ["n1"]) == set()

        second = await reconciler.drain()

        assert second.upserted == 1
        assert await episode_ready(env.outbox, "ep1")

    async def test_store_error_releases_the_claimed_entries(self, make_env):
        """A backend failure does not burn an attempt."""
        env = await make_env()
        item = EmbedItem("n1", "a")
        env.graph.add(NODE, item)
        await env.outbox.enqueue([upsert_entry(NODE, item)])
        backend = BrokenBackend(
            env.backend.pool,
            env.tenant,
            env.backend.specs,
            PgVectorSettings(),
        )
        broken = VectorStore(backend, gateway(fixed_embedder()), graph=env.graph)
        await broken.ensure_ready()
        reconciler = Reconciler(broken, env.outbox, env.graph)

        with pytest.raises(RuntimeError):
            await reconciler.drain()

        again = await env.outbox.claim(10)
        assert len(again) == 1
        assert again[0].attempts == 1

    async def test_delete_op_removes_the_vector(self, make_env):
        """A delete entry removes the stored vector."""
        env = await make_env()
        env.graph.add(NODE, EmbedItem("n1", "a"))
        await env.store.upsert(NODE, [node_item("n1", "a")])
        await env.outbox.enqueue([delete_entry(NODE, "n1")])

        report = await Reconciler(env.store, env.outbox, env.graph).drain()

        assert report.deleted == 1
        assert await env.store.exists(NODE, ["n1"]) == set()

    async def test_validity_op_updates_the_mirror_without_embedding(self, make_env):
        """A validity entry changes only the mirror; a missing row is missed."""
        env = await make_env()
        env.graph.add(EDGE, edge_item("f1", "a", validity=ACTIVE))
        await env.store.upsert(EDGE, [edge_item("f1", "a", validity=ACTIVE)])
        before = await env.store.fetch_vectors(EDGE, ["f1"])
        shortened = Validity("active", 100, 100, valid_to_us=200)
        await env.outbox.enqueue(
            [validity_entry("f1", shortened), validity_entry("nope", shortened)]
        )

        report = await Reconciler(env.store, env.outbox, env.graph).drain()

        assert (report.validity_applied, report.validity_missed) == (1, 1)
        assert await env.store.fetch_vectors(EDGE, ["f1"]) == before
        current = await env.store.search(EDGE, A, k=5)
        at = await env.store.search(
            EDGE,
            A,
            k=5,
            filters=SearchFilters(temporal=TemporalFilter(valid_at_us=1000)),
        )
        assert [hit.id for hit in current] == ["f1"]
        assert at == []

    async def test_orphan_sweep_deletes_rows_the_graph_no_longer_has(self, make_env):
        """A keyset sweep removes vectors whose graph object is gone."""
        env = await make_env()
        ids = [f"n{i}" for i in range(7)]
        env.graph.add(NODE, *(EmbedItem(object_id, "a") for object_id in ids))
        await env.store.upsert(NODE, [node_item(object_id, "a") for object_id in ids])
        for object_id in ("n1", "n4", "n6"):
            env.graph.remove(NODE, object_id)
        reconciler = Reconciler(env.store, env.outbox, env.graph)

        report = await reconciler.sweep_orphans(NODE, page=2)

        assert (report.scanned, report.changed, report.pages) == (7, 3, 4)
        assert await env.store.exists(NODE, ids) == {"n0", "n2", "n3", "n5"}
        assert (await reconciler.sweep_orphans(NODE, page=2)).changed == 0

    async def test_sweep_page_cap_is_reported(self, make_env):
        """A capped sweep says so instead of looking complete."""
        env = await make_env()
        ids = [f"n{i}" for i in range(6)]
        env.graph.add(NODE, *(EmbedItem(object_id, "a") for object_id in ids))
        await env.store.upsert(NODE, [node_item(object_id, "a") for object_id in ids])

        report = await Reconciler(env.store, env.outbox, env.graph).sweep_orphans(
            NODE, page=2, max_pages=2
        )

        assert report.capped
        assert report.scanned == 4

    async def test_backfill_enqueues_missing_and_stale_items_only(self, make_env):
        """Backfill re-embeds a renamed object and fills a missing vector."""
        env = await make_env()
        env.graph.add(NODE, EmbedItem("n1", "a"), EmbedItem("n2", "a"))
        await env.store.upsert(NODE, [node_item("n1", "a"), node_item("n2", "a")])
        env.graph.add(
            NODE,
            EmbedItem("n1", "a"),
            EmbedItem("n2", "half"),
            EmbedItem("n3", "half"),
        )
        reconciler = Reconciler(env.store, env.outbox, env.graph)

        report = await reconciler.backfill(NODE, page=2)

        assert (report.scanned, report.changed) == (3, 2)
        await drain_all(reconciler)
        scores = await env.store.score_by_ids(NODE, A, ["n1", "n2", "n3"])
        assert scores["n1"] == pytest.approx(1.0)
        assert scores["n2"] == pytest.approx(0.75)
        assert scores["n3"] == pytest.approx(0.75)
        assert (await reconciler.backfill(NODE)).changed == 0

    async def test_mock_embedder_backfill_end_to_end(self, make_env):
        """A page-capped backfill drains to a complete vector table."""
        env = await make_env(MockEmbedder(model_id="mock"))
        items = [EmbedItem(f"n{i}", f"name {i}") for i in range(30)]
        env.graph.add(NODE, *items)
        reconciler = Reconciler(
            env.store, env.outbox, env.graph, ReconcileConfig(batch=7)
        )

        await reconciler.backfill(NODE, page=10)
        await drain_all(reconciler)

        assert await env.store.exists(NODE, [item.id for item in items]) == {
            item.id for item in items
        }
        assert await env.outbox.depth() == 0

    async def test_validity_enqueued_while_the_upsert_is_in_flight_is_not_lost(
        self, make_env
    ):
        """A newer validity token survives the in-flight drain's stale ack."""
        env = await make_env(
            settings=PgVectorSettings(lease_s=0.05),
        )
        env.graph.add(EDGE, edge_item("f1", "a", validity=ACTIVE))
        await env.outbox.enqueue(
            [upsert_entry(EDGE, edge_item("f1", "a", validity=ACTIVE))]
        )
        in_flight = await env.outbox.claim(10)
        shortened = Validity("active", 100, 100, valid_to_us=200)
        await env.outbox.enqueue([validity_entry("f1", shortened)])

        await env.store.upsert(EDGE, [in_flight[0].to_item()])
        await env.outbox.ack(in_flight)
        assert await env.outbox.depth() == 1

        await asyncio.sleep(0.1)
        reconciler = Reconciler(env.store, env.outbox, env.graph)
        await drain_all(reconciler)

        assert await env.outbox.depth() == 0
        at_1000 = SearchFilters(temporal=TemporalFilter(valid_at_us=1000))
        at_150 = SearchFilters(temporal=TemporalFilter(valid_at_us=150))
        assert await env.store.search(EDGE, A, k=5, filters=at_1000) == []
        assert [
            hit.id for hit in await env.store.search(EDGE, A, k=5, filters=at_150)
        ] == ["f1"]


class TestCommitHook:
    """The commit hook turns fact changes into outbox entries."""

    def test_new_fact_becomes_an_upsert_with_text_and_mirror(self) -> None:
        """A new fact carries its sentence, mirror and episode."""
        (entry,) = entries_for_commit([commit(episode_id="ep1")])

        assert entry.op is OutboxOp.UPSERT
        assert entry.text == "a"
        assert entry.episode_id == "ep1"
        assert entry.extras is not None
        assert entry.extras.validity == ACTIVE

    def test_empty_statement_falls_back_to_endpoint_names(self) -> None:
        """Without a sentence the endpoints and relation make the text."""
        (entry,) = entries_for_commit(
            [
                commit(
                    statement="",
                    subject_name="Alice",
                    relation_name="works_at",
                    object_name="Acme",
                )
            ]
        )

        assert entry.text == "Alice works_at Acme"

    def test_existing_fact_becomes_validity_and_deleted_becomes_delete(self) -> None:
        """Existing facts change only the mirror; deleted facts remove the row."""
        ops = [
            entry.op
            for entry in entries_for_commit(
                [commit(is_new=False, validity=CLOSED), commit("f2", deleted=True)]
            )
        ]

        assert ops == [OutboxOp.VALIDITY, OutboxOp.DELETE]

    @pytest.mark.parametrize("status", ["withdrawn", "corrected", "overridden"])
    def test_status_changes_never_delete(self, status: str) -> None:
        """A withdrawn or corrected fact keeps its vector for as-of queries."""
        (entry,) = entries_for_commit(
            [commit(is_new=False, validity=Validity(status, 1, 1))]
        )

        assert entry.op is OutboxOp.VALIDITY

    async def test_enqueue_commit_writes_one_upsert_entry(self, pg_pool):
        """`enqueue_commit` writes one upsert entry for a new fact to the outbox."""
        env = await open_pg_env(pg_pool)
        try:
            written = await enqueue_commit(env.outbox, [commit(episode_id="ep1")])

            assert written == 1
            claimed = await env.outbox.claim(10)
            assert [(entry.object_id, entry.op) for entry in claimed] == [
                ("f1", OutboxOp.UPSERT)
            ]
        finally:
            await drop_tenant(pg_pool, env.tenant)

    async def test_new_fact_is_searchable_after_the_drain(self, pg_pool):
        """A committed fact becomes a vector through the drain."""
        env = await open_pg_env(pg_pool)
        try:
            env.graph.add(EDGE, edge_item("f1", "a", validity=ACTIVE))
            await enqueue_commit(env.outbox, [commit()])

            await drain_all(Reconciler(env.store, env.outbox, env.graph))

            assert [hit.id for hit in await env.store.search(EDGE, A, k=5)] == ["f1"]
        finally:
            await drop_tenant(pg_pool, env.tenant)

    async def test_close_changes_only_the_mirror(self, pg_pool):
        """A closed fact leaves current search but keeps its vector."""
        env = await open_pg_env(pg_pool)
        try:
            env.graph.add(EDGE, edge_item("f1", "a", validity=ACTIVE))
            await enqueue_commit(env.outbox, [commit()])
            reconciler = Reconciler(env.store, env.outbox, env.graph)
            await drain_all(reconciler)
            before = await env.store.fetch_vectors(EDGE, ["f1"])

            shortened = Validity("active", 100, 100, valid_to_us=500)
            await enqueue_commit(env.outbox, [commit(is_new=False, validity=shortened)])
            await drain_all(reconciler)

            at_600 = SearchFilters(temporal=TemporalFilter(valid_at_us=600))
            at_300 = SearchFilters(temporal=TemporalFilter(valid_at_us=300))
            assert await env.store.search(EDGE, A, k=5, filters=at_600) == []
            assert [
                hit.id for hit in await env.store.search(EDGE, A, k=5, filters=at_300)
            ] == ["f1"]
            assert await env.store.fetch_vectors(EDGE, ["f1"]) == before
        finally:
            await drop_tenant(pg_pool, env.tenant)

    async def test_retraction_and_restoration(self, pg_pool):
        """A withdrawn fact keeps its vector and returns when restored."""
        env = await open_pg_env(pg_pool)
        try:
            env.graph.add(EDGE, edge_item("f1", "a", validity=ACTIVE))
            await enqueue_commit(env.outbox, [commit()])
            reconciler = Reconciler(env.store, env.outbox, env.graph)
            await drain_all(reconciler)

            await enqueue_commit(
                env.outbox,
                [commit(is_new=False, validity=Validity("withdrawn", 100, 100))],
            )
            await drain_all(reconciler)
            assert await env.store.search(EDGE, A, k=5) == []
            assert await env.store.exists(EDGE, ["f1"]) == {"f1"}

            await enqueue_commit(env.outbox, [commit(is_new=False, validity=ACTIVE)])
            await drain_all(reconciler)

            assert [hit.id for hit in await env.store.search(EDGE, A, k=5)] == ["f1"]
        finally:
            await drop_tenant(pg_pool, env.tenant)

    async def test_rewritten_fact_gets_a_new_vector_under_a_new_id(self, pg_pool):
        """A rewrite deletes the old id and stores the new one."""
        env = await open_pg_env(pg_pool)
        try:
            env.graph.add(
                EDGE,
                edge_item("f1", "a", validity=ACTIVE),
                edge_item("f2", "half", validity=ACTIVE),
            )
            await enqueue_commit(env.outbox, [commit("f1")])
            reconciler = Reconciler(env.store, env.outbox, env.graph)
            await drain_all(reconciler)

            env.graph.remove(EDGE, "f1")
            await enqueue_commit(
                env.outbox,
                [
                    commit("f1", is_new=False, deleted=True),
                    commit("f2", statement="half", validity=ACTIVE),
                ],
            )
            await drain_all(reconciler)

            assert await env.store.exists(EDGE, ["f1", "f2"]) == {"f2"}
        finally:
            await drop_tenant(pg_pool, env.tenant)
