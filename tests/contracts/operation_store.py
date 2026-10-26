"""Shared operation lifecycle and usage behavior for operation stores."""

import pytest

from ontogr_mem.data_models.clock import Clock
from ontogr_mem.data_models.tenant import InvalidTenantError
from ontogr_mem.data_models.usage import OperationUsage
from ontogr_mem.relational_store import (
    OperationKind,
    OperationOutcome,
    OperationStateError,
    OperationStore,
    TenantUsage,
)


class OperationStoreContract:
    """Operation lifecycle, filtering and usage rules shared by every store."""

    @pytest.fixture
    def clock(self) -> Clock:
        """Return a fixed clock for exact timestamp assertions."""
        raise NotImplementedError

    @pytest.fixture
    def operation_store(self, clock: Clock) -> OperationStore:
        """Return an isolated operation store."""
        raise NotImplementedError

    async def test_start_returns_a_running_record_with_request_fields(
        self, operation_store: OperationStore, clock: Clock
    ) -> None:
        """A new row keeps its caller, Tenant, trace and start time."""
        operation_id = await operation_store.start(
            kind=OperationKind.WRITE,
            principal_id="owner",
            tenant="acme",
            session_id="session-1",
            trace_id="trace-1",
        )

        record = await operation_store.get(operation_id)

        assert operation_id
        assert record is not None
        assert record.id == operation_id
        assert record.kind is OperationKind.WRITE
        assert record.tenant == "acme"
        assert record.principal_id == "owner"
        assert record.session_id == "session-1"
        assert record.trace_id == "trace-1"
        assert record.started_at == clock.now()
        assert record.ended_at is None
        assert record.outcome is OperationOutcome.RUNNING

    async def test_start_keeps_parent_operation_id(
        self, operation_store: OperationStore
    ) -> None:
        """A child row points to the parent operation."""
        parent_id = await operation_store.start(
            kind=OperationKind.WRITE,
            principal_id="owner",
            tenant="acme",
        )

        child_id = await operation_store.start(
            kind=OperationKind.RETRIEVE,
            principal_id="owner",
            tenant="acme",
            parent_id=parent_id,
        )

        child = await operation_store.get(child_id)
        assert child is not None
        assert child.parent_id == parent_id

    async def test_finish_keeps_end_time_and_usage(
        self, operation_store: OperationStore, clock: Clock
    ) -> None:
        """Finishing an operation stores its outcome, time and usage."""
        operation_id = await operation_store.start(
            kind=OperationKind.RETRIEVE,
            principal_id="owner",
            tenant="acme",
        )
        usage = OperationUsage(120, 30, 0.25)

        await operation_store.finish(
            operation_id,
            outcome=OperationOutcome.OK,
            usage=usage,
        )

        record = await operation_store.get(operation_id)
        assert record is not None
        assert record.outcome is OperationOutcome.OK
        assert record.ended_at == clock.now()
        assert record.usage == usage
        assert not record.cost_unknown

    async def test_finish_summarises_error_to_its_first_line(
        self, operation_store: OperationStore
    ) -> None:
        """A failed operation keeps only the exception class and first line."""
        operation_id = await operation_store.start(
            kind=OperationKind.WRITE,
            principal_id="owner",
            tenant="acme",
        )

        await operation_store.finish(
            operation_id,
            outcome=OperationOutcome.FAILED,
            error=ValueError("first line\nsecond line"),
        )

        record = await operation_store.get(operation_id)
        assert record is not None
        assert (record.error_class, record.error) == ("ValueError", "first line")

    async def test_finish_cuts_error_summary_at_200_characters(
        self, operation_store: OperationStore
    ) -> None:
        """Long error summaries are limited to 200 characters."""
        operation_id = await operation_store.start(
            kind=OperationKind.WRITE,
            principal_id="owner",
            tenant="acme",
        )

        await operation_store.finish(
            operation_id,
            outcome=OperationOutcome.FAILED,
            error=ValueError("x" * 300),
        )

        record = await operation_store.get(operation_id)
        assert record is not None
        assert record.error == "x" * 200

    async def test_finish_redacts_minted_key_from_error_summary(
        self, operation_store: OperationStore
    ) -> None:
        """Error summaries mask minted keys before storing the row."""
        operation_id = await operation_store.start(
            kind=OperationKind.WRITE,
            principal_id="owner",
            tenant="acme",
        )

        await operation_store.finish(
            operation_id,
            outcome=OperationOutcome.FAILED,
            error=ValueError("request used mk_abc123def"),
        )

        record = await operation_store.get(operation_id)
        assert record is not None
        assert record.error == "request used mk_***"
        assert "mk_abc123def" not in repr(record)
        assert record.cost_unknown is False

    async def test_finish_rejects_unknown_and_already_finished_ids(
        self, operation_store: OperationStore
    ) -> None:
        """Only a running operation can be finished."""
        with pytest.raises(OperationStateError):
            await operation_store.finish("missing", outcome=OperationOutcome.OK)

        operation_id = await operation_store.start(
            kind=OperationKind.WRITE,
            principal_id="owner",
            tenant="acme",
        )
        await operation_store.finish(operation_id, outcome=OperationOutcome.OK)

        with pytest.raises(OperationStateError):
            await operation_store.finish(operation_id, outcome=OperationOutcome.OK)

    async def test_finish_rejects_running_outcome_and_error_on_success(
        self, operation_store: OperationStore
    ) -> None:
        """A finish must be terminal and successful rows cannot carry errors."""
        operation_id = await operation_store.start(
            kind=OperationKind.WRITE,
            principal_id="owner",
            tenant="acme",
        )

        with pytest.raises(ValueError):
            await operation_store.finish(operation_id, outcome=OperationOutcome.RUNNING)
        with pytest.raises(ValueError):
            await operation_store.finish(
                operation_id,
                outcome=OperationOutcome.OK,
                error=ValueError("failure"),
            )

        record = await operation_store.get(operation_id)
        assert record is not None
        assert record.outcome is OperationOutcome.RUNNING

    async def test_usage_for_tenant_without_operations_is_zero(
        self, operation_store: OperationStore
    ) -> None:
        """An empty Tenant total has zero tokens, cost and unpriced rows."""
        assert await operation_store.usage("acme") == TenantUsage(0, 0, 0, 0.0, 0)

    @pytest.mark.parametrize("tokens_in,tokens_out", [(-1, 0), (0, -1)])
    def test_usage_rejects_negative_token_counts(self, tokens_in, tokens_out) -> None:
        """Token counts cannot be negative."""
        with pytest.raises(ValueError, match="token counts must not be negative"):
            OperationUsage(tokens_in, tokens_out)

    async def test_usage_counts_operations_tokens_priced_cost_and_unknown_cost(
        self, operation_store: OperationStore
    ) -> None:
        """Usage includes running rows and excludes unknown costs from the sum."""
        await operation_store.start(
            kind=OperationKind.WRITE,
            principal_id="owner",
            tenant="acme",
        )
        priced_id = await operation_store.start(
            kind=OperationKind.RETRIEVE,
            principal_id="owner",
            tenant="acme",
        )
        unpriced_id = await operation_store.start(
            kind=OperationKind.HISTORY,
            principal_id="owner",
            tenant="acme",
        )
        await operation_store.finish(
            priced_id,
            outcome=OperationOutcome.OK,
            usage=OperationUsage(120, 30, 0.25),
        )
        await operation_store.finish(
            unpriced_id,
            outcome=OperationOutcome.FAILED,
            usage=OperationUsage(5, 3, None),
        )

        unpriced = await operation_store.get(unpriced_id)
        assert unpriced is not None and unpriced.cost_unknown

        assert await operation_store.usage("acme") == TenantUsage(
            operations=3,
            tokens_in=125,
            tokens_out=33,
            cost_usd=0.25,
            unpriced_operations=1,
        )

    async def test_usage_excludes_operations_from_other_tenants(
        self, operation_store: OperationStore
    ) -> None:
        """Tenant totals do not include another Tenant's operation rows."""
        for tenant, tokens in [("acme", 4), ("beta", 90)]:
            operation_id = await operation_store.start(
                kind=OperationKind.WRITE,
                principal_id="owner",
                tenant=tenant,
            )
            await operation_store.finish(
                operation_id,
                outcome=OperationOutcome.OK,
                usage=OperationUsage(tokens, 1, 0.1),
            )

        assert await operation_store.usage("acme") == TenantUsage(1, 4, 1, 0.1, 0)

    async def test_list_operations_sorts_filters_and_limits(
        self, operation_store: OperationStore
    ) -> None:
        """Listing breaks equal-time ties by insertion and honors filters."""
        ids = []
        for tenant in ("acme", "beta", "acme"):
            ids.append(
                await operation_store.start(
                    kind=OperationKind.WRITE,
                    principal_id="owner",
                    tenant=tenant,
                )
            )

        records = await operation_store.list_operations()
        assert [record.id for record in records] == list(reversed(ids))
        assert [
            record.id for record in await operation_store.list_operations(limit=2)
        ] == [ids[2], ids[1]]
        assert [
            record.tenant
            for record in await operation_store.list_operations(tenant="acme")
        ] == ["acme", "acme"]
        with pytest.raises(ValueError):
            await operation_store.list_operations(limit=0)

    async def test_start_and_usage_reject_invalid_tenant_ids(
        self, operation_store: OperationStore
    ) -> None:
        """Invalid Tenant ids fail before storage or aggregation."""
        with pytest.raises(InvalidTenantError):
            await operation_store.start(
                kind=OperationKind.WRITE,
                principal_id="owner",
                tenant="bad id",
            )
        with pytest.raises(InvalidTenantError):
            await operation_store.usage("bad id")
        assert await operation_store.list_operations() == []

    async def test_start_rejects_an_empty_principal_id(
        self, operation_store: OperationStore
    ) -> None:
        """A new operation requires an identifiable principal."""
        with pytest.raises(ValueError, match="principal id must not be empty"):
            await operation_store.start(
                kind=OperationKind.WRITE,
                principal_id="",
                tenant="acme",
            )
