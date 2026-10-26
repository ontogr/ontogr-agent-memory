"""Exercise audit, operation and request-context state in one SQLite file."""

import asyncio
import sqlite3

from ontogr_mem.data_models.clock import Clock
from ontogr_mem.data_models.usage import OperationUsage
from ontogr_mem.relational_store import (
    OperationKind,
    OperationOutcome,
    TenantUsage,
)
from ontogr_mem.relational_store.audit import AuditAction, AuditOutcome
from ontogr_mem.relational_store.sqlite.audit import SqliteAuditLog
from ontogr_mem.relational_store.sqlite.operations import SqliteOperationStore
from ontogr_mem.request_context import RequestContext, bind_context, record_usage
from ontogr_mem.service.access.audit_sink import AuditEvent, ChainedAuditSink
from tests.support.clock import FixedClock


class TestAuditOperationsEndToEnd:
    """The state file preserves usage, audit history and restart outcomes."""

    async def test_concurrent_operations_survive_audit_edit_and_restart(self, tmp_path):
        """Independent contexts and SQLite rows keep their expected totals."""
        path = str(tmp_path / "state.sqlite3")
        clock: Clock = FixedClock()
        connection = sqlite3.connect(path, isolation_level=None)
        operations = SqliteOperationStore(connection, clock)
        audit_log = SqliteAuditLog(connection, clock)
        audit_sink = ChainedAuditSink(audit_log)

        parent_id = await operations.start(
            kind=OperationKind.WRITE,
            principal_id="owner",
            tenant="acme",
            trace_id="trace-1",
        )

        async def run_child(
            kind: OperationKind,
            principal_id: str,
            tenant: str,
            calls: list[OperationUsage] | None = None,
            error: BaseException | None = None,
        ) -> str:
            operation_id = await operations.start(
                kind=kind,
                principal_id=principal_id,
                tenant=tenant,
                parent_id=parent_id,
                trace_id="trace-1",
            )
            context = RequestContext(
                principal_id=principal_id,
                tenant=tenant,
                trace_id="trace-1",
                operation_id=operation_id,
            )
            with bind_context(context):
                for call in calls or []:
                    record_usage(call.tokens_in, call.tokens_out, call.cost_usd)
                await asyncio.sleep(0)
                await operations.finish(
                    operation_id,
                    outcome=(
                        OperationOutcome.FAILED
                        if error is not None
                        else OperationOutcome.OK
                    ),
                    usage=(
                        context.usage.snapshot()
                        if calls is not None
                        else OperationUsage()
                    ),
                    error=error,
                )
            await audit_sink.append(
                AuditEvent(
                    actor_id=principal_id,
                    action="search",
                    outcome="allowed",
                    tenant=tenant,
                    target=operation_id,
                    detail={},
                    at=clock.now(),
                )
            )
            return operation_id

        retrieve_id, write_id, export_id = await asyncio.gather(
            run_child(
                OperationKind.RETRIEVE,
                "owner",
                "acme",
                [OperationUsage(100, 20, 0.002), OperationUsage(5, 0, None)],
            ),
            run_child(
                OperationKind.WRITE,
                "owner",
                "acme",
                [OperationUsage(40, 10, 0.5)],
            ),
            run_child(
                OperationKind.EXPORT,
                "owner",
                "beta",
                error=ValueError("first line\nsecond line"),
            ),
        )
        await operations.finish(parent_id, outcome=OperationOutcome.OK)
        await audit_sink.append(
            AuditEvent(
                actor_id="owner",
                action="export",
                outcome="ok",
                tenant="beta",
                target=export_id,
                detail={},
                at=clock.now(),
            )
        )

        assert await operations.usage("acme") == TenantUsage(
            operations=3,
            tokens_in=145,
            tokens_out=30,
            cost_usd=0.5,
            unpriced_operations=1,
        )
        assert await operations.usage("beta") == TenantUsage(1, 0, 0, 0.0, 0)
        retrieve = await operations.get(retrieve_id)
        assert retrieve is not None and retrieve.parent_id == parent_id
        beta_operation = await operations.get(export_id)
        assert beta_operation is not None
        assert (beta_operation.error_class, beta_operation.error) == (
            "ValueError",
            "first line",
        )
        rows = await audit_log.list_rows()
        assert len(rows) == 4
        assert sum(row.entry.action is AuditAction.AUTHORIZE for row in rows) == 3
        assert (
            sum(
                row.entry.action is AuditAction.EXPORT
                and row.entry.outcome is AuditOutcome.OK
                and row.entry.tenant == "beta"
                for row in rows
            )
            == 1
        )
        assert write_id != retrieve_id and await audit_log.verify() is None

        owner = sqlite3.connect(path, isolation_level=None)
        try:
            owner.execute("UPDATE audit_rows SET outcome = 'denied' WHERE seq = 2")
        finally:
            owner.close()
        assert await audit_log.verify() == 2

        interrupted_id = await operations.start(
            kind=OperationKind.WRITE,
            principal_id="owner",
            tenant="acme",
            parent_id=parent_id,
        )
        connection.close()

        reopened_connection = sqlite3.connect(path, isolation_level=None)
        try:
            reopened = SqliteOperationStore(reopened_connection, clock)
            interrupted = await reopened.get(interrupted_id)
            assert interrupted is not None
            assert (interrupted.outcome, interrupted.error_class) == (
                OperationOutcome.FAILED,
                "Interrupted",
            )
            assert (await reopened.usage("acme")).operations == 4
        finally:
            reopened_connection.close()
