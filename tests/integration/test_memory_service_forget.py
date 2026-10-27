"""Public MemoryService behavior for forget and tenant clearing."""

import pytest

from ontogr_mem.relational_store.audit import AuditAction
from ontogr_mem.service.contracts import (
    AddMemory,
    ClearTenant,
    ForgetMode,
    ListEpisodes,
    MemoryKind,
)
from ontogr_mem.service.errors import ConfirmationRequiredError, ObjectNotFoundError
from ontogr_mem.service.forget.selector import ForgetSelector
from ontogr_mem.service.principal import Grant
from tests.support.graph_backend import GraphBackend
from tests.support.memory_service import memory_service_harness


class TestMemoryServiceForget:
    """Exercise confirmation, cleanup, and audit through the service API."""

    async def test_forget_requires_a_matching_preview_and_removes_episode(
        self, graph_backend: GraphBackend
    ):
        """A fact selector can be previewed and then deleted with its token."""
        harness = await memory_service_harness(
            graph_backend,
        )
        receipt = await harness.service.add_memory(
            harness.principal,
            AddMemory(
                tenant="t1",
                kind=MemoryKind.FACT,
                subject="Ana",
                relation="works_at",
                object="Acme",
                wait_seconds=5,
            ),
        )
        job = await harness.service.get_job(harness.principal, receipt.job_id)
        assert job.result is not None
        fact_id = job.result["fact_ids"][0]
        selector = ForgetSelector(fact_id=fact_id)
        forget_principal = harness.principal.model_copy(
            update={"grants": {"t1": harness.principal.grants["t1"] | {Grant.FORGET}}}
        )

        preview = await harness.service.preview_forget(
            forget_principal,
            tenant="t1",
            selector=selector,
            mode=ForgetMode.DELETE,
        )
        with pytest.raises(ConfirmationRequiredError):
            await harness.service.forget(
                forget_principal,
                tenant="t1",
                selector=selector,
                mode=ForgetMode.DELETE,
                confirm_token="invalid",
            )

        result = await harness.service.forget(
            forget_principal,
            tenant="t1",
            selector=selector,
            mode=ForgetMode.DELETE,
            confirm_token=preview.confirm_token,
        )

        assert result.status == "completed"
        assert result.removed.episodes == 1
        page = await harness.service.list_episodes(
            harness.principal, ListEpisodes(tenant="t1")
        )
        assert page.episodes == []

    async def test_clear_removes_memory_jobs_but_keeps_grants_and_audits(
        self, graph_backend: GraphBackend
    ):
        """Admin clear removes tenant data and job rows, not access grants."""
        harness = await memory_service_harness(
            graph_backend,
        )
        admin = harness.principal.model_copy(
            update={"grants": {"t1": frozenset({Grant.ADMIN})}}
        )
        write_principal = harness.principal.model_copy(
            update={"grants": {"t1": frozenset({Grant.ADMIN})}}
        )
        receipt = await harness.service.add_memory(
            write_principal,
            AddMemory(
                tenant="t1",
                kind=MemoryKind.FACT,
                subject="Ana",
                relation="works_at",
                object="Acme",
                wait_seconds=5,
            ),
        )
        preview = await harness.service.preview_clear_tenant(admin, "t1")

        result = await harness.service.clear_tenant(
            admin,
            ClearTenant(tenant="t1", confirm_token=preview.confirm_token),
        )

        assert result.mode == "clear"
        assert not result.dry_run
        assert result.not_cleaned == []
        with pytest.raises(ObjectNotFoundError):
            await harness.service.get_job(admin, receipt.job_id)
        assert await harness.state.tenant_ids() == []
        audit_rows = await harness.service.audit_log(admin, "t1", limit=20)
        clear_rows = [
            row
            for row in audit_rows
            if row.entry.action is AuditAction.ERASE
            and row.entry.detail.get("mode") == "clear"
        ]
        assert len(clear_rows) == 1
        assert clear_rows[0].entry.detail["status"] == "completed"
        assert admin.holds("t1", Grant.ADMIN)

    async def test_clear_rejects_a_token_after_new_episode_is_added(
        self, graph_backend: GraphBackend
    ):
        """The clear token stops applying when tenant contents change."""
        harness = await memory_service_harness(
            graph_backend,
        )
        admin = harness.principal.model_copy(
            update={"grants": {"t1": frozenset({Grant.ADMIN})}}
        )
        await harness.service.add_memory(
            admin,
            AddMemory(
                tenant="t1",
                kind=MemoryKind.FACT,
                subject="Bob",
                relation="lives_in",
                object="Paris",
                wait_seconds=5,
            ),
        )
        preview = await harness.service.preview_clear_tenant(admin, "t1")
        await harness.service.add_memory(
            admin,
            AddMemory(
                tenant="t1",
                kind=MemoryKind.FACT,
                subject="Ana",
                relation="works_at",
                object="Acme",
                wait_seconds=5,
            ),
        )

        with pytest.raises(ConfirmationRequiredError):
            await harness.service.clear_tenant(
                admin,
                ClearTenant(tenant="t1", confirm_token=preview.confirm_token),
            )
