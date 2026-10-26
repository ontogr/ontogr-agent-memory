"""Forget confirmations, tenant clearing, audit, and recovery."""

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime

from ontogr_mem.data_models.jobs import JobState
from ontogr_mem.relational_store import (
    OperationKind,
)
from ontogr_mem.relational_store.audit import (
    AuditAction,
    AuditEntry,
    AuditOutcome,
)
from ontogr_mem.service.contracts import (
    ClearTenant,
    ForgetMemory,
    ForgetMode,
    InvalidateMemory,
    WriteReceipt,
)
from ontogr_mem.service.contracts import (
    ForgetResult as ForgetContractResult,
)
from ontogr_mem.service.dependencies import ServiceDeps
from ontogr_mem.service.errors import (
    ConfirmationRequiredError,
    ConflictError,
    InvalidArgumentError,
)
from ontogr_mem.service.forget.confirm import ConfirmTokens, target_digest
from ontogr_mem.service.forget.plan import ForgetPlan, build_plan
from ontogr_mem.service.forget.results import (
    ClearTenantPreview,
    ForgetPreview,
    ForgetResult,
)
from ontogr_mem.service.forget.selector import ForgetSelector
from ontogr_mem.service.jobs import ServiceJobs
from ontogr_mem.service.operation_context import OperationRecorder
from ontogr_mem.service.principal import Grant, Principal
from ontogr_mem.service.runtime import TenantRuntime
from ontogr_mem.temporal.report import Outcome


@dataclass(frozen=True, slots=True)
class RecoveryReport:
    """Summary of temporal episode and erasure recovery at service startup."""

    tenant_ids: tuple[str, ...]
    episodes_recovered: int
    erases: tuple[ForgetResult, ...]


async def _forget_plan(runtime: TenantRuntime, selector: ForgetSelector) -> ForgetPlan:
    """Resolve a selector against one tenant runtime's public store ports."""
    return await build_plan(
        selector,
        episodes=runtime.stores.episodes,
        temporal=runtime.stores.temporal_store,
        resolver=runtime.stores.resolver_store,
    )


async def _clear_plan(runtime: TenantRuntime) -> ForgetPlan:
    """Plan tenant-wide cleanup using all currently stored episodes."""
    episode_ids = await _all_episode_ids(runtime)
    if episode_ids:
        plan = await _forget_plan(runtime, ForgetSelector(episode_ids=episode_ids))
        return ForgetPlan(
            selector_kind="tenant",
            episode_ids=plan.episode_ids,
            observation_ids=plan.observation_ids,
            fact_ids=plan.fact_ids,
            entity_ids=plan.entity_ids,
            collateral_fact_ids=plan.collateral_fact_ids,
            reopened_fact_ids=plan.reopened_fact_ids,
            target=target_digest("tenant", plan.episode_ids),
        )
    return ForgetPlan(
        selector_kind="tenant",
        episode_ids=(),
        observation_ids=(),
        fact_ids=(),
        entity_ids=(),
        collateral_fact_ids=(),
        reopened_fact_ids=(),
        target=target_digest("tenant", ()),
    )


async def _all_episode_ids(runtime: TenantRuntime) -> tuple[str, ...]:
    """Read all tenant episode ids using the store's keyset page contract."""
    found: list[str] = []
    before = None
    while True:
        page = await runtime.stores.episodes.list_episodes(limit=500, before=before)
        if not page:
            break
        found.extend(episode.id for episode in page)
        last = page[-1]
        before = (last.ingested_at, last.id)
        if len(page) < 500:
            break
    return tuple(sorted(set(found)))


def _forget_preview(plan: ForgetPlan, mode: ForgetMode, token: str) -> ForgetPreview:
    """Shape a plan and token as the public forget-preview result."""
    return ForgetPreview(
        mode=mode,
        selector_kind=plan.selector_kind,
        episode_ids=plan.episode_ids,
        fact_ids=plan.fact_ids,
        entity_ids=plan.entity_ids,
        collateral_fact_ids=plan.collateral_fact_ids,
        reopened_fact_ids=plan.reopened_fact_ids,
        confirm_token=token,
        expires_at=_token_expiry(token),
    )


def _token_expiry(token: str) -> datetime:
    """Return a token expiry as an aware UTC datetime."""
    return datetime.fromtimestamp(ConfirmTokens.expiry(token), UTC)


def _forget_detail(
    mode: ForgetMode | str, result: ForgetResult
) -> dict[str, str | int | bool]:
    """Build audit-safe detail from scalar status and count fields."""
    mode_value = mode.value if isinstance(mode, ForgetMode) else mode
    return {"mode": mode_value, "status": result.status, **_counts_detail(result)}


def _counts_detail(result: ForgetResult) -> dict[str, int]:
    """Return each removal count as an audit-safe scalar."""
    return {
        "removed_episodes": result.removed.episodes,
        "removed_facts": result.removed.facts,
        "removed_observations": result.removed.observations,
        "removed_entities": result.removed.entities,
        "removed_vector_documents": result.removed.vector_documents,
        "removed_cache_entries": result.removed.cache_entries,
        "removed_entity_text_updates": result.removed.entity_text_updates,
        "residue_count": len(result.residue),
    }


class ForgetService:
    """Confirm removal requests and audit the injected resumable workflow."""

    def __init__(
        self,
        deps: ServiceDeps,
        recorder: OperationRecorder,
        jobs: ServiceJobs,
        write_lock: asyncio.Lock,
    ) -> None:
        """Bind dependencies, the operation recorder, jobs, and the write lock."""
        self._deps = deps
        self._recorder = recorder
        self._jobs = jobs
        self._write_lock = write_lock

    async def preview_forget(
        self,
        principal: Principal,
        *,
        tenant: str,
        selector: ForgetSelector,
        mode: ForgetMode,
        trace_id: str | None = None,
    ) -> ForgetPreview:
        """Resolve a forget target and return its short-lived confirmation."""
        self._deps.limiter.check(principal.id, "write")
        tenant = await self._deps.tenants.require(
            principal, tenant, Grant.FORGET, "forget_preview", target=selector.kind
        )
        async with self._recorder.track(
            OperationKind.RETRIEVE, principal.id, tenant, trace_id=trace_id
        ):
            runtime = await self._deps.tenants.get(tenant, create=False)
            plan = await _forget_plan(runtime, selector)
            token = self._deps.confirm_tokens.issue(
                principal.id, tenant, plan.target, mode.value
            )
            return _forget_preview(plan, mode, token)

    async def forget(
        self,
        principal: Principal,
        *,
        tenant: str,
        selector: ForgetSelector,
        mode: ForgetMode,
        confirm_token: str | None = None,
        reason: str = "",
        wait_seconds: float = 0.0,
        trace_id: str | None = None,
    ) -> ForgetResult | WriteReceipt:
        """Invalidate a fact, or verify the confirmation token and run a hard forget."""
        self._deps.limiter.check(principal.id, "write")
        tenant = await self._deps.tenants.require(
            principal, tenant, Grant.FORGET, "forget_memory", target=selector.kind
        )
        operation_kind = (
            OperationKind.ERASE
            if mode is ForgetMode.ERASE
            else OperationKind.INVALIDATE
        )
        async with self._recorder.track(
            operation_kind, principal.id, tenant, trace_id=trace_id
        ) as context:
            runtime = await self._deps.tenants.get(tenant, create=False)
            plan = await _forget_plan(runtime, selector)
            if mode is ForgetMode.INVALIDATE:
                if selector.fact_id is None:
                    raise InvalidArgumentError(
                        "invalidate mode requires a fact_id selector"
                    )
                return await self._jobs.submit(
                    "invalidate_memory",
                    principal,
                    InvalidateMemory(
                        tenant=tenant,
                        fact_id=selector.fact_id,
                        reason=reason,
                        wait_seconds=wait_seconds,
                    ),
                    tenant,
                    context,
                )
            if confirm_token is None:
                raise ConfirmationRequiredError(
                    "request a forget preview and pass its confirmation token"
                )
            self._deps.confirm_tokens.verify(
                confirm_token, principal.id, tenant, plan.target, mode.value
            )
            workflow = self._deps.workflow
            try:
                result = await workflow.run(principal.id, tenant, plan, mode)
            except Exception as exc:
                await self._audit_forget(
                    principal.id,
                    tenant,
                    mode,
                    AuditOutcome.FAILED,
                    target=plan.target,
                    error_class=type(exc).__name__,
                    trace_id=context.trace_id,
                )
                raise
            action = (
                AuditAction.ERASE if mode is ForgetMode.ERASE else AuditAction.FORGET
            )
            await self._deps.audit_log.append(
                AuditEntry(
                    action=action,
                    outcome=(
                        AuditOutcome.FAILED
                        if result.status == "failed"
                        else AuditOutcome.OK
                    ),
                    principal_id=principal.id,
                    tenant=tenant,
                    target=plan.target,
                    trace_id=context.trace_id,
                    detail=_forget_detail(mode, result),
                )
            )
            return result

    async def forget_memory(
        self,
        principal: Principal,
        request: ForgetMemory,
        *,
        trace_id: str | None = None,
    ) -> ForgetContractResult:
        """Run a forget request through the selector-based forget operations."""
        tenant = request.tenant or principal.default_tenant
        selector = ForgetSelector(
            fact_id=request.fact_id,
            episode_ids=(request.episode_id,)
            if request.episode_id is not None
            else None,
        )
        if request.dry_run:
            preview = await self.preview_forget(
                principal,
                tenant=tenant,
                selector=selector,
                mode=request.mode,
                trace_id=trace_id,
            )
            return ForgetContractResult(
                mode=preview.mode,
                dry_run=True,
                fact_ids=list(preview.fact_ids),
                episode_ids=list(preview.episode_ids),
                collateral_fact_ids=list(preview.collateral_fact_ids),
                confirm_token=preview.confirm_token,
            )
        result = await self.forget(
            principal,
            tenant=tenant,
            selector=selector,
            mode=request.mode,
            confirm_token=request.confirm_token,
            reason=request.reason,
            wait_seconds=request.wait_seconds,
            trace_id=trace_id,
        )
        if isinstance(result, WriteReceipt):
            return ForgetContractResult(
                mode=request.mode,
                dry_run=False,
                receipt=result,
            )
        return ForgetContractResult(
            mode=request.mode,
            dry_run=False,
            not_cleaned=list(result.residue),
        )

    async def preview_clear_tenant(
        self,
        principal: Principal,
        tenant: str,
        *,
        trace_id: str | None = None,
    ) -> ClearTenantPreview:
        """Preview tenant-wide removal under the admin grant."""
        self._deps.limiter.check(principal.id, "write")
        tenant = await self._deps.tenants.require(
            principal, tenant, Grant.ADMIN, "clear_preview", target="tenant"
        )
        async with self._recorder.track(
            OperationKind.RETRIEVE, principal.id, tenant, trace_id=trace_id
        ):
            runtime = await self._deps.tenants.get(tenant, create=True)
            plan = await _clear_plan(runtime)
            token = self._deps.confirm_tokens.issue(
                principal.id, tenant, plan.target, "clear"
            )
            return ClearTenantPreview(
                mode="clear",
                episode_ids=plan.episode_ids,
                fact_ids=plan.fact_ids,
                entity_ids=plan.entity_ids,
                job_count=await self._deps.state.count_jobs([tenant], list(JobState)),
                confirm_token=token,
                expires_at=_token_expiry(token),
            )

    async def clear_tenant(
        self,
        principal: Principal,
        request: ClearTenant,
        *,
        trace_id: str | None = None,
    ) -> ForgetContractResult:
        """Confirm a tenant-wide clear and remove memory, vectors, and jobs."""
        tenant = request.tenant or principal.default_tenant
        if request.dry_run:
            preview = await self.preview_clear_tenant(
                principal, tenant, trace_id=trace_id
            )
            return ForgetContractResult(
                mode="clear",
                dry_run=True,
                fact_ids=list(preview.fact_ids),
                episode_ids=list(preview.episode_ids),
                confirm_token=preview.confirm_token,
            )
        self._deps.limiter.check(principal.id, "write")
        tenant = await self._deps.tenants.require(
            principal, tenant, Grant.ADMIN, "clear_tenant", target="tenant"
        )
        async with self._write_lock:
            pending = await self._deps.state.count_jobs(
                [tenant], [JobState.QUEUED, JobState.RUNNING]
            )
            if pending:
                raise ConflictError("wait for tenant jobs to finish before clearing")
            async with self._recorder.track(
                OperationKind.ERASE,
                principal.id,
                tenant,
                trace_id=trace_id,
            ) as context:
                runtime = await self._deps.tenants.get(tenant, create=True)
                plan = await _clear_plan(runtime)
                self._deps.confirm_tokens.verify(
                    request.confirm_token or "",
                    principal.id,
                    tenant,
                    plan.target,
                    "clear",
                )
                try:
                    result = await self._deps.workflow.run(
                        principal.id, tenant, plan, "clear"
                    )
                except Exception as exc:
                    await self._deps.audit_log.append(
                        AuditEntry(
                            action=AuditAction.ERASE,
                            outcome=AuditOutcome.FAILED,
                            principal_id=principal.id,
                            tenant=tenant,
                            target="tenant",
                            trace_id=context.trace_id,
                            detail={"mode": "clear", "error_class": type(exc).__name__},
                        )
                    )
                    raise
                finally:
                    self._deps.tenants.discard(tenant)
                await self._deps.audit_log.append(
                    AuditEntry(
                        action=AuditAction.ERASE,
                        outcome=(
                            AuditOutcome.FAILED
                            if result.status == "failed"
                            else AuditOutcome.OK
                        ),
                        principal_id=principal.id,
                        tenant=tenant,
                        target="tenant",
                        trace_id=context.trace_id,
                        detail=_forget_detail("clear", result),
                    )
                )
                return ForgetContractResult(
                    mode="clear",
                    dry_run=False,
                    fact_ids=list(plan.fact_ids),
                    episode_ids=list(plan.episode_ids),
                    collateral_fact_ids=list(plan.collateral_fact_ids),
                    not_cleaned=list(result.residue),
                )

    async def recover_all(self) -> RecoveryReport:
        """Finish pending commits and erasures for every known tenant."""
        tenant_ids = tuple(await self._deps.tenants.tenant_ids())
        episodes_recovered = 0
        erases: list[ForgetResult] = []
        workflow = self._deps.workflow
        for tenant_id in tenant_ids:
            runtime = await self._deps.tenants.get(tenant_id, create=True)
            ontology = await runtime.ontology.catalog()
            report = await runtime.engine(ontology, runtime.resolver.store).recover()
            episodes_recovered += report.counts[Outcome.EPISODE_RECOVERED.value]

            records = await self._deps.state.running(tenant_id)
            results = await workflow.recover(tenant_id)
            for record, result in zip(records, results, strict=True):
                async with self._recorder.track(
                    OperationKind.ERASE, "system", record.tenant
                ):
                    pass
                action = (
                    AuditAction.ERASE
                    if record.mode in {"erase", "clear"}
                    else AuditAction.FORGET
                )
                await self._deps.audit_log.append(
                    AuditEntry(
                        action=action,
                        outcome=(
                            AuditOutcome.FAILED
                            if result.status == "failed"
                            else AuditOutcome.OK
                        ),
                        principal_id="system",
                        tenant=record.tenant,
                        target=record.target,
                        detail={
                            "mode": record.mode,
                            "recovered": True,
                            **_counts_detail(result),
                        },
                    )
                )
                erases.append(result)
        return RecoveryReport(tenant_ids, episodes_recovered, tuple(erases))

    async def _audit_forget(
        self,
        principal_id: str,
        tenant: str,
        mode: ForgetMode,
        outcome: AuditOutcome,
        *,
        target: str,
        error_class: str,
        trace_id: str,
    ) -> None:
        """Record a failed forget attempt without retaining source text."""
        await self._deps.audit_log.append(
            AuditEntry(
                action=(
                    AuditAction.ERASE
                    if mode is ForgetMode.ERASE
                    else AuditAction.FORGET
                ),
                outcome=outcome,
                principal_id=principal_id,
                tenant=tenant,
                target=target,
                trace_id=trace_id,
                detail={
                    "mode": mode.value,
                    "status": "failed",
                    "error_class": error_class,
                },
            )
        )
