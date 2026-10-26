"""Dependencies assembled by bootstrap for the service collaborators."""

from dataclasses import dataclass

from ontogr_mem.data_models.clock import Clock
from ontogr_mem.llm import LLMClient
from ontogr_mem.observability import Tracer
from ontogr_mem.relational_store import OperationStore
from ontogr_mem.relational_store.audit import AuditLog
from ontogr_mem.relational_store.jobs.ports import StateStore
from ontogr_mem.service.forget.confirm import ConfirmTokens
from ontogr_mem.service.forget.workflow import EraseWorkflow
from ontogr_mem.service.limits import CallerRateLimiter
from ontogr_mem.service.tenants import TenantRuntimeCache


@dataclass(frozen=True, slots=True)
class ServiceDeps:
    """Process-owned ports and policy shared by the service collaborators."""

    tenants: TenantRuntimeCache
    state: StateStore
    operations: OperationStore
    clock: Clock
    llm: LLMClient | None
    backend: str
    limiter: CallerRateLimiter
    tracer: Tracer
    audit_log: AuditLog
    confirm_tokens: ConfirmTokens
    workflow: EraseWorkflow
    max_attempts: int = 3
