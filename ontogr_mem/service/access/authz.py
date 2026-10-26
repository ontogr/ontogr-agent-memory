"""Tenant grant checks and audit records for caller-supplied principals."""

from ontogr_mem.data_models.clock import Clock
from ontogr_mem.data_models.tenant import validate_tenant_id
from ontogr_mem.observability import NoOpTracer, SpanName, Tracer, open_span
from ontogr_mem.service.access.audit_sink import AuditEvent, AuditSink
from ontogr_mem.service.errors import ObjectNotFoundError
from ontogr_mem.service.principal import Grant, Principal


_DENIED_MESSAGE = "tenant or object not found"
_DENIED_HINT = (
    "Check the tenant id and whether this principal has the required grant "
    "(read, write, forget, share, export, or admin)."
)


class Authorizer:
    """Check principals' tenant grants and audit each decision."""

    def __init__(
        self,
        audit: AuditSink,
        clock: Clock,
        tracer: Tracer | None = None,
    ) -> None:
        """Use the supplied audit sink, clock, and optional tracer."""
        self._audit_sink = audit
        self._clock = clock
        self._tracer = tracer if tracer is not None else NoOpTracer()

    async def require(
        self,
        principal: Principal,
        tenant: str,
        grant: Grant,
        action: str,
        *,
        target: str = "",
    ) -> str:
        """Check and audit a tenant grant, returning the validated tenant.

        Raises:
            ObjectNotFoundError: If the principal does not hold the grant.
            InvalidTenantError: If the tenant id is malformed.
        """
        validate_tenant_id(tenant)
        with open_span(
            self._tracer,
            SpanName.AUTH_CHECK,
            principal=principal.id,
            tenant=tenant,
            grant=grant.value,
        ) as span:
            allowed = principal.holds(tenant, grant)
            outcome = "allowed" if allowed else "denied"
            span.set("outcome", outcome)
            await self._audit_sink.append(
                AuditEvent(
                    actor_id=principal.id,
                    action=action,
                    outcome=outcome,
                    tenant=tenant,
                    target=target,
                    detail={},
                    at=self._clock.now(),
                )
            )
            if not allowed:
                raise ObjectNotFoundError(_DENIED_MESSAGE, hint=_DENIED_HINT)
        return tenant
