"""Tenant runtime caching, discovery, and grant checks."""

import asyncio
from collections.abc import Awaitable, Callable

from ontogr_mem.relational_store.jobs.ports import StateStore
from ontogr_mem.service.access.authz import Authorizer
from ontogr_mem.service.errors import (
    ObjectNotFoundError,
)
from ontogr_mem.service.principal import ALL_TENANTS, Grant, Principal
from ontogr_mem.service.relations import load_defaults
from ontogr_mem.service.runtime import TenantRuntime
from ontogr_mem.tenant_stores import TenantStores


StoresFactory = Callable[..., Awaitable[TenantStores]]


class TenantRuntimeCache:
    """Cache initialized tenant runtimes and remember authorized tenant ids."""

    def __init__(
        self,
        factory: Callable[[str], Awaitable[TenantRuntime]],
        state: StateStore,
        authorizer: Authorizer,
    ) -> None:
        """Bind runtime construction, durable tenant discovery, and grant checks."""
        self._factory = factory
        self._state = state
        self._authorizer = authorizer
        self._runtimes: dict[str, TenantRuntime] = {}
        self._known_tenants: set[str] = set()
        self._tenant_locks: dict[str, asyncio.Lock] = {}

    async def get_or_create(self, tenant_id: str) -> TenantRuntime:
        """Initialize a tenant runtime when it is not already cached."""
        return await self.get(tenant_id, create=True)

    async def require_existing(self, tenant_id: str) -> TenantRuntime:
        """Return a runtime only for an authorized or durable tenant."""
        return await self.get(tenant_id, create=False)

    def discard(self, tenant_id: str) -> None:
        """Forget cached state after clearing a tenant."""
        self._runtimes.pop(tenant_id, None)
        self._known_tenants.discard(tenant_id)

    async def tenant_ids(self) -> list[str]:
        """Return known and durable tenant ids for startup recovery."""
        return sorted(self._known_tenants | set(await self._state.tenant_ids()))

    async def get(self, tenant_id: str, *, create: bool) -> TenantRuntime:
        """Return a cached runtime or assemble it from the tenant factory."""
        lock = self._tenant_locks.setdefault(tenant_id, asyncio.Lock())
        async with lock:
            runtime = self._runtimes.get(tenant_id)
            if runtime is not None:
                return runtime
            if not create and tenant_id not in self._known_tenants:
                known = await self._state.tenant_ids()
                if tenant_id not in known:
                    raise ObjectNotFoundError("tenant or object not found")
            runtime = await self._factory(tenant_id)
            ontology = await load_defaults(runtime.ontology)
            runtime.refresh_relation_labels(ontology)
            self._runtimes[tenant_id] = runtime
            self._known_tenants.add(tenant_id)
            return runtime

    async def require(
        self,
        principal: Principal,
        tenant_id: str,
        grant: Grant,
        action: str,
        *,
        target: str = "",
    ) -> str:
        """Check a grant and remember the authorized tenant."""
        selected = await self._authorizer.require(
            principal, tenant_id, grant, action, target=target
        )
        self._known_tenants.add(selected)
        return selected

    async def visible(self, principal: Principal) -> list[str]:
        """Find the caller's readable tenants from grants and job records."""
        candidates = set(await self._state.tenant_ids())
        candidates.add(principal.default_tenant)
        candidates.update(
            tenant for tenant in principal.grants if tenant != ALL_TENANTS
        )
        return sorted(
            tenant for tenant in candidates if principal.holds(tenant, Grant.READ)
        )
