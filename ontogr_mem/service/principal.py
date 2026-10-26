"""Who is calling and what they may do: grants per tenant on a principal."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Final

from pydantic import AfterValidator, ConfigDict, Field

from ontogr_mem.data_models.tenant import validate_tenant_id
from ontogr_mem.service.contracts import ContractModel, TenantId
from ontogr_mem.service.errors import ObjectNotFoundError


_NOT_VISIBLE_MESSAGE = "tenant or object not found"
_NOT_VISIBLE_HINT = "Check the tenant id and the grants on this principal."


def _check_grant_key(value: str) -> str:
    """Accept the wildcard key or a valid tenant id.

    Args:
        value: The grant key to check.

    Returns:
        The same key.

    Raises:
        InvalidTenantError: If the key is neither the wildcard nor a valid
            tenant id.
    """
    if value == ALL_TENANTS:
        return value
    return validate_tenant_id(value)


class Grant(StrEnum):
    """One permission on one tenant; ``admin`` implies the other five."""

    READ = "read"
    WRITE = "write"
    FORGET = "forget"
    SHARE = "share"
    EXPORT = "export"
    ADMIN = "admin"


ALL_TENANTS: Final = "*"
"""Grant key that covers every tenant."""

GrantKey = Annotated[str, AfterValidator(_check_grant_key)]


class Principal(ContractModel):
    """A caller: an id, grants per tenant and the tenant used by default.

    A principal with a ``parent_id`` is an agent created by that principal,
    and its grants must stay within the parent's (see ``is_within``).

    Attributes:
        id: Stable id of the caller.
        grants: Grants held on each tenant id, or on ``*`` for every tenant.
        default_tenant: Tenant a request lands on when it names none.
        parent_id: Principal that created this agent, or None for a person
            or service.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1)
    grants: dict[GrantKey, frozenset[Grant]] = Field(default_factory=dict)
    default_tenant: TenantId = "default"
    parent_id: str | None = None

    def holds(self, tenant: str, grant: Grant) -> bool:
        """Return whether the grant, or admin, is held on the tenant.

        Args:
            tenant: Tenant id to check, or ``*`` for the wildcard key.
            grant: The grant to look for.

        Returns:
            True when the named tenant key or the wildcard key carries the
            grant or ``admin``.
        """
        held = self.grants.get(tenant, frozenset()) | self.grants.get(
            ALL_TENANTS, frozenset()
        )
        return grant in held or Grant.ADMIN in held

    def require(self, tenant: str, grant: Grant) -> None:
        """Check the grant, or fail with the one not-found error.

        A tenant the caller holds no grant on, a tenant with a different
        grant, and a tenant that does not exist all give the same error, so
        the refusal cannot be used to probe tenant ids.

        Args:
            tenant: Tenant id the verb runs in.
            grant: The grant the verb needs.

        Raises:
            ObjectNotFoundError: If the grant does not hold.
        """
        if not self.holds(tenant, grant):
            raise ObjectNotFoundError(_NOT_VISIBLE_MESSAGE, hint=_NOT_VISIBLE_HINT)

    def is_within(self, parent: Principal) -> bool:
        """Return whether every grant of this principal holds for the parent.

        An agent's grants must satisfy the subset rule when the agent is
        created. The wildcard key is checked against the parent's
        wildcard key.

        Args:
            parent: The principal that created this one.

        Returns:
            True when the parent holds every (key, grant) pair of this
            principal.
        """
        return all(
            parent.holds(key, grant)
            for key, grants in self.grants.items()
            for grant in grants
        )
