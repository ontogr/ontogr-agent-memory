"""Commands for inspecting and clearing tenant data."""

import sys

from ontogr_mem.data_models.tenant import validate_tenant_id
from ontogr_mem.service.bootstrap import clear_session, open_graph_store
from ontogr_mem.service.contracts import ClearTenant
from ontogr_mem.service.errors import ConfirmationRequiredError, ConflictError
from ontogr_mem.service.forget.results import ClearTenantPreview
from ontogr_mem.service.principal import Grant, Principal


async def list_tenants_text() -> str:
    """Return tenants reported by the configured graph backend."""
    async with open_graph_store() as store:
        tenants = sorted(await store.list_tenants())
    return "\n".join(tenants) if tenants else "No tenants found."


def _print_preview(tenant: str, preview: ClearTenantPreview) -> None:
    """Print the data counts and retained records before confirmation."""
    print(f"Tenant: {tenant}")
    print(f"Graph: m_{tenant}")
    print(f"Episodes: {len(preview.episode_ids)}")
    print(f"Facts: {len(preview.fact_ids)}")
    print(f"Entities: {len(preview.entity_ids)}")
    print(f"Jobs: {preview.job_count}")
    print(
        "Will remove: the graph (its nodes, edges and ontology versions), "
        f"its vectors and {preview.job_count} jobs."
    )
    print(
        "Will keep: audit rows, operation rows and erasure rows for this tenant. "
        "They record this clear."
    )


async def clear_tenant_command(tenant: str) -> int:  # noqa: PLR0911
    """Interactively remove one tenant's graph, vectors, and jobs."""
    tenant = validate_tenant_id(tenant)
    async with open_graph_store() as store:
        tenants = await store.list_tenants()
    if tenant not in tenants:
        print(f"No data for tenant {tenant}; nothing to clear.")
        return 0

    principal = Principal(
        id="local-cli",
        default_tenant=tenant,
        grants={tenant: frozenset({Grant.ADMIN})},
    )
    async with clear_session(tenant) as service:
        preview = await service.preview_clear_tenant(principal, tenant)
        _print_preview(tenant, preview)
        if not sys.stdin.isatty() or not sys.stdout.isatty():
            print("Clear needs an interactive terminal. Nothing was removed.")
            return 1
        try:
            answer = input(f"Type {tenant} to remove it: ")
        except (EOFError, KeyboardInterrupt):
            print("Nothing was removed.")
            return 1
        if answer != tenant:
            print("The name did not match. Nothing was removed.")
            return 1
        try:
            await service.clear_tenant(
                principal,
                ClearTenant(tenant=tenant, confirm_token=preview.confirm_token),
            )
        except ConfirmationRequiredError:
            print("The confirmation expired; run clear again.")
            return 1
        except ConflictError as error:
            print(str(error))
            return 1
    print(
        f"Removed graph and vectors and {preview.job_count} jobs for tenant {tenant}."
    )
    print(f"Kept: audit rows, operation rows and erasure rows for tenant {tenant}.")
    return 0
