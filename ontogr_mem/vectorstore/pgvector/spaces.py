"""Postgres schema readiness and stored embedding-space metadata.

The backend-neutral SpaceRegistry probes the embedder and creates missing spaces.
This module checks the provisioned database and persists its metadata.
"""

import re
from collections.abc import Mapping, Sequence

from ontogr_mem.vectorstore.errors import IndexDriftError, SpaceMismatchError
from ontogr_mem.vectorstore.models import SpaceSpec, VectorKind
from ontogr_mem.vectorstore.pgvector.errors import TenantNotProvisionedError
from ontogr_mem.vectorstore.pgvector.pool import Conn, Pool
from ontogr_mem.vectorstore.pgvector.provision import check_manifest
from ontogr_mem.vectorstore.pgvector.schema import TABLES, IndexSpec, table_name


_SCHEMA_INDEXES = """
SELECT c.relname AS name
FROM pg_namespace n
LEFT JOIN pg_class c ON c.relnamespace = n.oid AND c.relkind = 'i'
WHERE n.nspname = $1
"""


async def check_schema(conn: Conn, schema: str, tenant: str) -> bool:
    """Check required tables and return whether pgvector supports iterative scans."""
    have = await conn.fetch(
        "SELECT tablename FROM pg_tables WHERE schemaname = $1", schema
    )
    missing = set(TABLES) - {row["tablename"] for row in have}
    if missing:
        msg = f"tenant {tenant!r} lacks tables {sorted(missing)}"
        raise TenantNotProvisionedError(msg)
    version = await conn.fetchval(
        "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
    )
    if version is None:
        msg = "extension vector is not installed"
        raise TenantNotProvisionedError(msg)
    parts = tuple(int(part) for part in re.findall(r"\d+", version)[:3])
    return parts >= (0, 8, 0)


async def missing_indexes(
    pool: Pool, schema: str, manifest: Sequence[IndexSpec]
) -> list[str]:
    """Return the manifest indexes the tenant does not have."""
    async with pool.acquire() as conn:
        present = {
            row["name"]
            for row in await conn.fetch(
                "SELECT c.relname AS name FROM pg_class c "
                "JOIN pg_namespace n ON n.oid = c.relnamespace "
                "WHERE n.nspname = $1 AND c.relkind = 'i'",
                schema,
            )
        }
    return [spec.name for spec in manifest if spec.name not in present]


async def verify_manifest(pool: Pool, schema: str, manifest: list[IndexSpec]) -> None:
    """Compare stored index definitions with the readiness manifest."""
    async with pool.acquire() as conn:
        await check_manifest(conn, schema, manifest)


async def check_indexes(
    conn: Conn, schema: str, tenant: str, manifest: Sequence[IndexSpec] | None
) -> None:
    """Check schema and index presence without rebuilding the readiness manifest."""
    rows = await conn.fetch(_SCHEMA_INDEXES, schema)
    if not rows:
        msg = f"tenant {tenant!r} is not provisioned"
        raise TenantNotProvisionedError(msg)
    if manifest is None:
        return
    present = {row["name"] for row in rows if row["name"] is not None}
    missing = [spec.name for spec in manifest if spec.name not in present]
    if missing:
        msg = f"index {missing[0]} is missing in schema {schema}"
        raise IndexDriftError(msg)


async def check_stored_spaces(
    conn: Conn, schema: str, specs: Mapping[VectorKind, SpaceSpec]
) -> None:
    """Refuse provisioned spaces that disagree with the configured metadata."""
    rows = await conn.fetch(
        "SELECT kind, model, dim, text_template, template_version "
        f"FROM {table_name(schema, 'vec_space')}"
    )
    stored = {row["kind"]: row for row in rows}
    for kind, spec in specs.items():
        row = stored.get(kind.value)
        if row is None:
            continue
        for field, want in (
            ("model", spec.model),
            ("dim", spec.dim),
            ("text_template", spec.text_template),
            ("template_version", spec.template_version),
        ):
            if row[field] != want:
                msg = (
                    f"{kind.value}: stored {field} {row[field]!r}, configured {want!r}"
                )
                raise SpaceMismatchError(msg)


async def read_space(pool: Pool, schema: str, kind: VectorKind) -> SpaceSpec | None:
    """Return the stored space of kind, if any."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT model, dim, text_template, template_version "
            f"FROM {table_name(schema, 'vec_space')} WHERE kind = $1",
            kind.value,
        )
    if not rows:
        return None
    row = rows[0]
    return SpaceSpec(
        kind=kind,
        model=row["model"],
        dim=int(row["dim"]),
        text_template=row["text_template"],
        template_version=int(row["template_version"]),
    )


async def write_space(pool: Pool, schema: str, spec: SpaceSpec) -> None:
    """Insert the space row unless one is already stored."""
    sql = f"""INSERT INTO {table_name(schema, "vec_space")}
            (kind, model, dim, text_template, template_version)
            VALUES ($1, $2, $3, $4, $5) ON CONFLICT (kind) DO NOTHING"""
    async with pool.acquire() as conn:
        await conn.execute(
            sql,
            spec.kind.value,
            spec.model,
            spec.dim,
            spec.text_template,
            spec.template_version,
        )
