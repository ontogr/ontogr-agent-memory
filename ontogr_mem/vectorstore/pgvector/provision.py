"""Tenant provisioning and the index manifest runner."""

import re
from collections.abc import Mapping

from ontogr_mem.vectorstore.errors import IndexDriftError
from ontogr_mem.vectorstore.models import SpaceSpec, VectorKind
from ontogr_mem.vectorstore.pgvector.pool import Conn, Pool
from ontogr_mem.vectorstore.pgvector.schema import (
    HnswParams,
    IndexSpec,
    build_manifest,
    quote_ident,
    schema_name,
    table_ddl,
)


_EXTENSION_LOCK = "pgvector:extension"


def _lock_sql() -> str:
    return "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))"


async def _ensure_extension(conn: Conn) -> None:
    present = await conn.fetchval("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
    if present:
        return
    await conn.execute(_lock_sql(), _EXTENSION_LOCK)
    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")


async def read_hnsw_params(conn: Conn, schema: str) -> HnswParams | None:
    """Return the HNSW build parameters recorded for `schema`, if any."""
    rows = await conn.fetch(
        "SELECT hnsw_m, hnsw_ef_construction "
        f"FROM {quote_ident(schema)}.vec_index_params WHERE id = 1"
    )
    if not rows:
        return None
    row = rows[0]
    return HnswParams(int(row["hnsw_m"]), int(row["hnsw_ef_construction"]))


async def write_hnsw_params(conn: Conn, schema: str, params: HnswParams) -> None:
    """Record the HNSW build parameters of `schema`, keeping the recorded ones."""
    await conn.execute(
        "INSERT INTO "
        f"{quote_ident(schema)}.vec_index_params "
        "(id, hnsw_m, hnsw_ef_construction) VALUES (1, $1, $2) "
        "ON CONFLICT (id) DO NOTHING",
        params.m,
        params.ef_construction,
    )


async def provision(
    pool: Pool,
    tenant: str,
    specs: Mapping[VectorKind, SpaceSpec],
    hnsw: HnswParams | None = None,
) -> str:
    """Create the tenant schema, tables, space rows and indexes.

    The whole step runs in one transaction under an advisory lock keyed on the
    schema name, so concurrent callers converge. It is idempotent. Indexes are
    plain `CREATE INDEX` here because the tables are empty on first provision
    and `CONCURRENTLY` cannot run inside a transaction.

    The HNSW build parameters are part of the space contract. The first call
    records them with the tenant, and every later check reads them back. Indexes
    are always compared against those recorded values.

    Args:
        pool: Connection source.
        tenant: Tenant id.
        specs: Embedding space of every kind.
        hnsw: HNSW build parameters for a tenant that has none yet. `None`
            keeps the recorded parameters, or the defaults for a new tenant.

    Returns:
        The schema name.

    Raises:
        InvalidIdentifierError: If the tenant id is not valid.
        IndexDriftError: If the recorded parameters differ from `hnsw`, or if
            an existing index differs from the manifest.
    """
    schema = schema_name(tenant)
    quoted = quote_ident(schema)
    async with pool.acquire() as conn, conn.transaction():
        await conn.execute(_lock_sql(), f"pgvector:{schema}")
        await _ensure_extension(conn)
        await conn.execute(f"CREATE SCHEMA IF NOT EXISTS {quoted}")
        for ddl in table_ddl(schema, specs):
            await conn.execute(ddl)
        for kind, spec in specs.items():
            await conn.execute(
                f"""INSERT INTO {quoted}.vec_space
                    (kind, model, dim, text_template, template_version)
                    VALUES ($1, $2, $3, $4, $5) ON CONFLICT (kind) DO NOTHING""",
                kind.value,
                spec.model,
                spec.dim,
                spec.text_template,
                spec.template_version,
            )
        recorded = await read_hnsw_params(conn, schema)
        if recorded is None:
            recorded = hnsw or HnswParams()
            await write_hnsw_params(conn, schema, recorded)
        elif hnsw is not None and hnsw != recorded:
            msg = (
                f"tenant {tenant!r} was built with m={recorded.m}, "
                f"ef_construction={recorded.ef_construction}; its indexes "
                f"cannot be re-parameterised to m={hnsw.m}, "
                f"ef_construction={hnsw.ef_construction}"
            )
            raise IndexDriftError(msg)
        await _apply_manifest(
            conn,
            schema,
            build_manifest(schema, specs, recorded),
            concurrently=False,
            allow_rebuild=False,
        )
    return schema


async def drop_tenant(pool: Pool, tenant: str) -> None:
    """Drop one tenant schema and everything in it."""
    schema = schema_name(tenant)
    async with pool.acquire() as conn:
        await conn.execute(f"DROP SCHEMA IF EXISTS {quote_ident(schema)} CASCADE")


_INDEX_STATE = """
SELECT c.relname AS name, i.indisvalid AS valid,
  obj_description(c.oid, 'pg_class') AS cmt
FROM pg_class c
JOIN pg_index i ON i.indexrelid = c.oid
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = $1
"""


async def index_state(conn: Conn, schema: str) -> dict[str, tuple[bool, str | None]]:
    """Return each index of `schema` as `name -> (is_valid, comment)`."""
    rows = await conn.fetch(_INDEX_STATE, schema)
    return {row["name"]: (row["valid"], row["cmt"]) for row in rows}


def _problem(spec: IndexSpec, state: tuple[bool, str | None]) -> str | None:
    valid, comment = state
    if not valid:
        return f"index {spec.name} is invalid"
    if comment != spec.digest:
        return f"index {spec.name} differs from the manifest (comment {comment!r})"
    return None


async def check_manifest(conn: Conn, schema: str, manifest: list[IndexSpec]) -> None:
    """Raise `IndexDriftError` when an index is missing, invalid or differs."""
    have = await index_state(conn, schema)
    for spec in manifest:
        if spec.name not in have:
            msg = f"index {spec.name} is missing in schema {schema}"
            raise IndexDriftError(msg)
        problem = _problem(spec, have[spec.name])
        if problem:
            raise IndexDriftError(problem)


async def _apply_manifest(
    conn: Conn,
    schema: str,
    manifest: list[IndexSpec],
    *,
    concurrently: bool,
    allow_rebuild: bool,
) -> list[str]:
    have = await index_state(conn, schema)
    quoted = quote_ident(schema)
    done: list[str] = []
    for spec in manifest:
        if spec.name in have:
            problem = _problem(spec, have[spec.name])
            if problem is None:
                continue
            if not allow_rebuild:
                raise IndexDriftError(problem)
            await conn.execute(f"DROP INDEX {quoted}.{spec.name}")
        sql = spec.sql
        if concurrently:
            sql = sql.replace("CREATE INDEX", "CREATE INDEX CONCURRENTLY", 1)
        await conn.execute(sql)
        await conn.execute(f"COMMENT ON INDEX {quoted}.{spec.name} IS '{spec.digest}'")
        done.append(spec.name)
    return done


async def ensure_indexes(
    pool: Pool,
    tenant: str,
    specs: Mapping[VectorKind, SpaceSpec],
    *,
    concurrently: bool = True,
    allow_rebuild: bool = False,
) -> list[str]:
    """Create the manifest indexes that are missing.

    The indexes are built with the parameters recorded for the tenant, so a
    repaired index matches the ones that are already there. Uses
    `CREATE INDEX CONCURRENTLY` outside a transaction under a session-level
    advisory lock. Returns the names created. A mismatch raises
    `IndexDriftError` unless `allow_rebuild`.
    """
    schema = schema_name(tenant)
    async with pool.acquire() as conn:
        recorded = await read_hnsw_params(conn, schema)
        manifest = build_manifest(schema, specs, recorded or HnswParams())
        key = f"pgvector:indexes:{schema}"
        await conn.execute("SELECT pg_advisory_lock(hashtextextended($1, 0))", key)
        try:
            return await _apply_manifest(
                conn,
                schema,
                manifest,
                concurrently=concurrently,
                allow_rebuild=allow_rebuild,
            )
        finally:
            await conn.execute(
                "SELECT pg_advisory_unlock(hashtextextended($1, 0))", key
            )


def parse_count(status: str) -> int:
    """Return the row count from a status string such as `INSERT 0 3`."""
    match = re.search(r"(\d+)\s*$", status)
    return int(match.group(1)) if match else 0
