"""The pgvector backend and its Postgres-specific operations."""

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import Any, TypeVar

from ontogr_mem.vectorstore.errors import (
    QueryTimeoutError,
    SpaceMismatchError,
)
from ontogr_mem.vectorstore.models import (
    BackendWrite,
    Capabilities,
    Row,
    SearchFilters,
    SpaceSpec,
    ValidityUpdate,
    VectorKind,
)
from ontogr_mem.vectorstore.pgvector import spaces, writes
from ontogr_mem.vectorstore.pgvector.errors import TenantNotProvisionedError
from ontogr_mem.vectorstore.pgvector.pool import Pool
from ontogr_mem.vectorstore.pgvector.provision import (
    ensure_indexes,
    read_hnsw_params,
)
from ontogr_mem.vectorstore.pgvector.queries import (
    build_search,
    parse_vector,
    search_rows,
)
from ontogr_mem.vectorstore.pgvector.schema import (
    ID_COLUMN,
    HnswParams,
    IndexSpec,
    build_manifest,
    cast_type,
    format_vector,
    schema_name,
    table_name,
)
from ontogr_mem.vectorstore.pgvector.settings import PgVectorSettings
from ontogr_mem.vectorstore.protocol import RawHit


T = TypeVar("T")


def _sqlstate(error: BaseException) -> str:
    value = getattr(error, "sqlstate", None)
    return value if isinstance(value, str) else ""


def _is_missing_relation(error: BaseException) -> bool:
    name = type(error).__name__
    return name in {"UndefinedTableError", "InvalidSchemaNameError"} or _sqlstate(
        error
    ) in {"42P01", "3F000"}


def _is_timeout(error: BaseException) -> bool:
    return type(error).__name__ == "QueryCanceledError" or _sqlstate(error) == "57014"


def _is_retryable(error: BaseException) -> bool:
    name = type(error).__name__
    return _sqlstate(error) in {"40P01", "40001"} or name in {
        "DeadlockDetectedError",
        "SerializationError",
        "ConnectionDoesNotExistError",
        "ConnectionFailureError",
    }


async def _retry(
    run: Callable[[], Awaitable[T]],
    *,
    attempts: int,
    backoff_s: float,
    sleep: Callable[[float], Awaitable[None]],
) -> T:
    """Run an idempotent statement, retrying deadlocks and connection resets."""
    for attempt in range(attempts):
        try:
            return await run()
        except Exception as error:
            if not _is_retryable(error) or attempt + 1 >= attempts:
                raise
            await sleep(backoff_s * 2**attempt)
    raise RuntimeError("retry loop ended without a result")


class PgBackend:
    """Rows of one tenant schema. Every statement names its tables.

    The backend keeps no session state.

    Args:
        pool: Connection source.
        tenant: Tenant id.
        specs: Embedding space of every kind.
        settings: Timeouts and search limits.
        sleep: Retry delay, injectable in tests.
    """

    def __init__(
        self,
        pool: Pool,
        tenant: str,
        specs: dict[VectorKind, SpaceSpec],
        settings: PgVectorSettings | None = None,
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        """Bind the backend to one tenant schema."""
        self.pool = pool
        self.tenant = tenant
        self.schema = schema_name(tenant)
        self.specs = specs
        self.settings = settings or PgVectorSettings()
        self.capabilities = Capabilities(
            validity_mirror=True,
            exact_scan=True,
            iterative_scan=False,
            ann_index=True,
        )
        self.force_exact: set[VectorKind] = set()
        self._sleep = sleep
        self._manifest: list[IndexSpec] | None = None

    def _t(self, kind: VectorKind | str) -> str:
        return table_name(self.schema, kind)

    @asynccontextmanager
    async def _translate(self):
        try:
            yield
        except Exception as error:
            if _is_missing_relation(error):
                msg = f"tenant {self.tenant!r} is not provisioned: {error}"
                raise TenantNotProvisionedError(msg) from error
            if _is_timeout(error):
                raise QueryTimeoutError(str(error)) from error
            raise

    async def _run(self, run: Callable[[], Awaitable[T]]) -> T:
        async def translated() -> T:
            async with self._translate():
                return await run()

        return await _retry(translated, attempts=3, backoff_s=0.2, sleep=self._sleep)

    async def ensure_ready(self, specs: Sequence[SpaceSpec]) -> None:
        """Check the schema, tables, pgvector version and index manifest.

        The manifest is built once, from the HNSW parameters recorded for the
        tenant, and kept for the lifetime of the backend. A later search
        checks that the indexes are still there, not that they still match
        the manifest.
        """
        self.specs = {spec.kind: spec for spec in specs}
        manifest = await self._run(self._prepare)
        self._manifest = manifest
        if await self._run(
            lambda: spaces.missing_indexes(self.pool, self.schema, manifest)
        ):
            await ensure_indexes(self.pool, self.tenant, self.specs)
        await self._run(
            lambda: spaces.verify_manifest(self.pool, self.schema, manifest)
        )

    async def _prepare(self) -> list[IndexSpec]:
        """Check the tenant and build the manifest of its indexes."""
        async with self.pool.acquire() as conn:
            iterative_scan = await spaces.check_schema(conn, self.schema, self.tenant)
            self.capabilities = replace(
                self.capabilities,
                iterative_scan=iterative_scan,
            )
            await spaces.check_stored_spaces(conn, self.schema, self.specs)
            recorded = await read_hnsw_params(conn, self.schema)
            return build_manifest(self.schema, self.specs, recorded or HnswParams())

    async def read_space(self, kind: VectorKind) -> SpaceSpec | None:
        """Return the stored space of `kind`, if any."""
        return await self._run(lambda: spaces.read_space(self.pool, self.schema, kind))

    async def write_space(self, spec: SpaceSpec) -> None:
        """Insert the space row unless one is already stored."""
        await self._run(lambda: spaces.write_space(self.pool, self.schema, spec))

    async def existing_hashes(
        self, kind: VectorKind, ids: Sequence[str]
    ) -> dict[str, str]:
        """Return the stored text hash of each id that has a vector."""
        id_column = ID_COLUMN[kind]

        async def run() -> list[Any]:
            async with self._translate(), self.pool.acquire() as conn:
                return await conn.fetch(
                    f"SELECT {id_column} AS id, text_hash FROM {self._t(kind)} "
                    f"WHERE {id_column} = ANY($1::text[])",
                    list(ids),
                )

        return {row["id"]: str(row["text_hash"]) for row in await self._run(run)}

    async def write(self, kind: VectorKind, rows: Sequence[Row]) -> BackendWrite:
        """Upsert rows; a row the database rejects is reported on its own.

        A row the database refuses is malformed, not missing a graph object.
        It is reported as rejected. The other rows are still written.
        """
        if not rows:
            return BackendWrite(0)
        try:
            written = await self._run(
                lambda: writes.write_batch(
                    self.pool,
                    self.schema,
                    self.specs[kind],
                    rows,
                    self.settings.bulk_timeout_s,
                )
            )
            return BackendWrite(written)
        except Exception as error:
            if not writes.is_bad_row(error):
                raise
        written, rejected = await self._run(
            lambda: writes.write_one_by_one(
                self.pool,
                self.schema,
                self.specs[kind],
                rows,
                self.settings.bulk_timeout_s,
            )
        )
        return BackendWrite(written, rejected=rejected)

    async def update_validity(self, updates: Sequence[ValidityUpdate]) -> int:
        """Set the mirror on existing edge rows, in id order to avoid deadlocks."""
        if not updates:
            return 0
        return await self._run(
            lambda: writes.update_validity(
                self.pool, self.schema, updates, self.settings.bulk_timeout_s
            )
        )

    async def delete(self, kind: VectorKind, ids: Sequence[str]) -> int:
        """Delete by id; missing ids are ignored."""
        return await self._run(lambda: writes.delete(self.pool, self.schema, kind, ids))

    async def search(
        self,
        kind: VectorKind,
        vector: Sequence[float],
        k: int,
        filters: SearchFilters,
        min_score: float | None,
        *,
        exact: bool = False,
    ) -> list[RawHit]:
        """Search the index, or scan every row when asked or when it is small."""
        dim = self.specs[kind].dim
        sql, params = build_search(
            self.schema,
            kind,
            dim,
            vector=vector,
            k=k,
            filters=filters,
            min_score=min_score,
        )

        async def run() -> list[Any]:
            async with (
                self._translate(),
                self.pool.acquire() as conn,
                conn.transaction(readonly=True),
            ):
                await spaces.check_indexes(
                    conn, self.schema, self.tenant, self._manifest
                )
                return await search_rows(
                    conn,
                    self.schema,
                    kind,
                    sql,
                    params,
                    k=k,
                    filters=filters,
                    exact=exact or kind in self.force_exact,
                    iterative_scan=self.capabilities.iterative_scan,
                    settings=self.settings,
                )

        try:
            rows = await self._run(run)
        except Exception as error:
            if _sqlstate(error).startswith("22"):
                msg = f"{kind.value}: server rejected the query vector: {error}"
                raise SpaceMismatchError(msg) from error
            raise
        return [(row["id"], float(row["score"])) for row in rows]

    async def score_by_ids(
        self, kind: VectorKind, vector: Sequence[float], ids: Sequence[str]
    ) -> dict[str, float]:
        """Return the normalised score of each id that has a vector."""
        cast = cast_type(self.specs[kind].dim)
        id_column = ID_COLUMN[kind]

        async def run() -> list[Any]:
            async with self._translate(), self.pool.acquire() as conn:
                return await conn.fetch(
                    f"SELECT {id_column} AS id, "
                    f"(2 - (embedding <=> $1::text::{cast})) / 2 AS score "
                    f"FROM {self._t(kind)} WHERE {id_column} = ANY($2::text[])",
                    format_vector(vector),
                    list(ids),
                )

        return {row["id"]: float(row["score"]) for row in await self._run(run)}

    async def exists(self, kind: VectorKind, ids: Sequence[str]) -> set[str]:
        """Return the ids that have a row."""
        id_column = ID_COLUMN[kind]

        async def run() -> list[Any]:
            async with self._translate(), self.pool.acquire() as conn:
                return await conn.fetch(
                    f"SELECT {id_column} AS id FROM {self._t(kind)} "
                    f"WHERE {id_column} = ANY($1::text[])",
                    list(ids),
                )

        return {row["id"] for row in await self._run(run)}

    async def iter_ids(
        self, kind: VectorKind, after: str | None, limit: int
    ) -> list[str]:
        """Return a keyset page of ids strictly greater than `after`."""
        id_column = ID_COLUMN[kind]

        async def run() -> list[Any]:
            async with self._translate(), self.pool.acquire() as conn:
                return await conn.fetch(
                    f"SELECT {id_column} AS id FROM {self._t(kind)} "
                    f"WHERE ($1::text IS NULL OR {id_column} > $1::text) "
                    f"ORDER BY {id_column} LIMIT $2",
                    after,
                    limit,
                )

        return [row["id"] for row in await self._run(run)]

    async def fetch_vectors(
        self, kind: VectorKind, ids: Sequence[str]
    ) -> dict[str, tuple[float, ...]]:
        """Return the stored vectors, parsed from their text form."""
        id_column = ID_COLUMN[kind]

        async def run() -> list[Any]:
            async with self._translate(), self.pool.acquire() as conn:
                return await conn.fetch(
                    f"SELECT {id_column} AS id, embedding::text AS vector "
                    f"FROM {self._t(kind)} WHERE {id_column} = ANY($1::text[])",
                    list(ids),
                )

        return {row["id"]: parse_vector(row["vector"]) for row in await self._run(run)}
