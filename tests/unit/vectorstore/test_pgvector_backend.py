"""Tests for the pgvector backend: the index manifest and the write outcome.

The database is replaced by a double that models one tenant schema: its
tables, its spaces, the HNSW parameters it was built with, and the indexes
it has. The backend is exercised through its public methods.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, cast

import pytest

from ontogr_mem.vectorstore.errors import IndexDriftError
from ontogr_mem.vectorstore.models import (
    FailReason,
    Row,
    SearchFilters,
    SpaceSpec,
    VectorKind,
)
from ontogr_mem.vectorstore.pgvector.pool import Conn, Pool
from ontogr_mem.vectorstore.pgvector.provision import provision
from ontogr_mem.vectorstore.pgvector.schema import (
    TABLES,
    HnswParams,
    build_manifest,
)
from ontogr_mem.vectorstore.pgvector.store import PgBackend


NODE = VectorKind.NODE
A = (1.0, 0.0, 0.0)


class DataError(Exception):
    """The error a PostgreSQL data exception raises."""

    sqlstate = "22023"


class FakeTenant:
    """The state of one tenant schema in the fake database."""

    def __init__(
        self,
        schema: str,
        specs: dict[VectorKind, SpaceSpec],
        params: HnswParams | None,
    ) -> None:
        """Record the schema, its spaces and its build parameters."""
        self.schema = schema
        self.specs = specs
        self.params = params
        self.tables = set(TABLES)
        self.provisioned = True
        self.rejected: set[str] = set()
        self.indexes: dict[str, str] = {}
        if params is not None:
            self.indexes = {
                spec.name: spec.digest for spec in build_manifest(schema, specs, params)
            }

    def index_rows(self) -> list[dict[str, Any]]:
        """Return the names of the indexes the tenant has."""
        return [{"name": name} for name in sorted(self.indexes)]


class FakeTransaction:
    """A transaction that changes nothing."""

    async def __aenter__(self) -> "FakeTransaction":
        """Enter the transaction."""
        return self

    async def __aexit__(self, *args: object) -> bool:
        """Leave the transaction."""
        return False


class FakeConn:
    """A connection that answers the statements of one tenant."""

    def __init__(self, tenant: FakeTenant) -> None:
        """Bind the connection to the tenant."""
        self.tenant = tenant

    async def execute(
        self, query: str, *args: Any, timeout: float | None = None
    ) -> str:
        """Answer a statement."""
        if "INSERT INTO" in query and "vec_index_params" in query:
            self.tenant.params = HnswParams(int(args[0]), int(args[1]))
            return "INSERT 0 1"
        if "INSERT INTO" in query:
            ids = [str(value) for value in args[0]]
            refused = [value for value in ids if value in self.tenant.rejected]
            if refused:
                raise DataError(f"row {refused[0]} is malformed")
            return f"INSERT 0 {len(ids)}"
        return "OK"

    async def fetchval(
        self, query: str, *args: Any, timeout: float | None = None
    ) -> Any:
        """Answer a query that returns one value."""
        if "pg_extension" in query:
            return "0.8.0"
        return None

    async def fetch(
        self, query: str, *args: Any, timeout: float | None = None
    ) -> list[Any]:
        """Answer a query with rows."""
        if "pg_tables" in query:
            return [{"tablename": name} for name in sorted(self.tenant.tables)]
        if "vec_index_params" in query:
            return [
                {
                    "hnsw_m": params.m,
                    "hnsw_ef_construction": params.ef_construction,
                }
                for params in [self.tenant.params]
                if params is not None
            ]
        if "vec_space" in query:
            return [
                {
                    "kind": kind.value,
                    "model": spec.model,
                    "dim": spec.dim,
                    "text_template": spec.text_template,
                    "template_version": spec.template_version,
                }
                for kind, spec in self.tenant.specs.items()
            ]
        if "obj_description" in query:
            return [
                {"name": name, "valid": True, "cmt": comment}
                for name, comment in sorted(self.tenant.indexes.items())
            ]
        if "pg_class" in query or "pg_namespace" in query:
            return self.tenant.index_rows() if self.tenant.provisioned else []
        return []

    def transaction(self, *, readonly: bool = False) -> FakeTransaction:
        """Open a transaction."""
        return FakeTransaction()


class FakePool:
    """A pool that lends connections to one tenant."""

    def __init__(self, tenant: FakeTenant) -> None:
        """Bind the pool to the tenant."""
        self.tenant = tenant

    @asynccontextmanager
    async def acquire(self) -> AsyncGenerator[Conn]:
        """Lend a connection to the tenant."""
        yield cast(Conn, FakeConn(self.tenant))

    async def close(self) -> None:
        """Close the pool."""


def _specs() -> dict[VectorKind, SpaceSpec]:
    """Return a space of every kind."""
    return {
        kind: SpaceSpec(kind, "mock", 3, f"{kind.value}.text", 1) for kind in VectorKind
    }


def _backend(tenant: FakeTenant) -> PgBackend:
    """Return a backend bound to the tenant."""
    return PgBackend(cast(Pool, FakePool(tenant)), "t1", tenant.specs)


class TestPgBackendManifest:
    """How the backend checks the indexes of its tenant."""

    async def test_a_tenant_built_with_other_parameters_is_usable(self) -> None:
        """The manifest is built from the parameters the tenant was built with."""
        specs = _specs()
        tenant = FakeTenant("m_t1", specs, HnswParams(m=32, ef_construction=128))
        backend = _backend(tenant)

        await backend.ensure_ready(list(specs.values()))

    async def test_provisioning_records_the_build_parameters(self) -> None:
        """The parameters a tenant is provisioned with are kept."""
        specs = _specs()
        tenant = FakeTenant("m_t1", specs, None)
        pool = FakePool(tenant)

        await provision(
            cast(Pool, pool),
            "t1",
            specs,
            HnswParams(m=32, ef_construction=128),
        )

        assert tenant.params == HnswParams(m=32, ef_construction=128)

    async def test_a_drifted_index_is_reported_when_the_store_is_made_ready(
        self,
    ) -> None:
        """An index built with another definition is not silently accepted."""
        specs = _specs()
        tenant = FakeTenant("m_t1", specs, HnswParams())
        backend = _backend(tenant)
        tenant.indexes["vec_node_hnsw"] = "vs:another-definition"

        with pytest.raises(IndexDriftError):
            await backend.ensure_ready(list(specs.values()))

    async def test_a_search_does_not_compare_the_manifest(self) -> None:
        """The manifest is compared once, when the store is made ready."""
        specs = _specs()
        tenant = FakeTenant("m_t1", specs, HnswParams())
        backend = _backend(tenant)
        await backend.ensure_ready(list(specs.values()))
        tenant.indexes["vec_node_hnsw"] = "vs:another-definition"

        assert await backend.search(NODE, A, 5, SearchFilters(), None) == []

    async def test_a_search_reports_a_missing_index(self) -> None:
        """A dropped index is an error, never an empty list."""
        specs = _specs()
        tenant = FakeTenant("m_t1", specs, HnswParams())
        backend = _backend(tenant)
        await backend.ensure_ready(list(specs.values()))
        del tenant.indexes["vec_node_hnsw"]

        with pytest.raises(IndexDriftError):
            await backend.search(NODE, A, 5, SearchFilters(), None)


class TestPgBackendWrite:
    """How the backend reports the rows it could not write."""

    async def test_a_rejected_row_is_not_reported_as_a_missing_graph_object(
        self,
    ) -> None:
        """A row the database refuses is reported on its own."""
        specs = _specs()
        tenant = FakeTenant("m_t1", specs, HnswParams())
        tenant.rejected.add("bad")
        backend = _backend(tenant)
        rows = [
            Row("ok1", "h1", A),
            Row("bad", "h2", A),
            Row("ok2", "h3", A),
        ]

        result = await backend.write(NODE, rows)

        assert result.written == 2
        assert result.missing == ()
        assert [(item.id, item.reason) for item in result.rejected] == [
            ("bad", FailReason.REJECTED)
        ]
