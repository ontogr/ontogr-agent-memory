"""Tenant naming, DDL builders and the index manifest. No I/O."""

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from ontogr_mem.data_models import (
    OPEN_MICROS,
    FactStatus,
    InvalidTenantError,
    validate_tenant_id,
)
from ontogr_mem.vectorstore.models import (
    EdgeExtras,
    NodeExtras,
    Row,
    SpaceSpec,
    VectorKind,
)
from ontogr_mem.vectorstore.pgvector.errors import InvalidIdentifierError


VECTOR_INDEX_LIMIT = 2000
"""Largest dimension an HNSW `vector` index accepts; `halfvec` goes to 4000."""

TABLES = (
    "vec_space",
    "vec_node",
    "vec_edge",
    "vec_episode",
    "vec_outbox",
    "vec_index_params",
)
"""The tables a provisioned tenant has."""


def schema_name(tenant: str) -> str:
    """Return `m_{tenant}` for a valid tenant id.

    Raises:
        InvalidIdentifierError: If `tenant` does not match the shared tenant rule.
    """
    try:
        validate_tenant_id(tenant)
    except InvalidTenantError as error:
        raise InvalidIdentifierError(str(error)) from error
    return f"m_{tenant}"


def quote_ident(name: str) -> str:
    """Double-quote a validated identifier.

    Raises:
        InvalidIdentifierError: If the name cannot be quoted safely.
    """
    if '"' in name or "\x00" in name or not name:
        msg = f"cannot quote identifier {name!r}"
        raise InvalidIdentifierError(msg)
    return f'"{name}"'


def table_name(schema: str, kind: VectorKind | str) -> str:
    """Return the schema-qualified table name of `kind`."""
    base = f"vec_{kind.value}" if isinstance(kind, VectorKind) else kind
    return f"{quote_ident(schema)}.{base}"


def vector_type(dim: int) -> str:
    """Return the embedding column type for `dim`."""
    if dim <= VECTOR_INDEX_LIMIT:
        return f"vector({dim})"
    return f"halfvec({dim})"


def opclass(dim: int) -> str:
    """Return the cosine operator class for the column type of `dim`."""
    return "vector_cosine_ops" if dim <= VECTOR_INDEX_LIMIT else "halfvec_cosine_ops"


def cast_type(dim: int) -> str:
    """Return the type used to cast a text parameter to a vector."""
    return "vector" if dim <= VECTOR_INDEX_LIMIT else "halfvec"


def format_vector(vector: Sequence[float]) -> str:
    """Return the pgvector text literal of `vector`."""
    return "[" + ",".join(repr(float(value)) for value in vector) + "]"


@dataclass(frozen=True, slots=True)
class ColumnLayout:
    """One stored column and its batch-write representation."""

    name: str
    sql_type: str
    array_type: str
    value: Callable[[Row], object]
    ddl_suffix: str = ""
    select_type: str | None = None


@dataclass(frozen=True, slots=True)
class TableLayout:
    """The shared DDL and batch-write contract for one vector table."""

    kind: VectorKind
    id_column: str
    columns: tuple[ColumnLayout, ...]

    @property
    def table(self) -> str:
        """Return the table name."""
        return f"vec_{self.kind.value}"

    def arrays(self, rows: Sequence[Row]) -> list[list[object]]:
        """Build one pgvector parameter array per stored column."""
        return [[column.value(row) for row in rows] for column in self.columns]


def _node_extras(row: Row) -> NodeExtras:
    extras = row.extras
    return extras if isinstance(extras, NodeExtras) else NodeExtras()


def _edge_extras(row: Row) -> EdgeExtras:
    if isinstance(row.extras, EdgeExtras):
        return row.extras
    msg = f"edge row {row.id!r} lacks EdgeExtras"
    raise TypeError(msg)


def table_layout(kind: VectorKind, dim: int) -> TableLayout:
    """Return the columns and row-value extraction for one vector kind."""
    vector = ColumnLayout(
        "embedding",
        vector_type(dim),
        "text",
        lambda row: format_vector(row.vector),
        ddl_suffix=" NOT NULL",
        select_type=cast_type(dim),
    )
    text_hash = ColumnLayout(
        "text_hash", "text", "text", lambda row: row.text_hash, " NOT NULL"
    )
    if kind is VectorKind.NODE:
        columns = (
            ColumnLayout(
                "node_id",
                "text",
                "text",
                lambda row: row.id,
                ' COLLATE "C" PRIMARY KEY',
            ),
            ColumnLayout(
                "entity_type", "int", "int", lambda row: _node_extras(row).entity_type
            ),
            ColumnLayout(
                "status", "text", "text", lambda row: _node_extras(row).status
            ),
            text_hash,
            vector,
        )
        id_column = "node_id"
    elif kind is VectorKind.EDGE:
        columns = (
            ColumnLayout(
                "fact_id",
                "text",
                "text",
                lambda row: row.id,
                ' COLLATE "C" PRIMARY KEY',
            ),
            ColumnLayout(
                "relation_type_id",
                "int",
                "int",
                lambda row: _edge_extras(row).relation_type_id,
            ),
            ColumnLayout(
                "subject_id",
                "text",
                "text",
                lambda row: _edge_extras(row).subject_id,
                " NOT NULL",
            ),
            ColumnLayout(
                "object_id",
                "text",
                "text",
                lambda row: _edge_extras(row).object_id,
                " NOT NULL",
            ),
            ColumnLayout(
                "status",
                "text",
                "text",
                lambda row: _edge_extras(row).validity.status,
                " NOT NULL",
            ),
            ColumnLayout(
                "valid_from",
                "bigint",
                "bigint",
                lambda row: _edge_extras(row).validity.valid_from_us,
                " NOT NULL",
            ),
            ColumnLayout(
                "valid_to",
                "bigint",
                "bigint",
                lambda row: _edge_extras(row).validity.valid_to_us,
                " NOT NULL",
            ),
            ColumnLayout(
                "recorded_at",
                "bigint",
                "bigint",
                lambda row: _edge_extras(row).validity.recorded_at_us,
                " NOT NULL",
            ),
            ColumnLayout(
                "expired_at",
                "bigint",
                "bigint",
                lambda row: _edge_extras(row).validity.expired_at_us,
                " NOT NULL",
            ),
            text_hash,
            vector,
        )
        id_column = "fact_id"
    else:
        columns = (
            ColumnLayout(
                "episode_id",
                "text",
                "text",
                lambda row: row.id,
                ' COLLATE "C" PRIMARY KEY',
            ),
            text_hash,
            vector,
        )
        id_column = "episode_id"
    return TableLayout(kind, id_column, columns)


ID_COLUMN = {kind: table_layout(kind, 1).id_column for kind in VectorKind}


def table_ddl(schema: str, specs: Mapping[VectorKind, SpaceSpec]) -> list[str]:
    """Return the CREATE TABLE statements of one tenant schema."""
    s = quote_ident(schema)
    vector_tables = []
    for kind in VectorKind:
        layout = table_layout(kind, specs[kind].dim)
        columns = [
            f"  {column.name} {column.sql_type}{column.ddl_suffix}"
            for column in layout.columns
        ]
        columns.append("  embedded_at timestamptz NOT NULL DEFAULT now()")
        vector_tables.append(
            f"CREATE TABLE IF NOT EXISTS {s}.{layout.table} (\n"
            + ",\n".join(columns)
            + ")"
        )
    return [
        f"""CREATE TABLE IF NOT EXISTS {s}.vec_space (
  kind text PRIMARY KEY CHECK (kind IN ('node','edge','episode')),
  model text NOT NULL,
  dim int NOT NULL,
  text_template text NOT NULL,
  template_version int NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now())""",
        f"""CREATE TABLE IF NOT EXISTS {s}.vec_index_params (
  id int PRIMARY KEY CHECK (id = 1),
  hnsw_m int NOT NULL,
  hnsw_ef_construction int NOT NULL)""",
        *vector_tables,
        f"""CREATE TABLE IF NOT EXISTS {s}.vec_outbox (
  kind text NOT NULL CHECK (kind IN ('node','edge','episode')),
  object_id text NOT NULL,
  op text NOT NULL CHECK (op IN ('upsert','delete','validity')),
  text text,
  episode_id text,
  extras jsonb,
  attempts int NOT NULL DEFAULT 0,
  last_error text,
  claimed_at timestamptz,
  token uuid NOT NULL DEFAULT gen_random_uuid(),
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (kind, object_id, op))""",
    ]


@dataclass(frozen=True, slots=True)
class HnswParams:
    """Explicit HNSW build parameters, not server defaults.

    The parameters belong to the tenant that owns the indexes. They are
    recorded at provisioning. Every later check reads them back, so an index
    set is never compared with one built using other parameters.

    Attributes:
        m: Number of neighbours per node.
        ef_construction: Size of the candidate list while building.
    """

    m: int = 16
    ef_construction: int = 64


@dataclass(frozen=True, slots=True)
class IndexSpec:
    """One manifest entry; `sql` is the CREATE INDEX statement."""

    table: str
    name: str
    sql: str

    @property
    def digest(self) -> str:
        """Return the stable hash of the definition, stored as its comment."""
        return "vs:" + hashlib.sha256(self.sql.encode("utf-8")).hexdigest()


def build_manifest(
    schema: str, specs: Mapping[VectorKind, SpaceSpec], hnsw: HnswParams | None = None
) -> list[IndexSpec]:
    """Return the indexes of one tenant schema.

    Every table that holds a vector gets an HNSW index on its embedding
    column, built with `hnsw`. The current-facts predicate uses literal
    constants so a generic prepared plan can still prove that the partial
    index applies.
    """
    params = hnsw or HnswParams()
    s = quote_ident(schema)
    with_ = f"WITH (m = {params.m}, ef_construction = {params.ef_construction})"
    node_dim = specs[VectorKind.NODE].dim
    edge_dim = specs[VectorKind.EDGE].dim
    episode_dim = specs[VectorKind.EPISODE].dim

    def index(table: str, name: str, body: str) -> IndexSpec:
        return IndexSpec(table, name, f"CREATE INDEX {name} ON {s}.{table} {body}")

    return [
        index(
            "vec_node",
            "vec_node_hnsw",
            f"USING hnsw (embedding {opclass(node_dim)}) {with_}",
        ),
        index(
            "vec_edge",
            "vec_edge_hnsw_cur",
            f"USING hnsw (embedding {opclass(edge_dim)}) {with_} "
            f"WHERE status = '{FactStatus.ACTIVE.value}' "
            f"AND expired_at = {OPEN_MICROS}",
        ),
        index(
            "vec_edge",
            "vec_edge_hnsw_all",
            f"USING hnsw (embedding {opclass(edge_dim)}) {with_}",
        ),
        index(
            "vec_episode",
            "vec_episode_hnsw",
            f"USING hnsw (embedding {opclass(episode_dim)}) {with_}",
        ),
        index("vec_edge", "vec_edge_subject", "(subject_id)"),
        index("vec_edge", "vec_edge_object", "(object_id)"),
        index(
            "vec_edge",
            "vec_edge_time",
            "(valid_from, valid_to, recorded_at, expired_at)",
        ),
        index("vec_outbox", "vec_outbox_created", "(created_at)"),
        index("vec_outbox", "vec_outbox_episode", "(episode_id)"),
    ]
