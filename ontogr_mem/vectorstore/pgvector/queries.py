"""Search SQL builders and vector text literals."""

from collections.abc import Sequence
from typing import Any

from ontogr_mem.data_models import OPEN_MICROS, FactStatus
from ontogr_mem.vectorstore.filters import reject_past_belief
from ontogr_mem.vectorstore.models import (
    SearchFilters,
    TemporalFilter,
    VectorKind,
)
from ontogr_mem.vectorstore.pgvector.pool import Conn
from ontogr_mem.vectorstore.pgvector.schema import (
    ID_COLUMN,
    cast_type,
    table_name,
)
from ontogr_mem.vectorstore.pgvector.settings import PgVectorSettings


def format_vector(vector: Sequence[float]) -> str:
    """Return the pgvector text literal of `vector`."""
    return "[" + ",".join(repr(float(value)) for value in vector) + "]"


def parse_vector(text: str) -> tuple[float, ...]:
    """Parse a pgvector or halfvec text literal."""
    body = text.strip()[1:-1]
    return tuple(float(value) for value in body.split(",")) if body else ()


def build_search(
    schema: str,
    kind: VectorKind,
    dim: int,
    *,
    vector: Sequence[float],
    k: int,
    filters: SearchFilters,
    min_score: float | None,
) -> tuple[str, list[object]]:
    """Build the search statement and its parameters.

    The inner query orders by distance so the HNSW index applies; the outer
    query applies `min_score` and re-sorts by score. The current-facts predicate
    is written with literals so a generic plan can prove the partial index.

    Raises:
        CapabilityError: If `filters` asks for `known_at`, which needs the
            graph.
    """
    reject_past_belief(filters)
    cast = cast_type(dim)
    id_column = ID_COLUMN[kind]
    params: list[object] = [format_vector(vector), k]
    predicates: list[str] = []

    def bind(value: object) -> str:
        params.append(value)
        return f"${len(params)}"

    if kind is VectorKind.NODE:
        if filters.entity_types is not None:
            predicates.append(
                f"entity_type = ANY({bind(sorted(filters.entity_types))}::int[])"
            )
        if filters.statuses is not None:
            predicates.append(f"status = ANY({bind(sorted(filters.statuses))}::text[])")
    if kind is VectorKind.EDGE:
        if filters.relation_type_ids is not None:
            predicates.append(
                "relation_type_id = ANY("
                f"{bind(sorted(filters.relation_type_ids))}::int[])"
            )
        if filters.subject_ids is not None:
            predicates.append(
                f"subject_id = ANY({bind(sorted(filters.subject_ids))}::text[])"
            )
        if filters.object_ids is not None:
            predicates.append(
                f"object_id = ANY({bind(sorted(filters.object_ids))}::text[])"
            )
        temporal = filters.temporal or TemporalFilter()
        if not temporal.history:
            predicates.append(
                f"status = '{FactStatus.ACTIVE.value}' AND expired_at = {OPEN_MICROS}"
            )
            if temporal.valid_at_us is not None:
                at = bind(temporal.valid_at_us)
                predicates.append(
                    f"valid_from <= {at}::bigint AND valid_to > {at}::bigint"
                )

    where = f"WHERE {' AND '.join(predicates)}" if predicates else ""
    inner = (
        f"SELECT {id_column} AS id, "
        f"(2 - (embedding <=> $1::text::{cast})) / 2 AS score "
        f"FROM {table_name(schema, kind)} {where} "
        f"ORDER BY embedding <=> $1::text::{cast} LIMIT $2"
    )
    outer = "SELECT id, score FROM (" + inner + ") t"
    if min_score is not None:
        outer += f" WHERE score >= {bind(float(min_score))}::float8"
    outer += " ORDER BY score DESC, id"
    return outer, params


async def search_rows(
    conn: Conn,
    schema: str,
    kind: VectorKind,
    sql: str,
    params: Sequence[object],
    *,
    k: int,
    filters: SearchFilters,
    exact: bool,
    iterative_scan: bool,
    settings: PgVectorSettings,
) -> list[Any]:
    """Search on an open read transaction with transaction-local planner settings."""
    await conn.execute(
        "SELECT set_config('statement_timeout', $1, true)",
        str(int(settings.search_timeout_s * 1000)),
    )
    use_exact = exact
    if not use_exact and filters.narrows:
        count = await conn.fetchval(
            "SELECT count(*) FROM (SELECT 1 FROM "
            f"{table_name(schema, kind)} LIMIT $1) x",
            settings.exact_threshold,
        )
        use_exact = bool(count < settings.exact_threshold)
    if use_exact:
        for setting in ("enable_indexscan", "enable_bitmapscan"):
            await conn.execute("SELECT set_config($1, 'off', true)", setting)
    else:
        await conn.execute(
            "SELECT set_config('hnsw.ef_search', $1, true)",
            str(min(max(k, 64), 1000)),
        )
        if iterative_scan and settings.iterative_scan != "off":
            await conn.execute(
                "SELECT set_config('hnsw.iterative_scan', $1, true)",
                settings.iterative_scan,
            )
            await conn.execute(
                "SELECT set_config('hnsw.max_scan_tuples', $1, true)",
                str(settings.max_scan_tuples),
            )
    return await conn.fetch(sql, *params)
