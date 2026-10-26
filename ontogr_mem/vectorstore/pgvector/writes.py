"""Transactional vector writes and validity mirrors for pgvector."""

from collections.abc import Sequence

from ontogr_mem.vectorstore.models import (
    FailedItem,
    FailReason,
    Row,
    SpaceSpec,
    ValidityUpdate,
    VectorKind,
)
from ontogr_mem.vectorstore.pgvector.pool import Conn, Pool
from ontogr_mem.vectorstore.pgvector.provision import parse_count
from ontogr_mem.vectorstore.pgvector.schema import ID_COLUMN, table_layout, table_name


def is_bad_row(error: BaseException) -> bool:
    """Return whether the server rejected row data or a row constraint."""
    value = getattr(error, "sqlstate", None)
    sqlstate = value if isinstance(value, str) else ""
    return sqlstate.startswith(("22", "23")) or type(error).__name__ in {
        "DataError",
        "IntegrityConstraintViolationError",
    }


def _write_sql(schema: str, spec: SpaceSpec) -> str:
    layout = table_layout(spec.kind, spec.dim)
    table = table_name(schema, spec.kind)
    columns = layout.columns
    names = ", ".join(column.name for column in columns)
    parameters = ", ".join(
        f"${index}::{column.array_type}[]"
        for index, column in enumerate(columns, start=1)
    )
    aliases = [f"c{index}" for index in range(len(columns))]
    selected = ", ".join(
        f"{alias}::{column.select_type}" if column.select_type is not None else alias
        for alias, column in zip(aliases, columns, strict=True)
    )
    assignments = ", ".join(
        f"{column.name} = EXCLUDED.{column.name}"
        for column in columns
        if column.name != layout.id_column
    )
    return (
        f"INSERT INTO {table} ({names}) SELECT {selected} "
        f"FROM unnest({parameters}) AS u({', '.join(aliases)}) "
        f"ON CONFLICT ({layout.id_column}) DO UPDATE SET {assignments}, "
        f"embedded_at = now() WHERE {layout.table}.text_hash "
        "IS DISTINCT FROM EXCLUDED.text_hash"
    )


async def _bulk(conn: Conn, timeout_s: float) -> None:
    await conn.execute(
        "SELECT set_config('statement_timeout', $1, true)",
        str(int(timeout_s * 1000)),
    )


async def _write_rows(
    conn: Conn, schema: str, spec: SpaceSpec, rows: Sequence[Row]
) -> int:
    """Write rows on an open connection."""
    sql = _write_sql(schema, spec)
    arrays = table_layout(spec.kind, spec.dim).arrays(rows)
    return parse_count(await conn.execute(sql, *arrays))


async def write_batch(
    pool: Pool, schema: str, spec: SpaceSpec, rows: Sequence[Row], timeout_s: float
) -> int:
    """Write a batch of rows in one transaction."""
    async with pool.acquire() as conn, conn.transaction():
        await _bulk(conn, timeout_s)
        return await _write_rows(conn, schema, spec, rows)


async def write_one_by_one(
    pool: Pool, schema: str, spec: SpaceSpec, rows: Sequence[Row], timeout_s: float
) -> tuple[int, tuple[FailedItem, ...]]:
    """Write accepted rows in one transaction, isolating rejections by savepoint."""
    written = 0
    rejected: list[FailedItem] = []
    async with pool.acquire() as conn, conn.transaction():
        await _bulk(conn, timeout_s)
        for row in rows:
            try:
                async with conn.transaction():
                    written += await _write_rows(conn, schema, spec, [row])
            except Exception as error:
                if not is_bad_row(error):
                    raise
                rejected.append(FailedItem(row.id, FailReason.REJECTED, str(error)))
    return written, tuple(rejected)


async def update_validity(
    pool: Pool, schema: str, updates: Sequence[ValidityUpdate], timeout_s: float
) -> int:
    """Set the mirror on existing edge rows, in id order to avoid deadlocks."""
    ordered = sorted(updates, key=lambda update: update.fact_id)
    sql = f"""UPDATE {table_name(schema, "vec_edge")} AS e
            SET status = u.st, valid_from = u.vf, valid_to = u.vt,
                recorded_at = u.ra, expired_at = u.ea
            FROM unnest($1::text[], $2::text[], $3::bigint[], $4::bigint[],
              $5::bigint[], $6::bigint[]) AS u(i, st, vf, vt, ra, ea)
            WHERE e.fact_id = u.i"""
    args: list[list[object]] = [
        [update.fact_id for update in ordered],
        [update.validity.status for update in ordered],
        [update.validity.valid_from_us for update in ordered],
        [update.validity.valid_to_us for update in ordered],
        [update.validity.recorded_at_us for update in ordered],
        [update.validity.expired_at_us for update in ordered],
    ]
    async with pool.acquire() as conn, conn.transaction():
        await _bulk(conn, timeout_s)
        return parse_count(await conn.execute(sql, *args))


async def delete(pool: Pool, schema: str, kind: VectorKind, ids: Sequence[str]) -> int:
    """Delete by id without opening a transaction; ignore missing ids."""
    id_column = ID_COLUMN[kind]
    async with pool.acquire() as conn:
        return parse_count(
            await conn.execute(
                f"DELETE FROM {table_name(schema, kind)} "
                f"WHERE {id_column} = ANY($1::text[])",
                sorted(ids),
            )
        )
