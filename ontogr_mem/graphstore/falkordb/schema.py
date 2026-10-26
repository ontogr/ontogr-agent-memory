"""Create and verify the indexes and constraints of a FalkorDB graph.

The set of rules comes from the shared plan in `graphstore.indices`, so every
backend provisions the same schema. FalkorDB backs a uniqueness constraint
with a range index and creates that index itself.
"""

import asyncio
import time
from collections.abc import Sequence
from typing import Any

from ontogr_mem.graphstore.errors import (
    AlreadyExistsError,
    SchemaDriftError,
    ValidationError,
)
from ontogr_mem.graphstore.executor import QuerySpec
from ontogr_mem.graphstore.falkordb.runner import QueryRunner
from ontogr_mem.graphstore.ident import ident, quote
from ontogr_mem.graphstore.indices import (
    DriftReport,
    IndexResult,
    IndexSpec,
    SchemaPart,
    plan,
)


class SchemaManager:
    """Create and check range indexes, fulltext indexes and constraints."""

    def __init__(
        self, runner: QueryRunner, poll_s: float = 0.2, wait_s: float = 15.0
    ) -> None:
        """Bind to a runner."""
        self._runner = runner
        self._poll = poll_s
        self._wait = wait_s

    async def provision(self, graph: str, schema: SchemaPart) -> list[IndexResult]:
        """Create everything missing. Safe to call repeatedly.

        Raises:
            SchemaDriftError: If an index is still not operational after the
                wait budget.
            ValidationError: If a label or property of the plan is unsafe.
        """
        specs = plan(schema)
        for spec in specs:
            _validate(spec)
        await self._create_graph(graph)
        indexes = await self._indexes(graph)
        constraints = await self._constraints(graph)
        results: list[IndexResult] = []
        for spec in specs:
            properties = ",".join(spec.props) if spec.kind == "fulltext" else spec.prop
            target = f"{spec.label}.{properties}"
            if spec.kind == "fulltext":
                if all(
                    self._find(indexes, spec.label, prop, False, "FULLTEXT")
                    for prop in spec.props
                ):
                    results.append(IndexResult("fulltext", target, "exists"))
                else:
                    args = ", ".join(f"'{name}'" for name in (spec.label, *spec.props))
                    results.append(
                        await self._ddl(
                            graph,
                            "fulltext",
                            target,
                            f"CALL db.idx.fulltext.createNodeIndex({args})",
                        )
                    )
                continue
            if spec.kind == "range":
                results.append(await self._range(graph, spec, target, indexes))
                continue
            results.append(await self._unique(graph, spec, target, constraints))
        await self._wait_operational(graph, schema)
        return results

    async def verify(self, graph: str, schema: SchemaPart) -> DriftReport:
        """Compare the live graph to the declared schema."""
        indexes = await self._indexes(graph)
        constraints = await self._constraints(graph)
        missing: list[str] = []
        pending: list[str] = []
        for spec in plan(schema):
            target = f"{spec.label}.{spec.prop}"
            if spec.kind == "fulltext":
                for prop in spec.props:
                    index = self._find(indexes, spec.label, prop, False, "FULLTEXT")
                    if index is None:
                        missing.append(f"fulltext {spec.label}.{prop}")
                    elif index.get("status") != "OPERATIONAL":
                        pending.append(f"fulltext {spec.label}.{prop}")
                continue
            if spec.kind == "range":
                index = self._find(indexes, spec.label, spec.prop, spec.edge)
                if index is None:
                    missing.append(f"range {target}")
                elif index.get("status") != "OPERATIONAL":
                    pending.append(f"range {target}")
                continue
            status = constraints.get((spec.label, (spec.prop,)))
            if status is None:
                missing.append(f"unique {target}")
            elif status != "OPERATIONAL":
                pending.append(f"unique {target}")
        return DriftReport(tuple(missing), tuple(pending))

    async def _create_graph(self, graph: str) -> None:
        """Create the graph if it does not exist yet.

        A read-only query on a graph that does not exist fails, so the first
        statement is a write. It returns a row and writes nothing.
        """
        await self._runner.run(
            graph,
            QuerySpec("RETURN 1 AS ok", idempotent=True, tag="ddl:bootstrap"),
        )

    async def _indexes(self, graph: str) -> list[dict[str, Any]]:
        rows = await self._runner.run(
            graph,
            QuerySpec(
                "CALL db.indexes()", read_only=True, idempotent=True, tag="indexes"
            ),
        )
        return rows.rows

    async def _constraints(self, graph: str) -> dict[tuple[str, tuple[str, ...]], Any]:
        """Return the unique constraints of the graph, by label and property."""
        listed = await self._runner.list_constraints(graph)
        return {
            (c["label"], tuple(c["properties"])): c.get("status")
            for c in listed
            if c.get("type") == "UNIQUE"
        }

    @staticmethod
    def _find(
        indexes: Sequence[dict[str, Any]],
        label: str,
        prop: str,
        edge: bool,
        kind: str = "RANGE",
    ) -> dict[str, Any] | None:
        entity = "RELATIONSHIP" if edge else "NODE"
        for index in indexes:
            if index["label"] != label or index.get("entitytype") != entity:
                continue
            if kind in ((index.get("types") or {}).get(prop) or []):
                return index
        return None

    async def _range(
        self,
        graph: str,
        spec: IndexSpec,
        target: str,
        indexes: Sequence[dict[str, Any]],
    ) -> IndexResult:
        if self._find(indexes, spec.label, spec.prop, spec.edge):
            return IndexResult("range", target, "exists")
        pattern = (
            f"()-[e:{quote(spec.label, 'relationship type')}]-()"
            if spec.edge
            else f"(e:{quote(spec.label, 'label')})"
        )
        return await self._ddl(
            graph,
            "range",
            target,
            f"CREATE INDEX FOR {pattern} ON (e.{quote(spec.prop, 'property key')})",
        )

    async def _unique(
        self,
        graph: str,
        spec: IndexSpec,
        target: str,
        constraints: dict[tuple[str, tuple[str, ...]], Any],
    ) -> IndexResult:
        if (spec.label, (spec.prop,)) in constraints:
            return IndexResult("unique", target, "exists")
        try:
            await self._runner.create_unique_constraint(graph, spec.label, spec.prop)
        except AlreadyExistsError:
            return IndexResult("unique", target, "exists")
        return IndexResult("unique", target, "created")

    async def _ddl(
        self, graph: str, kind: str, target: str, cypher: str
    ) -> IndexResult:
        try:
            await self._runner.run(
                graph, QuerySpec(cypher, idempotent=True, tag=f"ddl:{kind}")
            )
        except AlreadyExistsError:
            return IndexResult(kind, target, "exists")
        return IndexResult(kind, target, "created")

    async def _wait_operational(self, graph: str, schema: SchemaPart) -> None:
        deadline = time.monotonic() + self._wait
        while True:
            report = await self.verify(graph, schema)
            if not report.not_operational:
                return
            if time.monotonic() > deadline:
                msg = "indexes not operational after wait: " + "; ".join(
                    report.not_operational
                )
                raise SchemaDriftError(msg)
            await asyncio.sleep(self._poll)


def _validate(spec: IndexSpec) -> None:
    """Raise if the names of `spec` are unsafe to interpolate.

    Raises:
        ValidationError: If a label or property of the spec is unsafe.
    """
    if spec.kind == "unique" or not spec.edge:
        ident(spec.label, "label")
    else:
        ident(spec.label, "relationship type")
    if spec.kind == "fulltext":
        if not spec.props:
            raise ValidationError("a fulltext index needs at least one property")
        for prop in spec.props:
            ident(prop, "property key")
    else:
        ident(spec.prop, "property key")
