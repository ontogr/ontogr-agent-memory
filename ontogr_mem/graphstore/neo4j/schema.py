"""Create and verify the indexes and constraints of the shared Neo4j database.

Constraints and indexes belong to the database, not to a tenant, so each
unique constraint covers `(_tenant, property)`. The set of rules comes from
the shared plan in `graphstore.indices`, so every backend provisions the same
schema.
"""

import asyncio
import time
from collections.abc import Iterator
from dataclasses import dataclass

from ontogr_mem.graphstore.errors import (
    AlreadyExistsError,
    SchemaDriftError,
    ValidationError,
)
from ontogr_mem.graphstore.executor import QuerySpec
from ontogr_mem.graphstore.ident import quote
from ontogr_mem.graphstore.indices import DriftReport, SchemaPart, plan
from ontogr_mem.graphstore.neo4j.runner import Neo4jRunner
from ontogr_mem.graphstore.reserved import TENANT


@dataclass(frozen=True)
class _Rule:
    name: str
    kind: str
    target: str
    ddl: str


def _rules(schema: SchemaPart) -> Iterator[_Rule]:
    """Yield the DDL rules that create the plan of `schema`."""
    for spec in plan(schema):
        target = f"{spec.label}.{spec.prop}"
        if spec.kind == "fulltext":
            if not spec.props:
                raise ValidationError("a fulltext index needs at least one property")
            name = f"am_ft_{spec.label}"
            properties = ", ".join(
                f"n.{quote(prop, 'property key')}" for prop in spec.props
            )
            yield _Rule(
                name,
                "fulltext",
                f"{spec.label}.{','.join(spec.props)}",
                f"CREATE FULLTEXT INDEX {quote(name)} IF NOT EXISTS FOR "
                f"(n:{quote(spec.label, 'label')}) ON EACH [{properties}]",
            )
            continue
        if spec.kind == "unique":
            name = f"am_uq_{spec.label}_{spec.prop}"
            yield _Rule(
                name,
                "unique",
                target,
                f"CREATE CONSTRAINT {quote(name)} IF NOT EXISTS FOR "
                f"(n:{quote(spec.label, 'label')}) "
                f"REQUIRE (n.{quote(TENANT)}, n.{quote(spec.prop, 'property key')}) "
                "IS UNIQUE",
            )
            continue
        name = f"am_ix_{spec.label}_{spec.prop}"
        pattern = (
            f"()-[e:{quote(spec.label, 'relationship type')}]-()"
            if spec.edge
            else f"(e:{quote(spec.label, 'label')})"
        )
        yield _Rule(
            name,
            "range",
            target,
            f"CREATE INDEX {quote(name)} IF NOT EXISTS FOR {pattern} "
            f"ON (e.{quote(spec.prop, 'property key')})",
        )


class SchemaManager:
    """Create and check range indexes, fulltext indexes and constraints."""

    def __init__(
        self, runner: Neo4jRunner, poll_s: float = 0.2, wait_s: float = 30.0
    ) -> None:
        """Bind to a runner."""
        self._runner = runner
        self._poll = poll_s
        self._wait = wait_s

    async def provision(self, schema: SchemaPart) -> None:
        """Create everything missing. Safe to call repeatedly and concurrently.

        Raises:
            SchemaDriftError: If an index is still not online after the wait.
        """
        for rule in _rules(schema):
            try:
                await self._runner.run(
                    QuerySpec(
                        rule.ddl,
                        idempotent=True,
                        auto_commit=True,
                        tag=f"ddl:{rule.kind}",
                    )
                )
            except AlreadyExistsError:
                # Two clients creating the same rule at once: the loser is told
                # an equivalent one exists.
                continue
        deadline = time.monotonic() + self._wait
        while True:
            report = await self.verify(schema)
            if not report.not_operational:
                return
            if time.monotonic() > deadline:
                msg = "indexes not online after wait: " + "; ".join(
                    report.not_operational
                )
                raise SchemaDriftError(msg)
            await asyncio.sleep(self._poll)

    async def verify(self, schema: SchemaPart) -> DriftReport:
        """Compare the live database to the declared schema."""
        indexes = await self._names("SHOW INDEXES YIELD name, state", "state")
        constraints = await self._names("SHOW CONSTRAINTS YIELD name", None)
        missing: list[str] = []
        pending: list[str] = []
        for rule in _rules(schema):
            label = f"{rule.kind} {rule.target}"
            if rule.kind == "unique":
                if rule.name not in constraints:
                    missing.append(label)
            elif rule.name not in indexes:
                missing.append(label)
            elif indexes[rule.name] != "ONLINE":
                pending.append(label)
        return DriftReport(tuple(missing), tuple(pending))

    async def _names(self, cypher: str, state: str | None) -> dict[str, str | None]:
        rows = await self._runner.run(
            QuerySpec(
                cypher, read_only=True, idempotent=True, auto_commit=True, tag="schema"
            )
        )
        return {r["name"]: r[state] if state else None for r in rows.rows}
