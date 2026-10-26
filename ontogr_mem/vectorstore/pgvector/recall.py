"""Recall self-test: compare index results with the exact scan."""

from collections.abc import Sequence
from dataclasses import dataclass

from ontogr_mem.vectorstore.models import SearchFilters, VectorKind
from ontogr_mem.vectorstore.pgvector.errors import RecallBelowThresholdError
from ontogr_mem.vectorstore.pgvector.store import PgBackend
from ontogr_mem.vectorstore.vecmath import to_unit
from ontogr_mem.vectorstore.vector_store import VectorStore


@dataclass(frozen=True, slots=True)
class RecallReport:
    """Outcome of `recall_selftest`.

    Attributes:
        kind: The kind of vector the test searched.
        queries: Number of query vectors.
        k: Hits requested per query.
        recall: Mean fraction of exact results the index returned.
        min_recall: Threshold the run had to reach.
        shortfalls: Queries where the index returned fewer rows than exact.
    """

    kind: VectorKind
    queries: int
    k: int
    recall: float
    min_recall: float
    shortfalls: int

    @property
    def passed(self) -> bool:
        """Return true when mean recall reaches `min_recall`."""
        return self.recall >= self.min_recall


async def recall_selftest(
    store: VectorStore,
    kind: VectorKind,
    queries: Sequence[Sequence[float]],
    *,
    k: int,
    min_recall: float,
    filters: SearchFilters | None = None,
    force_exact_on_failure: bool = True,
) -> RecallReport:
    """Compare the index path with the exact scan on a sample of queries.

    On failure, the backend switches `kind` to the exact scan, so queries stay
    correct while the index is investigated. Callers that must fail a
    build use `require_recall`.

    Args:
        store: The store to probe.
        kind: The kind of vector to search.
        queries: Query vectors of the store's dimension.
        k: Hits requested per query.
        min_recall: Mean recall the run must reach.
        filters: Filters applied to both paths.
        force_exact_on_failure: Switch `kind` to the exact scan when the run
            fails.

    Returns:
        The report; it never raises for low recall.

    Raises:
        TypeError: If the store is not backed by `PgBackend`.
    """
    backend = store.backend
    if not isinstance(backend, PgBackend):
        msg = f"recall_selftest needs a PgBackend, got {type(backend).__name__}"
        raise TypeError(msg)
    spec = store.specs[kind]
    chosen = (filters or SearchFilters()).for_kind(kind)
    total = 0.0
    shortfalls = 0
    for query in queries:
        unit = to_unit(query, spec.dim)
        ann = await backend.search(kind, unit, k, chosen, None, exact=False)
        truth = await backend.search(kind, unit, k, chosen, None, exact=True)
        want = {object_id for object_id, _ in truth}
        got = {object_id for object_id, _ in ann}
        total += len(want & got) / len(want) if want else 1.0
        shortfalls += len(ann) < len(truth)
    recall = total / len(queries) if queries else 1.0
    report = RecallReport(kind, len(queries), k, recall, min_recall, shortfalls)
    store.metrics.inc(f"recall.{'passed' if report.passed else 'failed'}.{kind.value}")
    if not report.passed and force_exact_on_failure:
        backend.force_exact.add(kind)
    return report


def require_recall(report: RecallReport) -> None:
    """Raise `RecallBelowThresholdError` when the report did not pass.

    Raises:
        RecallBelowThresholdError: If mean recall is below the threshold.
    """
    if not report.passed:
        msg = (
            f"{report.kind.value}: recall {report.recall:.3f} < "
            f"{report.min_recall:.3f} ({report.shortfalls}/{report.queries} "
            "queries returned too few rows)"
        )
        raise RecallBelowThresholdError(msg)
