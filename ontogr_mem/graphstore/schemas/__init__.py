"""Registry of schema parts declared by graph storage and memory components."""

from ontogr_mem.consolidation import schema as consolidation_schema
from ontogr_mem.graphstore.indices import FulltextIndex, RangeIndex, SchemaPart
from ontogr_mem.graphstore.schemas.kernel import KERNEL
from ontogr_mem.ingestion import schema as ingestion_schema
from ontogr_mem.ontology import schema as ontology_schema
from ontogr_mem.resolution import schema as resolution_schema
from ontogr_mem.retrieval import schema as retrieval_schema
from ontogr_mem.temporal import schema as temporal_schema
from ontogr_mem.vectorstore.schema import VECTOR


def _part(
    name: str,
    labels: tuple[str, ...],
    ranges: tuple[tuple[str, str, bool], ...],
    requires: tuple[str, ...],
) -> SchemaPart:
    """Build a graphstore schema part from a component declaration."""
    return SchemaPart(
        name=name,
        labels=labels,
        ranges=tuple(RangeIndex(label, prop, edge) for label, prop, edge in ranges),
        requires=requires,
    )


EPISODES = _part(
    ingestion_schema.SCHEMA_NAME,
    ingestion_schema.LABELS,
    ingestion_schema.RANGES,
    ingestion_schema.REQUIRES,
)
ONTOLOGY = _part(
    ontology_schema.SCHEMA_NAME,
    ontology_schema.LABELS,
    ontology_schema.RANGES,
    ontology_schema.REQUIRES,
)
RESOLUTION = _part(
    resolution_schema.SCHEMA_NAME,
    resolution_schema.LABELS,
    resolution_schema.RANGES,
    resolution_schema.REQUIRES,
)
TEMPORAL = _part(
    temporal_schema.SCHEMA_NAME,
    temporal_schema.LABELS,
    temporal_schema.RANGES,
    temporal_schema.REQUIRES,
)
CONSOLIDATION = _part(
    consolidation_schema.SCHEMA_NAME,
    consolidation_schema.LABELS,
    consolidation_schema.RANGES,
    consolidation_schema.REQUIRES,
)

KEYWORD = SchemaPart(
    name=retrieval_schema.SCHEMA_NAME,
    fulltexts=tuple(
        FulltextIndex(label, props) for label, props in retrieval_schema.FULLTEXTS
    ),
    requires=retrieval_schema.REQUIRES,
)


ALL: tuple[SchemaPart, ...] = (
    KERNEL,
    EPISODES,
    ONTOLOGY,
    RESOLUTION,
    TEMPORAL,
    CONSOLIDATION,
)


__all__ = [
    "ALL",
    "CONSOLIDATION",
    "EPISODES",
    "KERNEL",
    "KEYWORD",
    "ONTOLOGY",
    "RESOLUTION",
    "TEMPORAL",
    "VECTOR",
]
