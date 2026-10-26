"""Storage ports bound to one tenant."""

from dataclasses import dataclass

from ontogr_mem.consolidation.store import ConsStore
from ontogr_mem.ingestion.store import EpisodeStore
from ontogr_mem.ontology.ontology_store import OntologyStore
from ontogr_mem.resolution.resolver_store import ResolverStore
from ontogr_mem.retrieval.keyword import KeywordSearch
from ontogr_mem.temporal.temporal_store import TemporalStore
from ontogr_mem.vectorstore.vector_store import VectorStore


@dataclass(slots=True)
class TenantStores:
    """The ports that read and write one tenant's memory."""

    tenant_id: str
    episodes: EpisodeStore
    resolver_store: ResolverStore
    temporal_store: TemporalStore
    ontology_store: OntologyStore
    cons_store: ConsStore
    keywords: KeywordSearch
    vectors: VectorStore
