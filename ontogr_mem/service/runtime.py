"""Build the runtime components that operate on one Tenant."""

from dataclasses import dataclass, field

from ontogr_mem.consolidation import (
    ConsolidationRunner,
    default_stages,
)
from ontogr_mem.consolidation.config import ConsolidationSettings
from ontogr_mem.data_models.clock import Clock
from ontogr_mem.llm import LLMClient, LLMGate
from ontogr_mem.observability import Tracer
from ontogr_mem.ontology.manager import OntologyManager
from ontogr_mem.ontology.models import Ontology
from ontogr_mem.ontology.settings import OntologySettings
from ontogr_mem.resolution.config import ResolverSettings
from ontogr_mem.resolution.policy import ResolutionPolicy
from ontogr_mem.resolution.resolver import EntityResolver
from ontogr_mem.retrieval import Retriever
from ontogr_mem.retrieval.settings import RetrievalSettings
from ontogr_mem.service.relations import relation_policy
from ontogr_mem.temporal.adapter import EntityRefResolver
from ontogr_mem.temporal.engine import TemporalEngine
from ontogr_mem.temporal.reader import FactReader
from ontogr_mem.tenant_stores import TenantStores
from ontogr_mem.vectorstore import Embedder, FactVectorObserver, VectorWriter


_MAX_LLM_INPUT_TOKENS = 8192


@dataclass(slots=True)
class TenantRuntime:
    """Components shared by operations for one tenant."""

    stores: TenantStores
    reader: FactReader
    llm: LLMGate | None
    embedder: Embedder
    resolver: EntityResolver
    writer: VectorWriter
    retriever: Retriever
    ontology: OntologyManager
    clock: Clock
    _labels: dict[int, str] = field(default_factory=dict)

    @property
    def llm_available(self) -> bool:
        """Whether a language-model client was injected."""
        return self.llm is not None

    def engine(self, ontology: Ontology, refs: EntityRefResolver) -> TemporalEngine:
        """Build an engine with cardinality rules from the supplied ontology."""
        return TemporalEngine(
            self.stores.temporal_store,
            relation_policy(ontology),
            refs,
            self.clock.now,
            hooks=(FactVectorObserver(self.writer),),
        )

    def refresh_relation_labels(self, ontology: Ontology) -> None:
        """Update the labels used by retrieval's fallback fact renderer."""
        self._labels.clear()
        self._labels.update(
            {relation.type_id: relation.label for relation in ontology.relations}
        )

    def relation_label(self, type_id: int) -> str:
        """Return a relation label used in service responses."""
        return self._labels.get(type_id, "")

    async def consolidation(self) -> ConsolidationRunner:
        """Build a consolidation runner over this tenant's current ports."""
        ontology = await self.ontology.catalog()
        return ConsolidationRunner(
            default_stages(),
            self.stores.cons_store,
            self.clock.now,
            ConsolidationSettings(),
            resolution=self.resolver,
            temporal=self.engine(ontology, self.resolver.store),
            llm=self.llm,
        )


def build_tenant_runtime(
    stores: TenantStores,
    *,
    llm: LLMClient | None,
    embedder: Embedder,
    clock: Clock,
    policy: ResolutionPolicy,
    resolver_settings: ResolverSettings,
    retrieval_settings: RetrievalSettings,
    ontology_settings: OntologySettings | None = None,
    tracer: Tracer | None = None,
) -> TenantRuntime:
    """Wire the memory components around one tenant's stores."""
    gate = (
        LLMGate(llm, max_input_tokens=_MAX_LLM_INPUT_TOKENS, tracer=tracer)
        if llm
        else None
    )
    writer = VectorWriter(stores.vectors)
    reader = FactReader(stores.temporal_store)
    resolver = EntityResolver(
        stores.resolver_store,
        stores.vectors,
        policy,
        resolver_settings,
        clock,
        llm=gate,
    )
    ontology = OntologyManager(
        stores.ontology_store,
        gate,
        ontology_settings or OntologySettings(),
        clock,
    )
    labels: dict[int, str] = {}
    retriever = Retriever(
        vectors=stores.vectors,
        keywords=stores.keywords,
        facts=reader,
        entities=stores.resolver_store,
        episodes=stores.episodes,
        settings=retrieval_settings,
        relation_label=lambda type_id: labels.get(type_id, ""),
    )
    return TenantRuntime(
        stores,
        reader,
        gate,
        embedder,
        resolver,
        writer,
        retriever,
        ontology,
        clock,
        labels,
    )
