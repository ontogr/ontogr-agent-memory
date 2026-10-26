"""Project a catalog to the dense-id view that extraction reads.

The projection is looser than ``validate``: ordered pairs become source and
target sets, so the direction check also accepts crossed pairs. Use
``validate`` for the authoritative pattern check.
"""

from dataclasses import dataclass

from ontogr_mem.data_models.type_catalog import (
    UNKNOWN_TYPE_ID,
    EntityType,
    RelationType,
    TypeCatalog,
    TypeMode,
    TypeView,
)
from ontogr_mem.ontology.errors import DenseIdError
from ontogr_mem.ontology.models import UNKNOWN_ID, Kind, Mode, Ontology, Status


@dataclass(frozen=True, slots=True)
class Projection:
    """The dense-id view plus the maps from its dense ids back to stable type ids.

    Dense ids depend on which types were active when the projection was built, so a
    dense id is only meaningful with the projection that issued it.
    """

    view: TypeView
    catalog_version: int
    content_hash: str
    entity_ids: tuple[int, ...]
    relation_ids: tuple[int, ...]

    def stable_entity_id(self, dense_id: int) -> int:
        """Return the stable entity type id for a dense id; -1 is Unknown.

        Raises:
            DenseIdError: If the dense id is outside this projection.
        """
        if dense_id == UNKNOWN_TYPE_ID:
            return UNKNOWN_ID
        if not 0 <= dense_id < len(self.entity_ids):
            msg = f"dense entity id {dense_id} is outside the projection"
            raise DenseIdError(msg)
        return self.entity_ids[dense_id]

    def stable_relation_id(self, dense_id: int) -> int:
        """Return the stable relation type id for a dense id.

        Raises:
            DenseIdError: If the dense id is outside this projection.
        """
        if not 0 <= dense_id < len(self.relation_ids):
            msg = f"dense relation id {dense_id} is outside the projection"
            raise DenseIdError(msg)
        return self.relation_ids[dense_id]


_MODE = {Mode.STRICT: TypeMode.STRICT, Mode.PREFERRED: TypeMode.PREFERRED}


def project(cat: Ontology, actor_type_label: str = "Person") -> Projection:
    """Project a catalog onto the dense-id view that extraction reads.

    Only active types are included, and Unknown is left out because the view
    represents it as id -1. Each relation's ordered patterns become source and target
    sets, expanded with descendant types. This is looser than ``validate``: a crossed
    pair can pass the direction check. The mode is taken from ``relation_mode``.

    Args:
        cat: The catalog to project.
        actor_type_label: Entity label used for message actors in the view.

    Returns:
        The view plus the maps from dense ids back to ``type_id``.
    """
    ents = [
        e for e in cat.entities if e.status is Status.ACTIVE and e.type_id != UNKNOWN_ID
    ]
    rels = [r for r in cat.relations if r.status is Status.ACTIVE]
    dense = {e.type_id: i for i, e in enumerate(ents)}

    def expand(type_id: int) -> set[int]:
        """Return the dense ids of a type and of its active descendants."""
        out = {dense[type_id]} if type_id in dense else set()
        for e in ents:
            if type_id in cat.ancestors(e.type_id) and e.type_id in dense:
                out.add(dense[e.type_id])
        return out

    rel_types: list[RelationType] = []
    for i, r in enumerate(rels):
        srcs: set[int] = set()
        tgts: set[int] = set()
        for p in r.patterns:
            if p.status is Status.ACTIVE:
                srcs |= expand(cat.follow(Kind.ENTITY, p.src))
                tgts |= expand(cat.follow(Kind.ENTITY, p.tgt))
        rel_types.append(
            RelationType(
                id=i,
                label=r.label,
                description=r.description,
                source_types=frozenset(srcs),
                target_types=frozenset(tgts),
            )
        )
    old = TypeCatalog(
        [
            EntityType(id=i, label=e.label, description=e.description)
            for i, e in enumerate(ents)
        ],
        rel_types,
        actor_type_label=actor_type_label,
    )
    return Projection(
        view=TypeView(old, _MODE.get(cat.relation_mode, TypeMode.OPEN)),
        catalog_version=cat.version,
        content_hash=cat.content_hash,
        entity_ids=tuple(e.type_id for e in ents),
        relation_ids=tuple(r.type_id for r in rels),
    )
