"""Ontology store on the graph backends."""

from collections.abc import Sequence

from ontogr_mem.data_models import to_micros
from ontogr_mem.graphstore.cypher import TenantScope
from ontogr_mem.graphstore.executor import QuerySpec
from ontogr_mem.graphstore.ident import quote
from ontogr_mem.graphstore.protocol import GraphStore
from ontogr_mem.graphstore.types import GraphHandle, Guard, Node
from ontogr_mem.ontology.errors import MissingVersionError, VersionConflict
from ontogr_mem.ontology.models import ChangeSet, Ontology


HEAD = "OntologyHead"
SNAPSHOT = "OntologySnapshot"
CHANGE_SET = "OntologyChangeSet"


def _version_id(version: int) -> str:
    return f"v{version}"


def _snapshot_node(ontology: Ontology) -> Node:
    return Node(
        _version_id(ontology.version),
        SNAPSHOT,
        {
            "version": ontology.version,
            "content_hash": ontology.content_hash,
            "parent_version": ontology.parent_version,
            "catalog": ontology.model_dump_json(),
        },
    )


class GraphOntologyStore:
    """`OntologyStore` over a tenant's graph.

    Args:
        store: The graph store that owns the tenant's graph.
        handle: Handle of the tenant's graph.
        tenant_keyed: Explicit scope for a custom runner without backend scope.
    """

    def __init__(
        self,
        store: GraphStore,
        handle: GraphHandle,
        *,
        tenant_keyed: bool | None = None,
    ) -> None:
        """Bind the store to a tenant's graph."""
        self._store = store
        self._handle = handle
        self._runner = store.runner(handle)
        self._scope = (
            TenantScope.from_mode(tenant_keyed)
            if tenant_keyed is not None
            else self._runner.scope
        )

    async def get(self, version: int | None = None) -> Ontology:
        """Return a version of the ontology, or the head if `version` is None."""
        head = await self._head_version()
        if head is None:
            await self._create_genesis()
            head = 0
        wanted = head if version is None else version
        rows = await self._store.get_nodes(
            self._handle, SNAPSHOT, [_version_id(wanted)]
        )
        if not rows:
            msg = f"ontology version {wanted} does not exist"
            raise MissingVersionError(msg)
        return Ontology.model_validate_json(rows[0].props["catalog"])

    async def commit(self, new: Ontology, change_set: ChangeSet) -> None:
        """Store `new` as the next version and move the head to it."""
        await self.get()
        applied = await self._store.guarded_write(
            self._handle,
            Guard(
                HEAD,
                "head",
                expect={"version": change_set.base_version},
                set={"version": new.version},
            ),
            [
                _snapshot_node(new),
                Node(
                    _version_id(new.version),
                    CHANGE_SET,
                    {
                        "base_version": change_set.base_version,
                        "actor": change_set.actor,
                        "reason": change_set.reason,
                        "channel": change_set.channel.value,
                        "created_at": to_micros(change_set.created_at),
                        "change_set": change_set.model_dump_json(),
                    },
                ),
            ],
        )
        if not applied:
            raise VersionConflict(change_set.base_version)

    async def history(self) -> Sequence[ChangeSet]:
        """Return every committed change set, oldest first."""
        rows = await self._runner.run(
            QuerySpec(
                f"MATCH (c:{quote(CHANGE_SET, 'label')} {self._scope.pattern()}) "
                "RETURN c.change_set AS change_set ORDER BY c.base_version",
                {"tenant": self._handle.tenant},
                read_only=True,
                idempotent=True,
                tag="ontology_history",
            )
        )
        return [ChangeSet.model_validate_json(r["change_set"]) for r in rows.rows]

    async def _head_version(self) -> int | None:
        rows = await self._store.get_nodes(self._handle, HEAD, ["head"])
        if not rows:
            return None
        return int(rows[0].props["version"])

    async def _create_genesis(self) -> None:
        # Concurrent first readers may race here. Only the writer whose guard
        # finds no `initialized` property writes the genesis snapshot.
        await self._store.guarded_write(
            self._handle,
            Guard(
                HEAD,
                "head",
                expect={"initialized": None},
                set={"initialized": True, "version": 0},
                create_with={"version": 0},
            ),
            [_snapshot_node(Ontology.genesis())],
        )
