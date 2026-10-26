"""The ontology manager, which combines the store with the pure logic."""

import asyncio
from collections.abc import Callable, Sequence

from ontogr_mem.data_models import Clock
from ontogr_mem.llm import LLMClient
from ontogr_mem.ontology.apply import apply_changeset, check_catalog
from ontogr_mem.ontology.diff import diff
from ontogr_mem.ontology.drafting import DraftRun, run_drafting
from ontogr_mem.ontology.errors import InvalidChange, OntologyError, VersionConflict
from ontogr_mem.ontology.models import (
    AddEntity,
    AddRelation,
    ChangeSet,
    Channel,
    Kind,
    Ontology,
    Op,
    Origin,
    ProposalStatus,
    RenderedView,
    Unmatched,
)
from ontogr_mem.ontology.ontology_store import OntologyStore
from ontogr_mem.ontology.projection import Projection, project
from ontogr_mem.ontology.proposals import aggregate
from ontogr_mem.ontology.resolve import How, TypeResolver
from ontogr_mem.ontology.settings import OntologySettings
from ontogr_mem.ontology.stale import stale_episodes
from ontogr_mem.ontology.validate import (
    Context,
    ExtractedItems,
    ValidationReport,
    validate,
)


class OntologyManager:
    """Owns a tenant's ontology, its history and its proposal queue.

    Args:
        store: Storage for the tenant's ontology.
        llm: Client used to draft proposals.
        settings: Thresholds and retry counts.
        clock: Source of the time stamped on change sets and decisions.
    """

    def __init__(
        self,
        store: OntologyStore,
        llm: LLMClient | None,
        settings: OntologySettings,
        clock: Clock,
    ) -> None:
        """Wire the store, the model client, the settings and the clock."""
        self._store = store
        self._llm = llm
        self._settings = settings
        self._clock = clock
        # Writers in one process queue here instead of racing on the head.
        self._commit_lock = asyncio.Lock()

    async def catalog(self, version: int | None = None) -> Ontology:
        """Return a snapshot of the ontology; the head if `version` is None."""
        return await self._store.get(version)

    async def history(self) -> Sequence[ChangeSet]:
        """Return every committed change set, oldest first."""
        return await self._store.history()

    async def unmatched(self) -> Sequence[Unmatched]:
        """Return every stored unmatched row."""
        return await self._store.unmatched()

    async def extraction_view(self) -> Projection:
        """Return the dense-id view of the head that extraction reads."""
        return project(await self._store.get())

    async def apply(self, change_set: ChangeSet) -> Ontology:
        """Check `change_set` against the head and commit it.

        Raises:
            InvalidChange: If the result breaks a rule.
            VersionConflict: If the head moved since `change_set.base_version`.
                Re-read the catalog and apply again.
        """
        head = await self._store.get()
        stamped = self._stamp(change_set)
        new = apply_changeset(head, stamped)
        await self._store.commit(new, stamped)
        return new

    async def load_catalog(self, catalog: Ontology) -> Ontology:
        """Load a whole user-built catalog into a tenant still at version 0.

        Types are matched by label, never by the ids in `catalog`.

        Raises:
            InvalidChange: If the tenant already has a catalog or `catalog` breaks
                a rule.
        """
        head = await self._store.get()
        if head.version != 0:
            raise InvalidChange(
                "tenant already has a catalog",
                hint="load_catalog is for a new tenant; use apply",
            )
        check_catalog(catalog)
        shaped = catalog.model_copy(
            update={
                "entities": sorted(
                    catalog.entities,
                    key=lambda e: (len(catalog.ancestors(e.type_id)), e.type_id),
                )
            }
        )
        change_set = diff(head, shaped).model_copy(
            update={
                "actor": "load",
                "reason": "load catalog",
                "channel": Channel.IMPORT,
            }
        )
        return await self.apply(change_set)

    async def record_unmatched(self, episode_id: str, rows: list[Unmatched]) -> None:
        """Make `rows` the unmatched set of one episode; empty clears it."""
        await self._store.replace_unmatched(episode_id, rows)

    async def process_extraction(
        self,
        items: ExtractedItems,
        context: Context,
        *,
        entity_view: RenderedView | None = None,
        relation_view: RenderedView | None = None,
    ) -> ValidationReport:
        """Validate one episode's extraction and act on the ontology mode.

        In preferred mode the unmatched rows replace the episode's stored rows. In
        open mode the new labels are registered as types, the items are validated
        again against the new catalog, and the stored rows come from that second
        pass. They therefore describe the catalog the episode was placed in. In
        strict mode nothing is recorded or registered.

        Returns:
            The report against the catalog after any registration.
        """
        head = await self._store.get()
        report = validate(head, items, context, entity_view, relation_view)
        if report.to_register:
            new = await self._register(report.to_register)
            report = validate(new, items, context, entity_view, relation_view)
        await self._store.replace_unmatched(context.episode_id, report.unmatched)
        return report

    async def propose(self) -> DraftRun:
        """Aggregate unmatched rows, draft one batch, check it and store it.

        Raises:
            OntologyError: If the model call fails.
        """
        llm = self._llm
        if llm is None:
            raise OntologyError("language model is unavailable")
        cat, rows, proposals = await asyncio.gather(
            self._store.get(), self._store.unmatched(), self._store.proposals()
        )
        rejected = [p for p in proposals if p.status is ProposalStatus.REJECTED]
        pending = {
            label
            for p in proposals
            if p.status is ProposalStatus.PENDING
            for label in p.labels
        }
        groups = [
            g
            for g in aggregate(
                cat,
                list(rows),
                rejected,
                min_support=self._settings.min_support,
                regrow_factor=self._settings.regrow_factor,
            )
            if g.label not in pending
        ]
        wanted = sorted(g.label for g in groups)
        attempt = sum(
            1
            for p in proposals
            if sorted(p.labels) == wanted and p.base_version == cat.version
        )
        run = await run_drafting(
            cat,
            groups,
            llm,
            prompt_version=self._settings.prompt_version,
            max_retries=self._settings.max_draft_retries,
            attempt=attempt,
        )
        for proposal in run.proposals:
            await self._store.save_proposal(proposal)
        return run

    async def decide(
        self, proposal_id: str, *, accept: bool, actor: str, reason: str
    ) -> Ontology | None:
        """Accept or reject a pending proposal.

        A rejection stores the decision and returns None. An acceptance commits the
        proposal's ops on the current head and returns the new catalog. The ops are
        the ones the proposal was drafted with; the labels they name are resolved
        against the catalog as it is when they are applied. If the ops no longer
        apply, the proposal is stored as superseded and the error is raised.

        Raises:
            InvalidChange: If there is no pending proposal with that id, or it no
                longer applies.
        """
        proposal = await self._store.get_proposal(proposal_id)
        if proposal is None or proposal.status is not ProposalStatus.PENDING:
            raise InvalidChange("no pending proposal", item=proposal_id)
        decided = {
            "decided_by": actor,
            "decided_at": self._clock.now(),
            "reason": reason,
        }
        if not accept:
            await self._store.save_proposal(
                proposal.model_copy(
                    update={"status": ProposalStatus.REJECTED, **decided}
                )
            )
            return None
        ops: list[Op] = [
            op.model_copy(update={"origin": Origin.ACCEPTED})
            if isinstance(op, AddEntity | AddRelation)
            else op
            for op in proposal.ops
        ]
        try:
            new = await self._commit_rebased(
                lambda _head: ops,
                actor,
                reason or f"proposal {proposal_id}",
                Channel.PROPOSAL,
            )
        except InvalidChange as exc:
            await self._store.save_proposal(
                proposal.model_copy(
                    update={
                        "status": ProposalStatus.SUPERSEDED,
                        **decided,
                        "reason": str(exc)[:500],
                    }
                )
            )
            raise
        await self._store.save_proposal(
            proposal.model_copy(update={"status": ProposalStatus.ACCEPTED, **decided})
        )
        return new

    async def stale_episodes(self, from_version: int, to_version: int) -> set[str]:
        """Return the episodes to extract again between two stored versions."""
        old = await self._store.get(from_version)
        new = await self._store.get(to_version)
        return stale_episodes(
            old,
            new,
            await self._store.unmatched(),
            await self._store.known_episode_ids(),
        )

    async def _register(self, labels: tuple[tuple[Kind, str], ...]) -> Ontology:
        """Add the labels that are still missing as auto-accepted types."""

        def ops_for(head: Ontology) -> list[Op]:
            ops: list[Op] = []
            for kind, label in labels:
                found = TypeResolver(head, kind).resolve(label).match
                if found is not None and found.how in (How.EXACT, How.ALIAS):
                    continue
                if kind is Kind.ENTITY:
                    ops.append(AddEntity(label=label, origin=Origin.AUTO_ACCEPTED))
                else:
                    ops.append(AddRelation(label=label, origin=Origin.AUTO_ACCEPTED))
            return ops

        return await self._commit_rebased(
            ops_for, "extraction", "registered in open mode", Channel.OPEN_MODE
        )

    async def _commit_rebased(
        self,
        ops_for: Callable[[Ontology], list[Op]],
        actor: str,
        reason: str,
        channel: Channel,
    ) -> Ontology:
        """Commit ops on the current head, retrying when another writer wins.

        `ops_for` builds the ops from the head read on each attempt, so a retry
        sees what the winner added. An `InvalidChange` is not retried.
        """
        async with self._commit_lock:
            retries = self._settings.commit_retries
            attempt = 0
            while True:
                head = await self._store.get()
                ops = ops_for(head)
                if not ops:
                    return head
                change_set = ChangeSet(
                    base_version=head.version,
                    ops=ops,
                    actor=actor,
                    reason=reason,
                    channel=channel,
                    created_at=self._clock.now(),
                )
                try:
                    new = apply_changeset(head, change_set)
                    await self._store.commit(new, change_set)
                except VersionConflict:
                    attempt += 1
                    if attempt >= retries:
                        raise
                    continue
                return new

    def _stamp(self, change_set: ChangeSet) -> ChangeSet:
        if change_set.created_at is not None:
            return change_set
        return change_set.model_copy(update={"created_at": self._clock.now()})
