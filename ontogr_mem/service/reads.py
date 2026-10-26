"""Read memory through the tenant-scoped retrieval and storage ports."""

from __future__ import annotations

import base64
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from ontogr_mem.data_models import Episode, JobState, require_utc
from ontogr_mem.llm import estimate_tokens
from ontogr_mem.resolution.keys import strict_key
from ontogr_mem.retrieval.errors import RetrievalError
from ontogr_mem.retrieval.models import FactResult, RetrievalRequest, RetrievalResult
from ontogr_mem.service.contracts import (
    EntityView,
    EpisodeOut,
    EpisodePage,
    FactOut,
    GetEntity,
    ListEpisodes,
    MemoryContext,
    SearchMemory,
)
from ontogr_mem.service.errors import (
    InternalError,
    InvalidArgumentError,
    ObjectNotFoundError,
)
from ontogr_mem.service.formatting import (
    entity_card_out,
    entity_out,
    fact_out,
    retrieved_fact_out,
)
from ontogr_mem.service.runtime import TenantRuntime
from ontogr_mem.temporal.query import FactView, TemporalQuery
from ontogr_mem.temporal.render import render_context


if TYPE_CHECKING:
    from ontogr_mem.relational_store.jobs.ports import JobStore


def encode_episode_cursor(position: tuple[datetime, str]) -> str:
    """Encode the keyset position of an episode page."""
    timestamp, episode_id = position
    value = f"{require_utc(timestamp).isoformat()}|{episode_id}".encode("utf-8")
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def decode_episode_cursor(cursor: str | None) -> tuple[datetime, str] | None:
    """Decode a page cursor, rejecting invalid timestamps and missing ids."""
    if not cursor:
        return None
    try:
        padded = cursor + "=" * (-len(cursor) % 4)
        value = base64.b64decode(padded, altchars=b"-_", validate=True).decode("utf-8")
        timestamp, episode_id = value.split("|", 1)
        if not episode_id:
            raise ValueError
        return require_utc(datetime.fromisoformat(timestamp)), episode_id
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError("invalid episode cursor") from exc


async def _retrieve_tenants(
    tenants: Sequence[tuple[str, TenantRuntime]],
    request: SearchMemory,
    now: datetime,
) -> list[RetrievalResult]:
    """Retrieve each tenant with an equal share of the context budget."""
    count = len(tenants)
    if count == 0:
        raise InvalidArgumentError("search needs at least one tenant")
    try:
        retrieval_request = RetrievalRequest(
            query=request.query,
            now=now,
            limit=request.limit,
            budget_tokens=request.budget_tokens // count,
            valid_at=request.valid_at,
            overlaps=(request.overlaps_start, request.overlaps_end)
            if request.overlaps_start is not None and request.overlaps_end is not None
            else None,
            known_at=request.known_at,
            include_history=request.include_history,
            disputed=request.disputed,
            mentions=tuple(request.mentions),
            graph_depth=request.graph_depth,
        )
    except RetrievalError as exc:
        raise InvalidArgumentError(str(exc)) from exc
    results = []
    for _, runtime in tenants:
        try:
            results.append(await runtime.retriever.retrieve(retrieval_request))
        except RetrievalError as exc:
            raise InternalError("retrieval failed") from exc
    return results


@dataclass(frozen=True, slots=True)
class _SearchAssembly:
    """Formatted tenant facts and context before the response format is applied."""

    facts: list[FactOut]
    contexts: list[str]
    tokens_used: int
    degraded: set[str]


async def _assemble_search(
    tenants: Sequence[tuple[str, TenantRuntime]],
    results: Sequence[RetrievalResult],
    request: SearchMemory,
) -> _SearchAssembly:
    """Filter sessions, format tenant results, and interleave their ranked facts."""
    allowed_episodes: list[set[str] | None] = []
    for (_, runtime), _ in zip(tenants, results, strict=True):
        if request.session_id is not None:
            allowed_episodes.append(
                await _session_episodes(runtime, request.session_id)
            )
        else:
            allowed_episodes.append(None)

    ranked: list[list[FactResult]] = []
    outputs = []
    contexts: list[str] = []
    degraded: set[str] = set()
    tokens_used = 0
    for (tenant_id, runtime), result, episode_filter in zip(
        tenants, results, allowed_episodes, strict=True
    ):
        facts = [
            item
            for item in result.facts
            if episode_filter is None or episode_filter.intersection(item.episode_ids)
        ]
        ranked.append(facts)
        ids = {
            entity_id
            for item in facts
            for entity_id in (item.view.subject_id, item.view.object_id)
        }
        entities = await runtime.stores.resolver_store.get_entities(sorted(ids))
        names = {entity.entity_id: entity.canonical_name for entity in entities}
        outputs.append(
            [
                retrieved_fact_out(
                    tenant_id,
                    item,
                    names,
                    runtime.relation_label,
                )
                for item in facts
            ]
        )
        if episode_filter is None:
            contexts.append(result.context.text)
            tokens_used += result.context.tokens_used
        else:
            rendered = render_context(
                [item.view for item in facts], runtime.relation_label
            )
            budget = request.budget_tokens // len(tenants)
            contexts.append(_within_budget(rendered, budget))
            tokens_used += min(estimate_tokens(contexts[-1]), budget)
        degraded.update(result.context.degraded)

    merged = _round_robin(ranked, request.limit)
    facts = [outputs[tenant_index][position] for tenant_index, position in merged]
    return _SearchAssembly(facts, contexts, tokens_used, degraded)


async def search(
    tenants: Sequence[tuple[str, TenantRuntime]],
    request: SearchMemory,
    now: datetime,
    jobs: JobStore,
) -> MemoryContext:
    """Search tenants, merge ranked facts fairly, and assemble context."""
    results = await _retrieve_tenants(tenants, request, now)
    assembly = await _assemble_search(tenants, results, request)
    facts = assembly.facts
    degraded = assembly.degraded
    entities_out = []
    seen_entities: set[tuple[str, str]] = set()
    for (tenant_id, _), result in zip(tenants, results, strict=True):
        for card in result.entities:
            key = (tenant_id, card.entity_id)
            if key not in seen_entities:
                seen_entities.add(key)
                entities_out.append(entity_card_out(tenant_id, card))

    found = bool(facts or entities_out)
    state = "found" if found else "none"
    if not found and degraded:
        state = "degraded"
    if not found and not degraded:
        pending = await jobs.count_jobs(
            [tenant_id for tenant_id, _ in tenants],
            [JobState.QUEUED, JobState.RUNNING],
        )
        if pending:
            state = "indexing"
    rendered = "\n\n".join(context for context in assembly.contexts if context) or None
    if request.format == "json":
        rendered = None
    if request.format == "prompt":
        facts = []
    return MemoryContext(
        state=state,
        facts=facts,
        entities=entities_out,
        rendered=rendered,
        tokens_used=assembly.tokens_used,
        degraded=sorted(degraded),
    )


async def get_entity(
    tenant_id: str, runtime: TenantRuntime, request: GetEntity, now: datetime
) -> EntityView:
    """Find an entity by id or strict name and return its latest fact versions."""
    if request.entity_id is not None:
        entities = await runtime.resolver.get_entities(
            [request.entity_id], follow_merged=True
        )
    elif request.name is not None:
        entities = await runtime.stores.resolver_store.entities_by_strict_key(
            strict_key(request.name)
        )
        if entities:
            entities = await runtime.resolver.get_entities(
                [entities[0].entity_id], follow_merged=True
            )
    else:
        raise InvalidArgumentError("name exactly one of entity_id and name")
    if not entities:
        raise ObjectNotFoundError("entity not found")
    entity = entities[0]
    aliases = await runtime.stores.resolver_store.aliases_of(entity.entity_id)
    views = await runtime.reader.touching(
        [entity.entity_id],
        TemporalQuery(now=now, include_history=request.include_history),
    )
    latest: dict[str, FactView] = {}
    for view in views:
        if view.fact_id not in latest or view.version > latest[view.fact_id].version:
            latest[view.fact_id] = view
    ordered = sorted(
        latest.values(), key=lambda view: (view.valid_from, view.fact_id), reverse=True
    )[: request.max_facts]
    entity_names = await runtime.stores.resolver_store.get_entities(
        sorted(
            {
                entity_id
                for view in ordered
                for entity_id in (view.subject_id, view.object_id)
            }
        )
    )
    names = {item.entity_id: item.canonical_name for item in entity_names}
    facts: list[FactOut] = []
    for view in ordered:
        evidence = await runtime.reader.observation_texts(view.evidence)
        facts.append(
            fact_out(
                tenant_id,
                view,
                names,
                runtime.relation_label,
                episode_ids=tuple(
                    dict.fromkeys(text.episode_id for text in evidence.values())
                ),
            )
        )
    return EntityView(
        entity=entity_out(tenant_id, entity, aliases),
        facts=facts,
    )


async def list_episodes(
    tenant_id: str, runtime: TenantRuntime, request: ListEpisodes
) -> EpisodePage:
    """Return one newest-first episode page and its next cursor."""
    try:
        before = decode_episode_cursor(request.cursor)
    except ValueError as exc:
        raise InvalidArgumentError(str(exc)) from exc
    episodes = await runtime.stores.episodes.list_episodes(
        limit=request.limit + 1,
        before=before,
        session_id=request.session_id,
    )
    page = episodes[: request.limit]
    cursor = (
        encode_episode_cursor((page[-1].ingested_at, page[-1].id))
        if len(episodes) > request.limit and page
        else None
    )
    return EpisodePage(
        episodes=[await _episode_out(tenant_id, runtime, episode) for episode in page],
        next_cursor=cursor,
    )


async def _episode_out(
    tenant_id: str, runtime: TenantRuntime, episode: Episode
) -> EpisodeOut:
    """Read session metadata and format one episode."""
    meta = await runtime.stores.episodes.get_episode_meta(episode.id)
    return EpisodeOut(
        episode_id=episode.id,
        tenant=tenant_id,
        kind=episode.kind.value,
        session_id=meta.session_id if meta else None,
        thread=meta.thread if meta else None,
        source_description=episode.source.description,
        job_id=episode.source.ref,
        created_at=episode.ingested_at,
    )


async def _session_episodes(runtime: TenantRuntime, session_id: str) -> set[str]:
    """Read every matching episode id with bounded keyset pages."""
    ids: set[str] = set()
    before = None
    while True:
        page = await runtime.stores.episodes.list_episodes(
            limit=100, before=before, session_id=session_id
        )
        ids.update(episode.id for episode in page)
        if len(page) < 100:
            return ids
        last = page[-1]
        before = (last.ingested_at, last.id)


def _round_robin(ranked: list[list[FactResult]], limit: int) -> list[tuple[int, int]]:
    """Interleave tenant ranks without comparing scores across tenants."""
    out = []
    index = 0
    while len(out) < limit:
        added = False
        for tenant_index, items in enumerate(ranked):
            if index < len(items):
                out.append((tenant_index, index))
                added = True
                if len(out) == limit:
                    break
        if not added:
            break
        index += 1
    return out


def _within_budget(text: str, budget: int) -> str:
    """Keep complete rendered fact lines within a tenant's token share."""
    lines = []
    for line in text.splitlines():
        candidate = "\n".join([*lines, line])
        if estimate_tokens(candidate) > budget:
            break
        lines.append(line)
    return "\n".join(lines)
