"""Build an escaped memory context block that fits a caller's token budget."""

from __future__ import annotations

import html
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime

from ontogr_mem.llm import estimate_tokens
from ontogr_mem.retrieval.models import (
    ContextBlock,
    EntityCard,
    EpisodeResult,
    FactResult,
    LaneStatus,
    RetrievalResult,
)
from ontogr_mem.retrieval.settings import RetrievalSettings
from ontogr_mem.temporal.render import ANSWER_RULE, render_fact_item


@dataclass(frozen=True, slots=True)
class _Item:
    """One budgetable rendered item and the identities and citations it carries."""

    kind: str
    ids: tuple[str, ...]
    text: str
    episode_ids: tuple[str, ...] = ()


def _degraded_names(result: RetrievalResult) -> tuple[str, ...]:
    """Return the failed lane names and failed searches in result order."""
    names: list[str] = []
    for lane in result.lanes:
        failed = (lane.name,) if lane.status is LaneStatus.FAILED else lane.failed
        for name in failed:
            if name not in names:
                names.append(name)
    return tuple(names)


def _fact_key(fact: FactResult) -> str:
    """Return the stable identity of one visible fact version."""
    return f"{fact.view.fact_id}@{fact.view.version}"


def _fact_items(
    facts: Sequence[FactResult], relation_label: Callable[[int], str] | None
) -> list[_Item]:
    """Build ranked fact items, keeping each connected dispute group atomic."""
    ordered = sorted(
        facts, key=lambda fact: (-fact.score, fact.view.fact_id, fact.view.version)
    )
    fact_ids = {fact.view.fact_id for fact in ordered}
    parents = {fact_id: fact_id for fact_id in fact_ids}

    def find(fact_id: str) -> str:
        while parents[fact_id] != fact_id:
            parents[fact_id] = parents[parents[fact_id]]
            fact_id = parents[fact_id]
        return fact_id

    for fact in ordered:
        if not fact.view.disputed:
            continue
        for counterpart_id in fact.view.counterpart_ids:
            if counterpart_id in parents:
                parents[find(counterpart_id)] = find(fact.view.fact_id)

    groups: dict[str, list[FactResult]] = {}
    for fact in ordered:
        groups.setdefault(find(fact.view.fact_id), []).append(fact)

    disputed_roots = {
        root
        for root, members in groups.items()
        if any(member.view.disputed for member in members)
    }
    emitted: set[str] = set()
    items: list[_Item] = []
    for fact in ordered:
        root = find(fact.view.fact_id)
        if root not in disputed_roots:
            items.append(
                _Item(
                    "fact",
                    (_fact_key(fact),),
                    render_fact_item(
                        fact.view,
                        episode_ids=fact.episode_ids,
                        relation_label=relation_label,
                    ),
                    fact.episode_ids,
                )
            )
            continue
        if root in emitted:
            continue
        emitted.add(root)
        members = groups[root]
        anchor = next(member for member in members if member.view.disputed)
        counterparts = [member.view for member in members if member is not anchor]
        items.append(
            _Item(
                "fact",
                tuple(_fact_key(member) for member in members),
                render_fact_item(
                    anchor.view,
                    counterparts,
                    episode_ids=anchor.episode_ids,
                    relation_label=relation_label,
                ),
                anchor.episode_ids,
            )
        )
    return items


def _entity_item(entity: EntityCard) -> _Item:
    """Render one escaped entity card."""
    aliases = (
        f" ({', '.join(html.escape(alias, quote=True) for alias in entity.aliases)})"
        if entity.aliases
        else ""
    )
    text = (
        f'<entity id="{html.escape(entity.entity_id, quote=True)}" '
        f'type="{html.escape(entity.type_label, quote=True)}">'
        f"{html.escape(entity.name, quote=True)}{aliases}: "
        f"{html.escape(entity.summary, quote=True)}</entity>"
    )
    return _Item("entity", (entity.entity_id,), text)


def _episode_item(episode: EpisodeResult, passage_chars: int) -> _Item:
    """Render one escaped episode with bounded text."""
    text = episode.text
    if len(text) > passage_chars:
        text = f"{text[:passage_chars]}…"
    rendered = (
        f'<episode id="{html.escape(episode.episode_id, quote=True)}" '
        f'occurred="{episode.occurred_at.isoformat()}">'
        f"{html.escape(text, quote=True)}</episode>"
    )
    return _Item("episode", (episode.episode_id,), rendered)


def _render_context(
    now: datetime,
    budget_tokens: int,
    degraded: Sequence[str],
    sections: dict[str, list[_Item]],
    dropped: int,
) -> str:
    """Render the frame and its non-empty sections."""
    lines = [
        f'<memory_context as_of="{now.isoformat()}" budget_tokens="{budget_tokens}">',
        ANSWER_RULE,
    ]
    if degraded:
        escaped = ", ".join(html.escape(name, quote=True) for name in degraded)
        lines.append(f"degraded: {escaped}")
    for name in ("facts", "entities", "episodes"):
        items = sections[name]
        if items:
            lines.extend((f"<{name}>", *(item.text for item in items), f"</{name}>"))
    if dropped:
        lines.append(f"Dropped: {dropped} items did not fit")
    lines.append("</memory_context>")
    return "\n".join(lines)


def assemble(
    result: RetrievalResult,
    now: datetime,
    budget_tokens: int,
    settings: RetrievalSettings,
    count_tokens: Callable[[str], int] = estimate_tokens,
    *,
    relation_label: Callable[[int], str] | None = None,
) -> ContextBlock:
    """Assemble ranked result parts into an escaped, budgeted context block.

    Args:
        result: Ranked facts, episodes, entity cards and lane records.
        now: Time shown in the context frame.
        budget_tokens: Maximum token count for the returned text.
        settings: Section shares and episode passage limit.
        count_tokens: Counter used for every budget decision and final count.
        relation_label: Returns a human-readable label for a relation type id.
            Omit it to show facts without a relation name.

    Returns:
        The context text, placed item identities, dropped item count and token
        usage. Fact and dispute groups remain whole.
    """
    fact_items = _fact_items(result.facts, relation_label)
    entity_items = [_entity_item(entity) for entity in result.entities]
    episode_items = [
        _episode_item(episode, settings.passage_chars) for episode in result.episodes
    ]
    all_item_count = len(fact_items) + len(entity_items) + len(episode_items)
    degraded = _degraded_names(result)
    empty_sections = {name: [] for name in ("facts", "episodes", "entities")}
    reserved_frame = _render_context(
        now, budget_tokens, degraded, empty_sections, all_item_count
    )
    frame_tokens = count_tokens(reserved_frame)
    if frame_tokens > budget_tokens:
        return ContextBlock("", (), all_item_count, 0, budget_tokens, degraded)

    candidates = {
        "facts": fact_items,
        "episodes": episode_items,
        "entities": entity_items,
    }
    placed = _place_items(
        candidates,
        now=now,
        budget_tokens=budget_tokens,
        degraded=degraded,
        all_item_count=all_item_count,
        frame_tokens=frame_tokens,
        settings=settings,
        count_tokens=count_tokens,
    )
    return _finish_context(
        placed,
        episode_items,
        now=now,
        budget_tokens=budget_tokens,
        degraded=degraded,
        all_item_count=all_item_count,
        count_tokens=count_tokens,
    )


def _place_items(
    candidates: dict[str, list[_Item]],
    *,
    now: datetime,
    budget_tokens: int,
    degraded: tuple[str, ...],
    all_item_count: int,
    frame_tokens: int,
    settings: RetrievalSettings,
    count_tokens: Callable[[str], int],
) -> dict[str, list[_Item]]:
    """Admit whole items by section share, carrying unused tokens forward."""
    remaining = budget_tokens - frame_tokens
    shares = {
        "facts": settings.facts_share,
        "episodes": settings.episodes_share,
        "entities": settings.entities_share,
    }
    placed = {name: [] for name in ("facts", "episodes", "entities")}
    current_tokens = frame_tokens
    carry = 0.0

    for section in ("facts", "episodes", "entities"):
        allowance = remaining * shares[section] + carry
        spent = 0
        for item in candidates[section]:
            if section == "episodes":
                cited = {
                    episode_id
                    for fact_item in placed["facts"]
                    for episode_id in fact_item.episode_ids
                }
                if item.ids[0] in cited:
                    continue

            proposed = [*placed[section], item]
            trial_sections = {name: list(items) for name, items in placed.items()}
            trial_sections[section] = proposed
            trial_text = _render_context(
                now,
                budget_tokens,
                degraded,
                trial_sections,
                all_item_count,
            )
            trial_tokens = count_tokens(trial_text)
            cost = max(0, trial_tokens - current_tokens)
            if trial_tokens > budget_tokens or spent + cost > allowance:
                continue
            placed[section].append(item)
            current_tokens = trial_tokens
            spent += cost
        carry = max(0.0, allowance - spent)
    return placed


def _final_text(
    placed: dict[str, list[_Item]],
    episode_items: Sequence[_Item],
    *,
    now: datetime,
    budget_tokens: int,
    degraded: tuple[str, ...],
    all_item_count: int,
) -> tuple[str, int]:
    """Render the admitted items without counting cited episodes as dropped."""
    cited = {
        episode_id
        for fact_item in placed["facts"]
        for episode_id in fact_item.episode_ids
    }
    skipped = sum(item.ids[0] in cited for item in episode_items)
    placed_count = sum(len(items) for items in placed.values())
    dropped = all_item_count - skipped - placed_count
    text = _render_context(now, budget_tokens, degraded, placed, dropped)
    return text, dropped


def _finish_context(
    placed: dict[str, list[_Item]],
    episode_items: Sequence[_Item],
    *,
    now: datetime,
    budget_tokens: int,
    degraded: tuple[str, ...],
    all_item_count: int,
    count_tokens: Callable[[str], int],
) -> ContextBlock:
    """Drop admitted items from the end until the text fits, then report it."""
    text, dropped = _final_text(
        placed,
        episode_items,
        now=now,
        budget_tokens=budget_tokens,
        degraded=degraded,
        all_item_count=all_item_count,
    )
    while text and count_tokens(text) > budget_tokens:
        for section in ("entities", "episodes", "facts"):
            if placed[section]:
                placed[section].pop()
                break
        else:
            return ContextBlock("", (), all_item_count, 0, budget_tokens, degraded)
        text, dropped = _final_text(
            placed,
            episode_items,
            now=now,
            budget_tokens=budget_tokens,
            degraded=degraded,
            all_item_count=all_item_count,
        )

    if not text:
        return ContextBlock("", (), all_item_count, 0, budget_tokens, degraded)
    items = tuple(
        (item.kind, item_id)
        for section in ("facts", "entities", "episodes")
        for item in placed[section]
        for item_id in item.ids
    )
    return ContextBlock(
        text,
        items,
        dropped,
        count_tokens(text),
        budget_tokens,
        degraded,
    )
