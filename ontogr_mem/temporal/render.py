"""Facts as text for an answering model, with disputes spelled out.

A pure function over ``FactView`` rows, so it can be tested without a model. A
fact in an open dispute is rendered with the facts it conflicts with and the
reason. The rule in ``ANSWER_RULE`` tells the model not to pick one side silently.
"""

import html
from collections.abc import Callable, Sequence
from datetime import datetime

from ontogr_mem.data_models.facts import Precision
from ontogr_mem.temporal.query import FactView


ANSWER_RULE = (
    "When facts are marked disputed, state both with their sources and dates; "
    "do not pick one silently."
)


def _stamp(at: datetime, precision: Precision | None) -> str:
    """Show a date no more precisely than it was stated: "2020", not a full date."""
    if precision is Precision.YEAR:
        return f"{at.year:04d}"
    if precision is Precision.MONTH:
        return f"{at.year:04d}-{at.month:02d}"
    return at.date().isoformat()


def _valid(view: FactView) -> str:
    """Return the valid interval as text, e.g. ``2019 to 2022``."""
    start = _stamp(view.valid_from, view.valid_from_precision)
    end = _stamp(view.valid_to, view.valid_to_precision) if view.valid_to else "open"
    return f"{start} to {end}"


def _text(view: FactView, relation_label: Callable[[int], str] | None) -> str:
    """Return the statement or a readable fallback triple."""
    if view.statement:
        return view.statement
    relation = relation_label(view.relation_type_id) if relation_label else ""
    return f"{view.subject_id} {relation or view.relation_type_id} {view.object_id}"


def _tag(
    view: FactView,
    episode_ids: Sequence[str],
    relation_label: Callable[[int], str] | None,
) -> str:
    """Return one ``<fact>`` element for the answering model.

    Stored text is HTML-escaped, so it cannot close the tag or inject markup.
    """
    status = "disputed" if view.disputed else view.status.value
    episodes = (
        f' episodes="{html.escape(" ".join(episode_ids), quote=True)}"'
        if episode_ids
        else ""
    )
    return (
        f'<fact id="{html.escape(view.fact_id, quote=True)}" valid="{_valid(view)}" '
        f'source="{html.escape(view.source or "unknown", quote=True)}" '
        f'status="{status}"{episodes}>'
        f"{html.escape(' '.join(_text(view, relation_label).split()))}</fact>"
    )


def render_fact_item(
    view: FactView,
    counterparts: Sequence[FactView] = (),
    *,
    episode_ids: Sequence[str] = (),
    relation_label: Callable[[int], str] | None = None,
) -> str:
    """Render one fact or its dispute group as escaped XML-like elements.

    Args:
        view: Fact version to render first.
        counterparts: Dispute counterparts included in this item.
        episode_ids: Supporting episode ids shown on the first fact element.
        relation_label: Returns a human-readable label for a relation id.

    Returns:
        One fact element, or one dispute line with its reason.
    """
    if not view.disputed:
        return _tag(view, episode_ids, relation_label)

    visible_ids = {counterpart.fact_id for counterpart in counterparts}
    unseen = [fact_id for fact_id in view.counterpart_ids if fact_id not in visible_ids]
    text = " conflicts with ".join(
        [
            _tag(view, episode_ids, relation_label),
            *(_tag(counterpart, (), relation_label) for counterpart in counterparts),
        ]
    )
    if unseen:
        text += f" (also disputed with {', '.join(html.escape(i) for i in unseen)})"
    return f"{text}; reason: {html.escape(view.dispute_reason)}"


def render_context(
    views: Sequence[FactView],
    relation_label: Callable[[int], str] | None = None,
) -> str:
    """Return one line per fact, or one line per group of facts in one dispute."""
    by_id = {v.fact_id: v for v in views}
    shown: set[str] = set()
    lines: list[str] = []
    for view in sorted(views, key=lambda v: (v.valid_from, v.fact_id)):
        if view.fact_id in shown:
            continue
        if not view.disputed:
            shown.add(view.fact_id)
            lines.append(render_fact_item(view, relation_label=relation_label))
            continue
        counterparts = [by_id[c] for c in view.counterpart_ids if c in by_id]
        shown.add(view.fact_id)
        shown.update(counterpart.fact_id for counterpart in counterparts)
        lines.append(
            render_fact_item(view, counterparts, relation_label=relation_label)
        )
    return "\n".join(lines)
