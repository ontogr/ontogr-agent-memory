"""Draft proposals with a typed model agent."""

from dataclasses import dataclass, field

from ontogr_mem.llm import (
    ModelFailure,
    ModelOutputError,
    ModelRole,
    RoleModels,
    run_agent,
)
from ontogr_mem.ontology.apply import apply_changeset
from ontogr_mem.ontology.drafting_agent import drafting_agent
from ontogr_mem.ontology.errors import InvalidChange, OntologyError
from ontogr_mem.ontology.models import (
    ChangeSet,
    Channel,
    Ontology,
    Proposal,
    ProposalSource,
    ProposalStatus,
)
from ontogr_mem.ontology.proposals import Group, build_prompt, proposal_id


@dataclass(slots=True)
class DraftRun:
    """Result of one drafting pass: stored proposals and recorded failures."""

    proposals: list[Proposal] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)


async def run_drafting(
    cat: Ontology,
    groups: list[Group],
    models: RoleModels,
    *,
    prompt_version: str,
    attempt: int = 0,
) -> DraftRun:
    """Draft additions, then check their domain rules without a repair call.

    Args:
        cat: The current catalog.
        groups: The label groups to draft for.
        models: Configured models for each kind of work.
        prompt_version: Identifies the prompt and model setup in the actor name.
        attempt: Earlier proposals for these labels, used in the proposal id.

    Returns:
        The pending, or rejected, proposals and the recorded failures.

    Raises:
        OntologyError: If the model call fails.
    """
    run = DraftRun()
    if not groups:
        return run
    labels = sorted(g.label for g in groups)
    identifier = proposal_id(labels, cat.version, attempt)
    support = sum(group.support for group in groups)
    failure = ""
    try:
        result = await run_agent(
            drafting_agent,
            ModelRole.ONTOLOGY,
            build_prompt(cat, groups),
            models=models,
            deps=None,
        )
    except ModelOutputError as exc:
        failure = str(exc)[:500]
    except ModelFailure as exc:
        raise OntologyError(f"draft failed: {exc}") from exc
    else:
        draft = result.output
        if not draft.ops:
            run.failures.append("empty draft")
            return run
        try:
            apply_changeset(
                cat,
                ChangeSet(
                    base_version=cat.version,
                    ops=list(draft.ops),
                    actor="drafter:" + prompt_version,
                    channel=Channel.PROPOSAL,
                ),
            )
        except InvalidChange as exc:
            failure = str(exc)[:500]
        else:
            run.proposals.append(
                Proposal(
                    proposal_id=identifier,
                    labels=labels,
                    support=support,
                    base_version=cat.version,
                    ops=list(draft.ops),
                    examples=[
                        example for group in groups for example in group.examples
                    ][:5],
                    source=ProposalSource.OBSERVED,
                )
            )
            return run
    run.proposals.append(
        Proposal(
            proposal_id=identifier,
            labels=labels,
            support=support,
            base_version=cat.version,
            ops=[],
            status=ProposalStatus.REJECTED,
            decided_by="validator",
            reason=failure or "no valid draft",
        )
    )
    return run
