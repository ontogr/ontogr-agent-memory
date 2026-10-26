"""Draft proposals with the shared language model client."""

from dataclasses import dataclass, field

from pydantic import ValidationError

from ontogr_mem.llm import LLMClient, LLMError, LLMRequest, ModelSize
from ontogr_mem.ontology.apply import apply_changeset
from ontogr_mem.ontology.errors import InvalidChange, OntologyError
from ontogr_mem.ontology.models import (
    ChangeSet,
    Channel,
    Ontology,
    Proposal,
    ProposalSource,
    ProposalStatus,
)
from ontogr_mem.ontology.proposals import Draft, Group, build_prompt, proposal_id


SYSTEM_PROMPT = "You extend an ontology. Answer with one JSON object and nothing else."


@dataclass(slots=True)
class DraftRun:
    """Result of one drafting pass: stored proposals and recorded failures."""

    proposals: list[Proposal] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)


async def run_drafting(
    cat: Ontology,
    groups: list[Group],
    llm: LLMClient,
    *,
    prompt_version: str,
    max_retries: int = 2,
    attempt: int = 0,
) -> DraftRun:
    """Draft one batch, check it by applying it to a copy, and retry on errors.

    The answer must parse as a `Draft` and apply cleanly to the catalog as a
    proposal-channel change set. On a parse or rule error the error text goes
    back in the next prompt, up to `max_retries` more times. An empty draft is
    recorded in `failures`. A draft that never validates is returned as a
    rejected proposal with the last error.

    Args:
        cat: The current catalog.
        groups: The label groups to draft for.
        llm: The model client.
        prompt_version: Identifies the prompt and model setup in the actor name.
        max_retries: Extra attempts after the first.
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
    pid = proposal_id(labels, cat.version, attempt)
    schema = Draft.model_json_schema()
    feedback = ""
    for _ in range(max_retries + 1):
        request = LLMRequest(
            system=SYSTEM_PROMPT,
            prompt=build_prompt(cat, groups, feedback),
            model_size=ModelSize.MEDIUM,
            schema=schema,
        )
        try:
            response = await llm.complete(request)
        except LLMError as exc:
            msg = f"draft failed: {exc}"
            raise OntologyError(msg) from exc
        try:
            draft = Draft.model_validate_json(response.text)
            if not draft.ops:
                run.failures.append("empty draft")
                return run
            apply_changeset(
                cat,
                ChangeSet(
                    base_version=cat.version,
                    ops=list(draft.ops),
                    actor="drafter:" + prompt_version,
                    channel=Channel.PROPOSAL,
                ),
            )
        except (ValidationError, InvalidChange) as exc:
            feedback = str(exc)[:500]
            continue
        run.proposals.append(
            Proposal(
                proposal_id=pid,
                ops=list(draft.ops),
                labels=labels,
                support=sum(g.support for g in groups),
                examples=[e for g in groups for e in g.examples][:5],
                source=ProposalSource.OBSERVED,
                base_version=cat.version,
            )
        )
        return run
    run.proposals.append(
        Proposal(
            proposal_id=pid,
            ops=[],
            labels=labels,
            support=sum(g.support for g in groups),
            status=ProposalStatus.REJECTED,
            decided_by="validator",
            reason=feedback or "no valid draft",
            base_version=cat.version,
        )
    )
    return run
