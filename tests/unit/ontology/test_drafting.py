"""Offline behavior of typed ontology proposal drafting."""

from collections.abc import Sequence
from typing import Any

import pytest
from pydantic_ai.exceptions import ModelHTTPError

from ontogr_mem.llm import ModelRole, RoleModels
from ontogr_mem.ontology.drafting import run_drafting
from ontogr_mem.ontology.errors import OntologyError
from ontogr_mem.ontology.models import (
    AddEntity,
    Example,
    Ontology,
    ProposalStatus,
    UnmatchedKind,
)
from ontogr_mem.ontology.proposals import Group
from tests.support.scripted_model import (
    Scripted,
    scripted_model,
    scripted_role_models,
)


def _group() -> Group:
    """Build one observed group for a proposed entity type."""
    return Group(
        kind=UnmatchedKind.ENTITY_TYPE,
        label="gadget",
        raw_labels=("Gadget",),
        support=3,
        examples=(Example(episode_id="episode-1", quote="a gadget"),),
    )


def _role_models(replies: Sequence[Any]) -> tuple[RoleModels, Scripted]:
    """Build an ontology-only model pool and return its request recorder."""
    model, recorder = scripted_model(replies)
    return scripted_role_models(**{ModelRole.ONTOLOGY.value: model}), recorder


class TestRunDrafting:
    """Draft outputs become proposals only after domain validation."""

    async def test_empty_groups_do_not_call_the_model(self) -> None:
        """Empty input returns an empty result without a model request."""
        models, recorder = _role_models([])

        result = await run_drafting(
            Ontology.genesis(), [], models, prompt_version="test"
        )

        assert result.proposals == []
        assert result.failures == []
        assert recorder.calls == []

    async def test_empty_ops_are_reported_as_an_empty_draft(self) -> None:
        """A typed draft with no operations is recorded as empty."""
        models, recorder = _role_models([{"ops": []}])

        result = await run_drafting(
            Ontology.genesis(), [_group()], models, prompt_version="test"
        )

        assert result.proposals == []
        assert result.failures == ["empty draft"]
        assert len(recorder.calls) == 1

    async def test_invalid_shape_is_reasked_then_valid_draft_is_returned(
        self,
    ) -> None:
        """PydanticAI re-asks for invalid shape before returning a valid draft."""
        models, recorder = _role_models(
            [
                {"ops": [{"op": "rename", "label": "Gadget"}]},
                {"ops": [{"op": "add_entity", "label": "Gadget"}]},
            ]
        )

        result = await run_drafting(
            Ontology.genesis(), [_group()], models, prompt_version="test"
        )

        assert len(result.proposals) == 1
        operation = result.proposals[0].ops[0]
        assert isinstance(operation, AddEntity) and operation.label == "Gadget"
        assert result.proposals[0].status is ProposalStatus.PENDING
        assert len(recorder.calls) == 2

    async def test_exhausted_shape_validation_returns_a_rejected_proposal(
        self,
    ) -> None:
        """Exhausted shape retries retain the validation reason on a rejection."""
        models, recorder = _role_models([{"ops": [{"op": "rename"}]}] * 3)

        result = await run_drafting(
            Ontology.genesis(), [_group()], models, prompt_version="test"
        )

        assert len(result.proposals) == 1
        proposal = result.proposals[0]
        assert proposal.status is ProposalStatus.REJECTED
        assert proposal.decided_by == "validator"
        assert proposal.reason
        assert len(recorder.calls) == 3

    async def test_domain_invalid_draft_is_rejected_without_a_reask(self) -> None:
        """A valid shape with an unknown endpoint fails only the dry-application."""
        models, recorder = _role_models(
            [
                {
                    "ops": [
                        {
                            "op": "add_entity",
                            "label": "Gadget",
                            "parent": "MissingParent",
                        }
                    ]
                }
            ]
        )

        result = await run_drafting(
            Ontology.genesis(), [_group()], models, prompt_version="test"
        )

        assert len(result.proposals) == 1
        proposal = result.proposals[0]
        assert proposal.status is ProposalStatus.REJECTED
        assert "MissingParent" in proposal.reason
        assert len(recorder.calls) == 1

    async def test_provider_failure_is_reported_as_an_ontology_error(
        self,
    ) -> None:
        """Provider failures keep the ontology-facing exception contract."""
        models, recorder = _role_models(
            [ModelHTTPError(503, "test-model", {"error": "unavailable"})]
        )

        with pytest.raises(OntologyError, match="draft failed"):
            await run_drafting(
                Ontology.genesis(), [_group()], models, prompt_version="test"
            )

        assert len(recorder.calls) == 1
