"""Selector validation for hard forget targets."""

import pytest

from ontogr_mem.service.errors import InvalidArgumentError
from ontogr_mem.service.forget.selector import ForgetSelector, validate_selector


class TestForgetSelector:
    """Selectors name exactly one non-blank target."""

    @pytest.mark.parametrize(
        "selector",
        [
            ForgetSelector(),
            ForgetSelector(episode_ids=("e1",), fact_id="f1"),
            ForgetSelector(fact_id="  "),
            ForgetSelector(subject_ref="\t"),
            ForgetSelector(source_system=""),
            ForgetSelector(episode_ids=()),
            ForgetSelector(episode_ids=("e1", " ")),
        ],
    )
    def test_rejects_missing_ambiguous_or_blank_target(self, selector):
        """A missing, ambiguous, or blank selector cannot be planned."""
        with pytest.raises(InvalidArgumentError):
            validate_selector(selector)

    @pytest.mark.parametrize(
        ("selector", "kind", "value"),
        [
            (ForgetSelector(episode_ids=("e1",)), "episode_ids", ("e1",)),
            (ForgetSelector(fact_id="f1"), "fact_id", "f1"),
            (ForgetSelector(subject_ref="ana"), "subject_ref", "ana"),
            (ForgetSelector(source_system="mail"), "source_system", "mail"),
        ],
    )
    def test_accepts_each_target_kind(self, selector, kind, value):
        """Each supported selector keeps its kind and exact value."""
        validated = validate_selector(selector)

        assert validated.kind == kind
        assert validated.value == value
