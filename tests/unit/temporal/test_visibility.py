"""Tests for the two rules about a fact version: the reader's and the writer's."""

from dataclasses import replace

from ontogr_mem.data_models.fact_versions import FactStatus
from ontogr_mem.temporal.visibility import in_current_view, is_withdrawable
from tests.unit.temporal.helpers import at, fact_of, obs


def test_a_corrected_fact_is_out_of_the_view_but_the_writer_still_holds_it():
    """A correction hides the fact from readers; the writer can still retire it."""
    fact = fact_of("f1", [obs("o1", "acme", at(2010))], status=FactStatus.CORRECTED)

    assert in_current_view(fact.status) is False
    assert is_withdrawable(fact) is True


def test_an_overridden_fact_stays_in_the_view():
    """A fact that lost a preference is shown, with its status."""
    fact = fact_of("f1", [obs("o1", "acme", at(2010))], status=FactStatus.OVERRIDDEN)

    assert in_current_view(fact.status) is True
    assert is_withdrawable(fact) is True


def test_a_withdrawn_fact_is_out_of_the_view_and_spent():
    """A tombstone is neither shown nor retirable."""
    fact = fact_of("f1", [obs("o1", "acme", at(2010))], status=FactStatus.WITHDRAWN)

    assert in_current_view(fact.status) is False
    assert is_withdrawable(fact) is False


def test_a_replaced_version_is_spent():
    """A version a later version replaced is no longer the writer's to retire."""
    fact = fact_of("f1", [obs("o1", "acme", at(2010))])
    expired = replace(fact, expired_at=at(2011))

    assert is_withdrawable(expired) is False
