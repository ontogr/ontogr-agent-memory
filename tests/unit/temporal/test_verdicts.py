"""Tests for the fingerprint a stored verdict is checked against."""

from ontogr_mem.temporal.verdicts import FactFingerprint, inputs_hash


def fingerprint(statement: str, *, slot_key: str = "slot") -> FactFingerprint:
    """The fingerprint of a fact that starts in 2010 and rests on one statement."""
    return FactFingerprint(
        slot_key=slot_key,
        subject_id="ana",
        relation_type_id=1,
        object_id="acme",
        valid_from="2010-01-01T00:00:00+00:00",
        statement=statement,
    )


def test_the_inputs_hash_does_not_depend_on_the_order_of_the_pair():
    """A pair of facts hashes the same either way round."""
    left = fingerprint("ana works at acme")
    right = fingerprint("ana works at globex", slot_key="other")

    assert inputs_hash(left, right) == inputs_hash(right, left)


def test_the_inputs_hash_changes_when_a_statement_moves():
    """A verdict about one wording does not apply to another."""
    before = fingerprint("ana works at acme")
    after = fingerprint("ana is employed by acme")
    other = fingerprint("ana works at globex", slot_key="other")

    assert inputs_hash(before, other) != inputs_hash(before, after)


def test_the_inputs_hash_changes_when_the_slot_moves():
    """The slot is part of what a fact is, so it is part of the hash."""
    here = fingerprint("ana works at acme")
    there = fingerprint("ana works at acme", slot_key="other")
    other = fingerprint("ana works at globex", slot_key="other")

    assert inputs_hash(here, other) != inputs_hash(here, there)
