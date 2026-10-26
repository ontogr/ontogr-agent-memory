"""Tests for ``derive``: timeline derivation with no store or engine.

They cover replacement, repeat, stated end, recurrence, ties, undated
statements and arrival-order independence.
"""

import itertools
import random
from collections.abc import Sequence
from datetime import datetime

import pytest

from ontogr_mem.data_models.facts import Precision
from ontogr_mem.data_models.observation import Observation
from ontogr_mem.temporal.cardinality import Cardinality
from ontogr_mem.temporal.reconcile import ConflictKind, Derived, derive
from tests.unit.temporal.helpers import at, obs, stated_end


def timeline(derived: Derived) -> list[tuple[str, datetime, datetime | None]]:
    """Reduce a result to sorted value and interval tuples."""
    return sorted(
        ((run.value_id, run.valid_from, run.valid_to) for run in derived.runs),
        key=lambda item: (item[1], item[0]),
    )


def run(
    observations: Sequence[Observation],
    cardinality: Cardinality = Cardinality.ONE,
) -> Derived:
    """Derive one slot with the given cardinality."""
    return derive("slot", observations, cardinality)


A10 = obs("a10", "A", at(2010))
B15 = obs("b15", "B", at(2015))
A18 = obs("a18", "A", at(2018))


class TestDerive:
    """Behavior of the pure timeline walk."""

    def test_note_scenario_is_order_independent(self):
        """Every arrival order gives A, B, then A again."""
        expected = [
            ("A", at(2010), at(2015)),
            ("B", at(2015), at(2018)),
            ("A", at(2018), None),
        ]
        for order in itertools.permutations([A10, B15, A18]):
            assert timeline(run(list(order))) == expected

    def test_late_older_fact_is_bounded_by_the_next_newer_one(self):
        """A late older fact is cut by the newer fact and cannot close it."""
        derived = run([B15, obs("a10", "A", at(2010))])
        assert timeline(derived) == [
            ("A", at(2010), at(2015)),
            ("B", at(2015), None),
        ]

    def test_repeat_of_active_value_adds_evidence_only(self):
        """Restating the active value adds evidence, not a new interval."""
        derived = run([A10, obs("a12", "A", at(2012))])
        assert len(derived.runs) == 1
        assert derived.runs[0].obs_ids == ("a10", "a12")
        assert derived.runs[0].valid_to is None

    def test_replacement_records_cause(self):
        """A cut records its kind and the observation that caused it."""
        derived = run([A10, B15])
        first = next(run for run in derived.runs if run.value_id == "A")
        assert first.close_kind is not None
        assert first.close_kind.value == "replacement"
        assert first.closed_by == "b15"

    def test_many_slot_never_closes_without_a_stated_end(self):
        """In a many-valued slot, values coexist until an explicit end."""
        derived = run(
            [A10, obs("a12", "A", at(2012))],
            Cardinality.MANY,
        )
        assert [(run.valid_from, run.valid_to) for run in derived.runs] == [
            (at(2010), None)
        ]
        derived = run(
            [obs("x", "A", at(2010)), stated_end("e", "A", at(2012))],
            Cardinality.MANY,
        )
        assert derived.runs[0].valid_to == at(2012)
        assert derived.runs[0].close_kind is not None
        assert derived.runs[0].close_kind.value == "stated_end"

    def test_stated_end_only_closes_its_own_value(self):
        """An end for one value leaves other values open."""
        derived = run([A10, B15, stated_end("e", "A", at(2020))])
        assert [run.valid_to for run in derived.runs if run.value_id == "B"] == [None]
        assert derived.unmatched_ends == ("e",)

    def test_stated_end_matches_once_the_earlier_assertion_arrives(self):
        """An end is unmatched alone and matched once its assertion is known."""
        end = stated_end("e", "A", at(2012))
        assert run([end]).unmatched_ends == ("e",)
        assert run([end, A10]).unmatched_ends == ()

    def test_assertion_valid_to_closes_and_a_later_repeat_reopens(self):
        """A recurrence opens a new run after the stated end."""
        first = obs("a10", "A", at(2010), end=at(2012), end_prec=Precision.YEAR)
        derived = run([first, obs("a15", "A", at(2015))])
        assert timeline(derived) == [
            ("A", at(2010), at(2012)),
            ("A", at(2015), None),
        ]

    def test_later_open_repeat_overrides_an_earlier_stated_end(self):
        """A repeat before the end passes keeps the run open."""
        first = obs("a10", "A", at(2010), end=at(2020), end_prec=Precision.YEAR)
        derived = run([first, obs("a15", "A", at(2015))])
        assert timeline(derived) == [("A", at(2010), None)]

    def test_equal_start_ties_neither_cut_and_open_one_conflict(self):
        """Two values starting in one period are reported, not ordered."""
        derived = run(
            [
                obs("p", "Paris", at(2023, 3), prec=Precision.MONTH),
                obs("b", "Berlin", at(2023, 3), prec=Precision.MONTH),
            ]
        )
        assert all(run.valid_to is None for run in derived.runs)
        assert [conflict.kind for conflict in derived.conflicts] == [
            ConflictKind.EQUAL_START
        ]
        assert derived.conflicts[0].obs_ids == ("b", "p")

    def test_mixed_precision_ties_form_one_component(self):
        """Ties chain across precisions, and a later fact closes all members."""
        year = obs("x", "X", at(2023), prec=Precision.YEAR)
        march = obs("y", "Y", at(2023, 3), prec=Precision.MONTH)
        june = obs("z", "Z", at(2023, 6), prec=Precision.MONTH)
        later = obs("w", "W", at(2024))
        for order in itertools.permutations([year, march, june, later]):
            derived = run(list(order))
            assert len(derived.conflicts) == 1
            assert derived.conflicts[0].obs_ids == ("x", "y", "z")
            ends = {run.value_id: run.valid_to for run in derived.runs}
            assert ends == {"X": at(2024), "Y": at(2024), "Z": at(2024), "W": None}

    def test_undated_repeat_is_evidence_and_other_value_conflicts(self):
        """An undated value either supports the open run or is reported."""
        derived = run([A10, obs("u", "A", None)])
        assert derived.runs[0].obs_ids == ("a10", "u")
        derived = run([A10, obs("u", "B", None)])
        assert [conflict.kind for conflict in derived.conflicts] == [
            ConflictKind.NO_DATES
        ]
        assert len(derived.runs) == 1 and derived.runs[0].valid_to is None

    def test_undated_with_nothing_to_attach_to_is_an_orphan(self):
        """An undated statement alone is reported, not dropped."""
        derived = run([obs("u", "A", None)])
        assert derived.runs == () and derived.orphans == ("u",)

    def test_undated_never_orders_or_cuts(self):
        """An undated statement does not change the dated timeline."""
        derived = run([A10, obs("u", "B", None), A18])
        assert timeline(derived) == [("A", at(2010), None)]

    @pytest.mark.parametrize("seed", range(40))
    def test_permutation_invariance_on_random_sets(self, seed):
        """Forty seeded sets survive twelve shuffles unchanged."""
        rng = random.Random(seed)
        precisions = [Precision.YEAR, Precision.MONTH, Precision.DAY]
        items = []
        for index in range(rng.randint(3, 7)):
            precision = rng.choice(precisions)
            when = at(2020 + rng.randint(0, 3), rng.randint(1, 12), rng.randint(1, 28))
            value = rng.choice("ABC")
            if rng.random() < 0.15:
                items.append(stated_end(f"o{index}", value, when, prec=precision))
            elif rng.random() < 0.1:
                items.append(obs(f"o{index}", value, None))
            else:
                items.append(obs(f"o{index}", value, when, prec=precision))
        baseline = run(items)
        for _ in range(12):
            shuffled = items[:]
            rng.shuffle(shuffled)
            again = run(shuffled)
            assert timeline(again) == timeline(baseline)
            assert again.conflicts == baseline.conflicts
            assert again.unmatched_ends == baseline.unmatched_ends
            assert again.orphans == baseline.orphans
