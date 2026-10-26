"""Tests for candidate ordering and the vote rule."""

import pytest

from ontogr_mem.resolution.config import ResolverSettings
from ontogr_mem.resolution.scoring import (
    Candidate,
    PairSignals,
    margin,
    needs_two_votes,
    rank_key,
)


SETTINGS = ResolverSettings()


class TestMargin:
    """Behavior of `margin`."""

    def test_returns_the_share_of_the_remaining_range(self) -> None:
        """A cosine of 0.90 clears the 0.80 floor by half the range."""
        assert margin(0.90, 0.80) == 0.5

    def test_clamps_a_score_below_the_floor_to_zero(self) -> None:
        """A score under its floor contributes nothing."""
        assert margin(0.50, 0.80) == 0.0

    def test_returns_a_literal_value_for_a_jaccard_floor(self) -> None:
        """The Jaccard scale is normalized by its own floor."""
        assert margin(0.75, 0.60) == pytest.approx(0.375)


class TestRankKey:
    """Behavior of `rank_key`."""

    def test_cosine_above_its_floor_outranks_jaccard_above_its_floor(self) -> None:
        """A cosine margin of 0.5 beats a Jaccard margin of 0.375."""
        cosine = Candidate("cos", signals=PairSignals(cosine=0.90))
        jaccard = Candidate("jac", signals=PairSignals(jaccard=0.75))

        assert sorted([jaccard, cosine], key=lambda c: rank_key(c, SETTINGS)) == [
            cosine,
            jaccard,
        ]

    def test_equal_margins_fall_back_to_the_smaller_entity_id(self) -> None:
        """The id breaks a tie between equally strong candidates."""
        first = Candidate("a", signals=PairSignals(cosine=0.90))
        second = Candidate("b", signals=PairSignals(cosine=0.90))

        assert rank_key(first, SETTINGS) < rank_key(second, SETTINGS)

    def test_a_block_only_candidate_sorts_last_but_is_kept(self) -> None:
        """A weak Jaccard hit has margin zero and still takes part."""
        weak = Candidate("weak", signals=PairSignals(jaccard=0.4))
        strong = Candidate("strong", signals=PairSignals(cosine=0.90))

        ranked = sorted([weak, strong], key=lambda c: rank_key(c, SETTINGS))

        assert ranked == [strong, weak]
        assert rank_key(weak, SETTINGS)[0] == 0.0


class TestNeedsTwoVotes:
    """Behavior of `needs_two_votes`."""

    def test_a_strong_cosine_needs_one_vote(self) -> None:
        """Cosine at or above the single-vote bar is accepted alone."""
        assert not needs_two_votes(PairSignals(cosine=0.97), SETTINGS, False)

    def test_a_mid_cosine_needs_two_votes(self) -> None:
        """A cosine below the single-vote bar needs a second vote."""
        assert needs_two_votes(PairSignals(cosine=0.95), SETTINGS, False)

    def test_a_strong_jaccard_needs_one_vote(self) -> None:
        """Jaccard at or above its single-vote bar is accepted alone."""
        assert not needs_two_votes(PairSignals(jaccard=0.93), SETTINGS, False)

    def test_a_jaccard_below_its_own_floor_needs_two_votes(self) -> None:
        """A Jaccard under the Jaccard bar needs a second vote."""
        assert needs_two_votes(PairSignals(jaccard=0.91), SETTINGS, False)

    def test_a_cross_family_pair_always_needs_two_votes(self) -> None:
        """Even a near-perfect cosine does not settle a cross-family pair."""
        assert needs_two_votes(PairSignals(cosine=0.99), SETTINGS, True)

    def test_a_pair_without_signals_needs_two_votes(self) -> None:
        """A pair no signal proposed is never settled on one vote."""
        assert needs_two_votes(PairSignals(), SETTINGS, False)
