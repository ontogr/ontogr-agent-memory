"""Tests for the extraction value types."""

import pytest

from ontogr_mem.data_models import ExtractedEntity, Method, Usage


class TestExtractedEntity:
    """Behavior of ExtractedEntity."""

    @pytest.mark.parametrize("confidence", [-0.1, 1.1])
    def test_rejects_confidence_outside_zero_to_one(self, confidence):
        """A confidence outside 0..1 is refused."""
        with pytest.raises(ValueError, match="confidence"):
            ExtractedEntity("e0", "Ana", 0, confidence, Method.LLM, "ep1")


class TestUsage:
    """Behavior of Usage."""

    def test_merge_adds_tokens_and_calls_per_bucket(self):
        """Merging two usages sums matching buckets."""
        first, second = Usage(), Usage()
        first.add("entities", "small", 10, 2)
        second.add("entities", "small", 5, 1)
        second.add("enrich", "small", 7, 3)

        first.merge(second)

        assert {
            k: (b.input_tokens, b.output_tokens, b.calls)
            for k, b in first.buckets.items()
        } == {
            "entities/small": (15, 3, 2),
            "enrich/small": (7, 3, 1),
        }
