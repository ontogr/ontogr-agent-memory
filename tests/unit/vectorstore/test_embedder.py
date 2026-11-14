"""Tests for the embedding gateway."""

import asyncio
import math
from collections.abc import Sequence

import pytest

from ontogr_mem.vectorstore import (
    EmbeddingGateway,
    EmbeddingPurpose,
    FailReason,
    TransientEmbeddingError,
    VectorStoreSettings,
)
from tests.support.embedder import (
    FailingEmbedder,
    MockEmbedder,
)
from tests.support.embedder import (
    gateway as make_gateway,
)


class SlowEmbedder(MockEmbedder):
    """An embedder that outlasts any attempt timeout."""

    async def embed_batch(
        self, texts: Sequence[str], *, purpose: EmbeddingPurpose
    ) -> list[list[float]]:
        """Wait, then embed."""
        await asyncio.sleep(1.0)
        return await super().embed_batch(texts, purpose=purpose)


class RaggedEmbedder(MockEmbedder):
    """An embedder that drops the last vector."""

    async def embed_batch(
        self, texts: Sequence[str], *, purpose: EmbeddingPurpose
    ) -> list[list[float]]:
        """Return one vector too few."""
        return (await super().embed_batch(texts, purpose=purpose))[:-1]


class TransientEmbedder(FailingEmbedder):
    """An embedder that fails every call and counts the attempts."""

    def __init__(self) -> None:
        """Always raise a transient error."""
        super().__init__(transient_over=0)
        self.calls = 0

    async def embed_batch(
        self, texts: Sequence[str], *, purpose: EmbeddingPurpose
    ) -> list[list[float]]:
        """Count the call, then fail."""
        self.calls += 1
        return await super().embed_batch(texts, purpose=purpose)


class PurposeSensitiveEmbedder(MockEmbedder):
    """Return a distinct vector for each purpose after one transient batch error."""

    def __init__(self) -> None:
        """Use a short limit so each request is segmented."""
        super().__init__(dim=2, max_input_tokens=5)
        self.failed_purposes: set[EmbeddingPurpose] = set()

    async def embed_batch(
        self, texts: Sequence[str], *, purpose: EmbeddingPurpose
    ) -> list[list[float]]:
        """Fail the first multi-segment call for each purpose."""
        if len(texts) > 1 and purpose not in self.failed_purposes:
            self.failed_purposes.add(purpose)
            raise TransientEmbeddingError("retry this purpose")
        vector = [1.0, 0.0] if purpose == "query" else [0.0, 1.0]
        return [vector.copy() for _ in texts]


class SleepRecorder:
    """Records the retry delays the gateway asks for."""

    def __init__(self) -> None:
        """Start with no recorded delays."""
        self.delays: list[float] = []

    async def __call__(self, seconds: float) -> None:
        """Record a delay instead of waiting."""
        self.delays.append(seconds)


def gateway_with_sleep(
    embedder: MockEmbedder, sleeper: SleepRecorder, **settings: float
) -> EmbeddingGateway:
    """Wrap `embedder` in a gateway whose retry delays are recorded."""
    return EmbeddingGateway(embedder, VectorStoreSettings(**settings), sleep=sleeper)


class TestEmbeddingGateway:
    """Behavior of `EmbeddingGateway.embed`."""

    async def test_outcomes_align_with_the_input_and_are_unit_vectors(self) -> None:
        """Each text gets the vector of that text, in order and of norm one."""
        texts = ["alpha", "beta", "gamma", "delta", "epsilon"]

        out = await make_gateway(MockEmbedder(), batch_size=2).embed(
            texts, purpose="document"
        )

        direct = await MockEmbedder().embed_batch(texts, purpose="document")
        assert len(out) == len(texts)
        for outcome, vector in zip(out, direct, strict=True):
            assert outcome.reason is None
            assert outcome.vector is not None
            assert outcome.vector == pytest.approx(vector)
            assert math.hypot(*outcome.vector) == pytest.approx(1.0)

    async def test_blank_texts_fail_in_place_and_others_succeed(self) -> None:
        """A blank text is reported at its index and does not block the rest."""
        out = await make_gateway(MockEmbedder()).embed(
            ["alpha", "  ", "beta", ""], purpose="document"
        )

        assert [o.reason for o in out] == [
            None,
            FailReason.EMPTY_TEXT,
            None,
            FailReason.EMPTY_TEXT,
        ]
        assert out[0].vector is not None
        assert out[2].vector is not None

    async def test_transient_failure_splits_until_every_text_passes(self) -> None:
        """A batch the provider rejects is retried in halves after a delay."""
        sleeper = SleepRecorder()
        texts = ["a", "b", "c", "d"]

        out = await gateway_with_sleep(
            FailingEmbedder(transient_over=1), sleeper, batch_size=4
        ).embed(texts, purpose="document")

        direct = await MockEmbedder().embed_batch(texts, purpose="document")
        for outcome, vector in zip(out, direct, strict=True):
            assert outcome.reason is None
            assert outcome.vector is not None
            assert outcome.vector == pytest.approx(vector)
        assert len(sleeper.delays) == 3
        assert all(delay > 0 for delay in sleeper.delays)

    async def test_a_text_that_keeps_failing_is_tried_max_attempts_times(
        self,
    ) -> None:
        """A single text that never passes fails after the configured tries."""
        sleeper = SleepRecorder()
        embedder = TransientEmbedder()

        out = await gateway_with_sleep(embedder, sleeper, max_attempts=3).embed(
            ["only"], purpose="document"
        )

        assert out[0].reason is FailReason.EMBED_FAILED
        assert "rate limited" in out[0].detail
        assert embedder.calls == 3
        assert len(sleeper.delays) == 2

    async def test_permanent_failure_fails_every_text_without_a_retry(self) -> None:
        """A permanent provider error is reported once and never retried."""
        sleeper = SleepRecorder()

        out = await gateway_with_sleep(
            FailingEmbedder(permanent=True), sleeper, batch_size=10
        ).embed(["a", "b"], purpose="document")

        assert [o.reason for o in out] == [FailReason.EMBED_FAILED] * 2
        assert "bad credentials" in out[0].detail
        assert sleeper.delays == []

    async def test_attempt_timeout_is_not_split_or_retried(self) -> None:
        """A call past the attempt timeout fails the whole batch at once."""
        sleeper = SleepRecorder()

        out = await gateway_with_sleep(
            SlowEmbedder(), sleeper, attempt_timeout_s=0.01, batch_size=10
        ).embed(["a", "b", "c"], purpose="document")

        assert [o.reason for o in out] == [FailReason.EMBED_TIMEOUT] * 3
        assert sleeper.delays == []

    async def test_deadline_stops_retries_and_reports_a_timeout(self) -> None:
        """Once the request deadline passes, the remaining texts time out."""
        embedder = TransientEmbedder()

        out = await EmbeddingGateway(
            embedder,
            VectorStoreSettings(batch_size=10, deadline_s=0.2, backoff_base_s=1.0),
        ).embed(["a", "b"], purpose="document")

        assert [o.reason for o in out] == [FailReason.EMBED_TIMEOUT] * 2
        assert "deadline" in out[0].detail
        assert embedder.calls == 1

    async def test_over_long_text_is_split_with_overlap_and_pooled(self) -> None:
        """A text over the piece limit is embedded as overlapping pieces."""
        text = "0123456789"
        embedder = MockEmbedder(
            max_input_tokens=5,
            vectors={
                "01234": [1.0, 0.0, 0.0],
                "45678": [0.0, 1.0, 0.0],
                "89": [0.0, 0.0, 1.0],
            },
        )

        out = await make_gateway(embedder, chars_per_token=1).embed(
            [text], purpose="document"
        )

        assert out[0].vector == pytest.approx(
            (1 / math.sqrt(3), 1 / math.sqrt(3), 1 / math.sqrt(3))
        )

    @pytest.mark.parametrize(
        ("max_input_tokens", "token_cap"),
        [(8192, 1), (0, 4096)],
        ids=["token_cap_of_one", "provider_reports_zero_input_tokens"],
    )
    async def test_over_long_text_with_a_one_character_piece_limit_terminates(
        self, max_input_tokens: int, token_cap: int
    ) -> None:
        """A piece limit of one character still splits the text and finishes."""
        embedder = MockEmbedder(max_input_tokens=max_input_tokens)

        # The split loop is synchronous, so a regression hangs this test.
        out = await make_gateway(
            embedder, token_cap=token_cap, chars_per_token=1
        ).embed(["abc"], purpose="document")

        assert out[0].reason is None
        assert out[0].vector is not None

    async def test_text_at_the_piece_limit_is_embedded_once(self) -> None:
        """A text exactly at the limit is not cut."""
        embedder = MockEmbedder(
            max_input_tokens=10, vectors={"0123456789": [1.0, 0.0, 0.0]}
        )

        out = await make_gateway(embedder, chars_per_token=1).embed(
            ["0123456789"], purpose="document"
        )

        assert out[0].vector == pytest.approx((1.0, 0.0, 0.0))

    async def test_each_bad_vector_fails_only_its_own_text(self) -> None:
        """Empty, short, non-finite and zero-norm vectors are reported in place."""
        embedder = MockEmbedder(
            vectors={
                "empty": [],
                "short": [1.0, 0.0],
                "nan": [math.nan, 0.0, 0.0],
                "zero": [0.0, 0.0, 0.0],
                "good": [3.0, 4.0, 0.0],
            }
        )

        out = await make_gateway(embedder, batch_size=10).embed(
            ["empty", "short", "nan", "zero", "good"], purpose="document"
        )

        assert [o.reason for o in out] == [
            FailReason.EMPTY_VECTOR,
            FailReason.BAD_DIMENSION,
            FailReason.NON_FINITE,
            FailReason.ZERO_NORM,
            None,
        ]
        assert out[4].vector == pytest.approx((0.6, 0.8, 0.0))

    async def test_wrong_number_of_vectors_fails_the_batch(self) -> None:
        """A provider that returns too few vectors fails every text."""
        out = await make_gateway(RaggedEmbedder(), batch_size=10).embed(
            ["a", "b"], purpose="document"
        )

        assert [o.reason for o in out] == [FailReason.EMBED_FAILED] * 2
        assert "ragged" in out[0].detail

    async def test_purpose_survives_retries_and_segment_pooling(self) -> None:
        """Each purpose keeps its vector through retries and long-text pooling."""
        embedder = PurposeSensitiveEmbedder()
        gateway_instance = make_gateway(embedder, chars_per_token=1, batch_size=3)

        query = await gateway_instance.embed(["abcdefghij"], purpose="query")
        document = await gateway_instance.embed(["abcdefghij"], purpose="document")

        assert query[0].vector == pytest.approx((1.0, 0.0))
        assert document[0].vector == pytest.approx((0.0, 1.0))
