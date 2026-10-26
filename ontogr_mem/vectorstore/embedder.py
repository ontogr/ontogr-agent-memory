"""Embedding gateway: batching, retries, splitting of long texts, timeouts."""

import asyncio
import random
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from ontogr_mem.observability import NoOpTracer, SpanName, Tracer, open_span
from ontogr_mem.request_context import current_context
from ontogr_mem.resilience import (
    CircuitBreaker,
    Kind,
    RateLimiter,
    backoff_delay,
    call_with_breaker,
    provider_limits,
)
from ontogr_mem.vectorstore.errors import (
    EmbeddingError,
    EmbeddingFailedError,
    EmbeddingTimeoutError,
    PermanentEmbeddingError,
    TransientEmbeddingError,
)
from ontogr_mem.vectorstore.models import FailReason
from ontogr_mem.vectorstore.settings import VectorStoreSettings
from ontogr_mem.vectorstore.vecmath import classify_vector, mean_unit, norm


def _breaker_error(exc: BaseException) -> Kind | None:
    """Separate provider failures from deadline and cancellation failures."""
    if isinstance(exc, EmbeddingTimeoutError):
        return None if "deadline" in str(exc).casefold() else Kind.TRANSIENT
    if isinstance(exc, TransientEmbeddingError):
        return Kind.TRANSIENT
    if isinstance(exc, PermanentEmbeddingError):
        return Kind.TERMINAL
    return None


class Embedder(Protocol):
    """An embedding provider.

    Implementations raise the typed errors in `ontogr_mem.vectorstore.errors`.

    Attributes:
        model_id: Model identity string, stored with the space.
        dim: Length of every vector.
        max_input_tokens: Longest input the model accepts.
    """

    model_id: str
    dim: int
    max_input_tokens: int

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed `texts`; one vector per text, in order."""
        ...


@dataclass(frozen=True, slots=True)
class EmbedOutcome:
    """Result for one input text: a unit vector, or the reason there is none."""

    vector: tuple[float, ...] | None
    reason: FailReason | None = None
    detail: str = ""


@dataclass(frozen=True, slots=True)
class _Segment:
    owner: int
    text: str


class EmbeddingGateway:
    """Turns texts into unit vectors or explicit failures; never returns `[]`."""

    def __init__(
        self,
        embedder: Embedder,
        settings: VectorStoreSettings | None = None,
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        breaker: CircuitBreaker | None = None,
        limiter: RateLimiter | None = None,
        rng: random.Random | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        """Wrap an embedder; `sleep` is injectable for tests."""
        self.embedder = embedder
        self.settings = settings or VectorStoreSettings()
        self._sleep = sleep
        self._rng = rng or random.Random()
        self._tracer = tracer if tracer is not None else NoOpTracer()
        self._breaker, self._limiter = provider_limits(
            self.settings, breaker=breaker, limiter=limiter, sleep=sleep
        )

    @property
    def model_id(self) -> str:
        """Model identity string of the provider."""
        return self.embedder.model_id

    @property
    def dim(self) -> int:
        """Vector length of the provider."""
        return self.embedder.dim

    async def embed(self, texts: Sequence[str]) -> list[EmbedOutcome]:
        """Embed `texts`; the result is aligned with the input.

        Texts longer than the model accepts are cut into overlapping pieces and
        the piece vectors are averaged.
        """
        with open_span(
            self._tracer,
            SpanName.LLM_EMBED,
            model=self.embedder.model_id,
            count=len(texts),
        ):
            return await self._embed(texts)

    async def _embed(self, texts: Sequence[str]) -> list[EmbedOutcome]:
        """Embed a batch using the configured split and retry rules."""
        cfg = self.settings
        now = time.monotonic()
        deadline = now + cfg.deadline_s
        context = current_context()
        remaining = None if context is None else context.remaining_seconds()
        if remaining is not None:
            deadline = min(deadline, now + remaining)
        char_limit = max(
            1,
            min(cfg.token_cap, self.embedder.max_input_tokens) * cfg.chars_per_token,
        )
        segments: list[_Segment] = []
        empty: set[int] = set()
        for i, text in enumerate(texts):
            if not text.strip():
                empty.add(i)
                continue
            segments.extend(_Segment(i, p) for p in _split(text, char_limit))

        gate = asyncio.Semaphore(cfg.concurrency)

        async def run(batch: list[_Segment]) -> list[EmbedOutcome]:
            async with gate:
                return await self._embed_batch(batch, deadline)

        batches = [
            segments[s : s + cfg.batch_size]
            for s in range(0, len(segments), cfg.batch_size)
        ]
        results = await asyncio.gather(*(run(b) for b in batches))
        by_owner: dict[int, list[EmbedOutcome]] = {}
        for seg, res in zip(
            segments, (r for batch in results for r in batch), strict=True
        ):
            by_owner.setdefault(seg.owner, []).append(res)

        return [
            EmbedOutcome(None, FailReason.EMPTY_TEXT, "empty text")
            if i in empty
            else _combine(by_owner[i])
            for i in range(len(texts))
        ]

    async def probe(self, text: str = "vectorstore probe") -> tuple[int, float]:
        """Embed a fixed string; return the length and norm of the raw vector."""
        now = time.monotonic()
        deadline = now + self.settings.deadline_s
        context = current_context()
        remaining = None if context is None else context.remaining_seconds()
        if remaining is not None:
            deadline = min(deadline, now + remaining)
        if deadline <= time.monotonic():
            raise EmbeddingTimeoutError("deadline exceeded")

        breaker = self._breaker
        if breaker is not None and not breaker.allow_request():
            raise EmbeddingError("circuit open")

        async def call() -> list[float]:
            return (await self._call([text], deadline))[0]

        vector = await call_with_breaker(call, breaker, _breaker_error)
        return len(vector), norm(vector)

    async def embed_all(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        """Return vectors for every text or raise on the first failed outcome.

        Raises:
            EmbeddingFailedError: If any input text has no vector.
        """
        outcomes = await self.embed(texts)
        vectors: list[tuple[float, ...]] = []
        for index, outcome in enumerate(outcomes):
            if outcome.vector is None:
                detail = outcome.detail or (
                    outcome.reason.value if outcome.reason is not None else "unknown"
                )
                raise EmbeddingFailedError(index, detail)
            vectors.append(outcome.vector)
        return vectors

    async def _call(self, texts: Sequence[str], deadline: float) -> list[list[float]]:
        if self._limiter is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise EmbeddingTimeoutError("deadline exceeded")
            limiter_timeout = asyncio.timeout(remaining)
            try:
                async with limiter_timeout:
                    await self._limiter.acquire()
            except TimeoutError as exc:
                if limiter_timeout.expired():
                    raise EmbeddingTimeoutError("deadline exceeded") from exc
                raise

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise EmbeddingTimeoutError("deadline exceeded")
        deadline_limited = remaining <= self.settings.attempt_timeout_s
        timeout_scope = asyncio.timeout(min(self.settings.attempt_timeout_s, remaining))
        try:
            async with timeout_scope:
                return await self.embedder.embed_batch(texts)
        except TimeoutError as exc:
            msg = (
                "deadline exceeded"
                if timeout_scope.expired() and deadline_limited
                else "embedder call timed out"
            )
            raise EmbeddingTimeoutError(msg) from exc

    def _backoff(self, attempt: int) -> float:
        return backoff_delay(
            attempt,
            base_s=self.settings.backoff_base_s,
            max_s=self.settings.backoff_max_s,
            rng=self._rng,
        )

    # Breaker probe and split retries share local state, so the flow stays whole.
    async def _embed_batch(  # noqa: PLR0912
        self,
        batch: list[_Segment],
        deadline: float,
        attempt: int = 0,
        breaker_checked: bool = False,
    ) -> list[EmbedOutcome]:
        """Embed one batch, splitting it on transient errors. Timeouts never split."""
        if time.monotonic() >= deadline:
            return _fail(batch, FailReason.EMBED_TIMEOUT, "deadline exceeded")
        owns_probe = False
        if not breaker_checked and self._breaker is not None:
            if not self._breaker.allow_request():
                return _fail(batch, FailReason.EMBED_FAILED, "circuit open")
            breaker_checked = True
            owns_probe = True
        try:
            try:
                raw = await self._call([s.text for s in batch], deadline)
            except EmbeddingTimeoutError as exc:
                if self._breaker is not None:
                    if "deadline" in str(exc).lower():
                        if owns_probe:
                            self._breaker.release_probe()
                    else:
                        self._breaker.record_failure()
                return _fail(batch, FailReason.EMBED_TIMEOUT, str(exc))
            except PermanentEmbeddingError as exc:
                if self._breaker is not None:
                    self._breaker.record_success()
                return _fail(batch, FailReason.EMBED_FAILED, f"permanent: {exc}")
            except TransientEmbeddingError as exc:
                return await self._retry(
                    batch,
                    deadline,
                    attempt,
                    exc,
                    breaker_checked,
                    owns_probe=owns_probe,
                )
            if self._breaker is not None:
                self._breaker.record_success()
            return self._classify(batch, raw)
        except asyncio.CancelledError:
            if owns_probe and self._breaker is not None:
                self._breaker.release_probe()
            raise
        except Exception:
            if owns_probe and self._breaker is not None:
                self._breaker.release_probe()
            raise

    async def _retry(
        self,
        batch: list[_Segment],
        deadline: float,
        attempt: int,
        error: TransientEmbeddingError,
        breaker_checked: bool,
        *,
        owns_probe: bool,
    ) -> list[EmbedOutcome]:
        if len(batch) == 1 and attempt + 1 >= self.settings.max_attempts:
            if self._breaker is not None:
                self._breaker.record_failure()
            return _fail(batch, FailReason.EMBED_FAILED, f"transient: {error}")
        delay = self._backoff(attempt)
        if time.monotonic() + delay >= deadline:
            if owns_probe and self._breaker is not None:
                self._breaker.release_probe()
            return _fail(batch, FailReason.EMBED_TIMEOUT, "deadline exceeded")
        await self._sleep(delay)
        if len(batch) == 1:
            outcomes = await self._embed_batch(
                batch, deadline, attempt + 1, breaker_checked
            )
            if owns_probe and self._breaker is not None and _deadline_only(outcomes):
                self._breaker.release_probe()
            return outcomes
        mid = len(batch) // 2
        left, right = await asyncio.gather(
            self._embed_batch(batch[:mid], deadline, attempt + 1, breaker_checked),
            self._embed_batch(batch[mid:], deadline, attempt + 1, breaker_checked),
        )
        outcomes = left + right
        if owns_probe and self._breaker is not None and _deadline_only(outcomes):
            self._breaker.release_probe()
        return outcomes

    def _classify(
        self, batch: list[_Segment], raw: list[list[float]]
    ) -> list[EmbedOutcome]:
        if len(raw) != len(batch):
            return _fail(
                batch,
                FailReason.EMBED_FAILED,
                f"ragged output: {len(raw)} vectors for {len(batch)} texts",
            )
        outcomes = []
        for vector in raw:
            unit, reason, detail = classify_vector(vector, self.embedder.dim)
            outcomes.append(EmbedOutcome(unit, reason, detail))
        return outcomes


def _fail(batch: list[_Segment], reason: FailReason, detail: str) -> list[EmbedOutcome]:
    return [EmbedOutcome(None, reason, detail) for _ in batch]


def _deadline_only(outcomes: list[EmbedOutcome]) -> bool:
    """Return whether every outcome came from an exhausted request deadline."""
    return bool(outcomes) and all(
        outcome.reason is FailReason.EMBED_TIMEOUT
        and "deadline" in outcome.detail.lower()
        for outcome in outcomes
    )


def _combine(parts: list[EmbedOutcome]) -> EmbedOutcome:
    for part in parts:
        if part.vector is None:
            return part
    vectors = [p.vector for p in parts if p.vector is not None]
    if len(vectors) == 1:
        return parts[0]
    pooled = mean_unit(vectors)
    if pooled is None:
        return EmbedOutcome(None, FailReason.ZERO_NORM, "pooled pieces cancel out")
    return EmbedOutcome(pooled)


def _split(text: str, char_limit: int) -> list[str]:
    """Cut over-long text into overlapping pieces of at most `char_limit`."""
    if len(text) <= char_limit:
        return [text]
    overlap = max(1, char_limit // 10)
    step = max(1, char_limit - overlap)
    pieces: list[str] = []
    start = 0
    while True:
        pieces.append(text[start : start + char_limit])
        if start + char_limit >= len(text):
            return pieces
        start += step
