"""The gate every model call passes through."""

import asyncio
import random
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from ontogr_mem.llm.base import LLMClient, LLMRequest, LLMResponse, ModelSize
from ontogr_mem.llm.cache import CacheEntry, ResponseCache, cache_key
from ontogr_mem.llm.errors import (
    CircuitOpen,
    DeadlineExceeded,
    InputTooLargeError,
    LLMCallError,
    LLMError,
    StructuredOutputError,
)
from ontogr_mem.llm.json_reply import SchemaError, load_json
from ontogr_mem.llm.pricing import PriceTable
from ontogr_mem.llm.settings import LLMSettings
from ontogr_mem.observability import NoOpTracer, Span, SpanName, Tracer, open_span
from ontogr_mem.request_context import (
    RequestContext,
    current_context,
    record_usage,
)
from ontogr_mem.resilience import (
    CircuitBreaker,
    Kind,
    RateLimiter,
    backoff_delay,
    call_with_breaker,
    classify,
    provider_limits,
)


M = TypeVar("M", bound=BaseModel)
T = TypeVar("T")


def estimate_tokens(text: str) -> int:
    """Estimate the token count as one token per four characters, rounded up."""
    return -(-len(text) // 4)


def _breaker_error(exc: BaseException) -> Kind | None:
    """Separate provider failures from deadline and cancellation failures."""
    if isinstance(exc, LLMCallError):
        return classify(exc)
    if isinstance(exc, TimeoutError):
        return Kind.TRANSIENT
    return None


@dataclass(frozen=True, slots=True)
class BatchItem:
    """One ordered result or typed failure from ``complete_many``.

    Attributes:
        index: Position of the request in the input sequence.
        response: Successful response, or None when the request failed.
        error: Typed request error, or None when the request succeeded.
    """

    index: int
    response: LLMResponse | None
    error: Exception | None


class LLMGate:
    """Apply bounded, tenant-aware protections to chat model calls."""

    def __init__(
        self,
        client: LLMClient,
        *,
        max_input_tokens: int,
        timeout_s: float = 60.0,
        concurrency: Mapping[ModelSize, int] | None = None,
        count_tokens: Callable[[str], int] = estimate_tokens,
        settings: LLMSettings | None = None,
        model_name: str = "unknown",
        breaker: CircuitBreaker | None = None,
        limiter: RateLimiter | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        rng: random.Random | None = None,
        cache: ResponseCache | None = None,
        prices: PriceTable | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        """Create the gate.

        Args:
            client: The provider adapter to call.
            max_input_tokens: Hard cap on system plus prompt tokens.
            timeout_s: Seconds a single call may take.
            concurrency: Maximum calls per size, overriding the settings default.
            count_tokens: Token counter; inject a real tokenizer in production.
            settings: Retry, concurrency, and resilience defaults.
            model_name: Name used for cache keys and cost lookup.
            breaker: Optional breaker override.
            limiter: Optional rate-limiter override.
            sleep: Async wait function for retry delays.
            rng: Random source used for equal-jitter retry delays.
            cache: Optional tenant-scoped response cache.
            prices: Optional model price table.
            tracer: Optional span recorder; defaults to no-op tracing.
        """
        self.settings = settings or LLMSettings()
        self._client = client
        self.max_input_tokens = max_input_tokens
        self._timeout_s = timeout_s
        self._count = count_tokens
        self._model_name = model_name
        self._sleep = sleep
        self._rng = rng or random.Random()
        self._cache = cache
        self._prices = prices or PriceTable()
        self._tracer = tracer if tracer is not None else NoOpTracer()
        self._breaker, self._limiter = provider_limits(
            self.settings, breaker=breaker, limiter=limiter, sleep=sleep
        )
        limits = dict.fromkeys(ModelSize, self.settings.max_concurrency) | dict(
            concurrency or {}
        )
        self._semaphores = {size: asyncio.Semaphore(n) for size, n in limits.items()}

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Run a request under the input cap, the timeout and the size limit.

        Raises:
            InputTooLargeError: If the prompt exceeds the cap. The client is not
                called.
            TimeoutError: If the call takes longer than the timeout.
        """
        size = self._count(request.system) + self._count(request.prompt)
        with open_span(
            self._tracer,
            SpanName.LLM_COMPLETE,
            model=self._model_name,
            size=request.model_size.value,
        ) as span:
            return await self._complete(request, size, span)

    async def _complete(
        self, request: LLMRequest, size: int, span: Span
    ) -> LLMResponse:
        """Run one completion and update its span with cache and usage data."""
        if size > self.max_input_tokens:
            msg = f"prompt is {size} tokens, cap is {self.max_input_tokens}"
            raise InputTooLargeError(msg)
        context = current_context()
        tenant = None if context is None else context.tenant
        key = None
        if self._cache is not None and tenant is not None:
            key = cache_key(
                model_name=self._model_name,
                model_size=request.model_size,
                tenant=tenant,
                system=request.system,
                prompt=request.prompt,
                schema=request.schema,
                max_output_tokens=request.max_output_tokens,
            )
            cached = await self._wait_with_deadline(self._cache.get(key), context)
            if cached is not None:
                span.set("cache.hit", True)
                span.set("tokens.in", 0)
                span.set("tokens.out", 0)
                tags = frozenset() if context is None else frozenset(context.cache_tags)
                if not tags.issubset(cached.tags):
                    cached = replace(cached, tags=cached.tags | tags)
                    await self._wait_with_deadline(
                        self._cache.put(key, cached), context
                    )
                return LLMResponse(
                    cached.text, cached.input_tokens, cached.output_tokens
                )

        span.set("cache.hit", False)
        response = await self._call(request)
        span.set("tokens.in", response.input_tokens)
        span.set("tokens.out", response.output_tokens)
        record_usage(
            response.input_tokens,
            response.output_tokens,
            self._prices.cost(
                self._model_name, response.input_tokens, response.output_tokens
            ),
        )
        if key is not None and self._cache is not None:
            tags = frozenset() if context is None else frozenset(context.cache_tags)
            await self._wait_with_deadline(
                self._cache.put(
                    key,
                    CacheEntry(
                        request.prompt,
                        response.text,
                        response.input_tokens,
                        response.output_tokens,
                        tags,
                    ),
                ),
                context,
            )
        return response

    async def complete_many(self, requests: Sequence[LLMRequest]) -> list[BatchItem]:
        """Complete requests concurrently, retaining typed failures by index."""

        async def complete_one(index: int, request: LLMRequest) -> BatchItem:
            try:
                response = await self.complete(request)
            except (LLMError, TimeoutError) as exc:
                return BatchItem(index, None, exc)
            return BatchItem(index, response, None)

        return await asyncio.gather(
            *(complete_one(index, request) for index, request in enumerate(requests))
        )

    async def structured(
        self,
        model: type[M],
        system: str,
        prompt: str,
        *,
        model_size: ModelSize = ModelSize.SMALL,
        max_output_tokens: int = 1024,
        attempts: int | None = None,
    ) -> M:
        """Return a model-validated response, feeding validation errors back.

        Args:
            model: Pydantic model that defines the expected JSON shape.
            system: System prompt sent on every attempt.
            prompt: Initial user prompt.
            model_size: Model tier used for concurrency limits.
            max_output_tokens: Output cap on each request.
            attempts: Total validation attempts; defaults to gateway settings.

        Raises:
            StructuredOutputError: If no response validates within the limit.
            LLMError: If a model call fails.
        """
        attempt_limit = (
            self.settings.structured_attempts if attempts is None else attempts
        )
        if attempt_limit < 1:
            raise ValueError("structured attempts must be positive")
        schema = model.model_json_schema()
        current_prompt = prompt
        for attempt in range(attempt_limit):
            request = LLMRequest(
                system=system,
                prompt=current_prompt,
                model_size=model_size,
                schema=schema,
                max_output_tokens=max_output_tokens,
            )
            response = await self.complete(request)
            try:
                return model.model_validate(load_json(response.text))
            except (SchemaError, ValidationError) as exc:
                errors = _validation_messages(exc)
                context = current_context()
                if self._cache is not None and context is not None and context.tenant:
                    await self._wait_with_deadline(
                        self._cache.delete(
                            cache_key(
                                model_name=self._model_name,
                                model_size=model_size,
                                tenant=context.tenant,
                                system=system,
                                prompt=current_prompt,
                                schema=schema,
                                max_output_tokens=max_output_tokens,
                            )
                        ),
                        context,
                    )
                if attempt + 1 == attempt_limit:
                    detail = "; ".join(errors)
                    msg = (
                        f"structured output for {model.__name__} failed after "
                        f"{attempt_limit} attempts: {detail}"
                    )
                    raise StructuredOutputError(msg) from exc
                current_prompt = (
                    f"{current_prompt}\n\nValidation errors:\n" + "\n".join(errors)
                )
        raise RuntimeError("unreachable structured-output state")

    async def _call(self, request: LLMRequest) -> LLMResponse:
        """Run one provider request under retries, limits, and the breaker."""
        last_error: LLMCallError | None = None
        for attempt in range(self.settings.max_attempts):
            context = current_context()
            self._remaining(context)
            if self._breaker is not None and not self._breaker.allow_request():
                raise CircuitOpen("circuit breaker is open")
            try:
                return await call_with_breaker(
                    partial(self._attempt, request, context),
                    self._breaker,
                    _breaker_error,
                )
            except LLMCallError as exc:
                if classify(exc) is Kind.TERMINAL:
                    raise
                last_error = exc

            if attempt + 1 < self.settings.max_attempts:
                delay = backoff_delay(
                    attempt,
                    base_s=self.settings.base_delay_s,
                    max_s=self.settings.max_delay_s,
                    rng=self._rng,
                )
                remaining = self._remaining(current_context())
                if remaining is not None and delay > remaining:
                    raise DeadlineExceeded("request deadline exceeded")
                await self._wait_with_deadline(self._sleep(delay), current_context())

        msg = f"LLM call failed after {self.settings.max_attempts} attempts"
        raise LLMCallError(msg) from last_error

    async def _attempt(
        self, request: LLMRequest, context: RequestContext | None
    ) -> LLMResponse:
        """Run one attempt under concurrency, rate and deadline limits."""
        semaphore = self._semaphores[request.model_size]
        acquired = False
        try:
            await self._wait_with_deadline(semaphore.acquire(), context)
            acquired = True
            self._remaining(context)
            if self._limiter is not None:
                await self._wait_with_deadline(self._limiter.acquire(), context)
            remaining = self._remaining(context)
            call_timeout = (
                self._timeout_s
                if remaining is None
                else min(self._timeout_s, remaining)
            )
            timeout_scope = asyncio.timeout(call_timeout)
            try:
                async with timeout_scope:
                    return await self._client.complete(request)
            except TimeoutError as exc:
                if (
                    timeout_scope.expired()
                    and remaining is not None
                    and remaining <= self._timeout_s
                ):
                    raise DeadlineExceeded("request deadline exceeded") from exc
                raise
        finally:
            if acquired:
                semaphore.release()

    @staticmethod
    def _remaining(context: RequestContext | None) -> float | None:
        """Return a request's remaining time or refuse an expired request."""
        remaining = None if context is None else context.remaining_seconds()
        if remaining is not None and remaining <= 0:
            raise DeadlineExceeded("request deadline exceeded")
        return remaining

    async def _wait_with_deadline(
        self, awaitable: Awaitable[T], context: RequestContext | None
    ) -> T:
        """Await a value without exceeding the request budget."""
        remaining = self._remaining(context)
        if remaining is None:
            return await awaitable
        timeout_scope = asyncio.timeout(remaining)
        try:
            async with timeout_scope:
                return await awaitable
        except TimeoutError as exc:
            if timeout_scope.expired():
                raise DeadlineExceeded("request deadline exceeded") from exc
            raise


def _validation_messages(error: SchemaError | ValidationError) -> list[str]:
    """Return concise parse or field errors suitable for model feedback."""
    if isinstance(error, SchemaError):
        return error.errors
    return [
        f"{'.'.join(map(str, item['loc']))}: {item['msg']}" for item in error.errors()
    ]
