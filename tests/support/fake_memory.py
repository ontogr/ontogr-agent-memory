"""A value-based stand-in for the memory client in unit tests.

The fake records what callers pass in as plain values, so tests assert on the
queries, budgets, texts and invalidations it saw instead of on call histories.
Its method signatures match ``ontogr_mem.client.Memory`` for the operations the
memory capability, the default agent and the chat loop use. It does not run any
background work.
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from ontogr_mem.data_models import Role
from ontogr_mem.data_models.jobs import JobState
from ontogr_mem.llm.models import RoleModels
from ontogr_mem.service.contracts import (
    EntityRef,
    FactOut,
    MemoryContext,
    WriteReceipt,
)
from tests.support.scripted_model import scripted_role_models


@dataclass(frozen=True, slots=True)
class AcceptedText:
    """A text that a write accepted, with the options it was given.

    Attributes:
        content: The text passed to ``write_text``.
        role: The speaker's role, or None when the caller left it unset.
        speaker: The speaker id and display name, or None.
        session_id: The session the text belongs to, or None.
        thread: The thread the text belongs to, or None.
        reference_time: When the text's event happened, or None.
    """

    content: str
    role: Role | None
    speaker: str | None
    session_id: str | None
    thread: str | None
    reference_time: datetime | None


@dataclass(frozen=True, slots=True)
class Invalidation:
    """A fact that ``invalidate`` closed.

    Attributes:
        fact_id: The id of the closed fact.
        reason: The reason given to ``invalidate``, which may be empty.
    """

    fact_id: str
    reason: str


class FakeMemory:
    """A memory client that records accepted calls as plain values.

    Search returns the context that the test scripts. Writes and invalidations
    return queued receipts. Each receipt's job id counts every accepted write
    and invalidation so far, so the first receipt is ``job-1``.

    Args:
        context: A function from query to context, a fixed context, or None.
            When None, every search returns a context with state ``"none"``.
        models: The model roles to expose. When None, the scripted roles from
            ``tests.support.scripted_model`` are used.
        tenant: The tenant id reported by ``tenant`` and set on every receipt.
        write_error_on: The one-based number of the ``write_text`` call that
            raises ``RuntimeError``. Every call counts, including the failed
            one. None means no call raises.

    Attributes:
        queries: Every query given to ``search``, in order.
        budgets: The ``budget_tokens`` of every ``search`` call, in order.
        texts: Texts accepted by ``write_text``, in order. A call that raises
            records nothing.
        invalidations: Facts closed by ``invalidate``, in order.
        drained: True once ``drain`` has completed.
        drain_started: Set when ``drain`` begins.
        drain_gate: When set, ``drain`` waits on it before it completes. Assign
            an ``asyncio.Event`` to hold ``drain`` open until the test releases it.
    """

    def __init__(
        self,
        *,
        context: Callable[[str], MemoryContext] | MemoryContext | None = None,
        models: RoleModels | None = None,
        tenant: str = "tenant",
        write_error_on: int | None = None,
    ) -> None:
        """Create a fake client with the scripted context and models."""
        self._context = context
        self._models = models if models is not None else scripted_role_models()
        self._tenant = tenant
        self._write_error_on = write_error_on
        self._write_calls = 0
        self.queries: list[str] = []
        self.budgets: list[int] = []
        self.texts: list[AcceptedText] = []
        self.invalidations: list[Invalidation] = []
        self.drained = False
        self.drain_started = asyncio.Event()
        self.drain_gate: asyncio.Event | None = None

    @property
    def tenant(self) -> str:
        """Return the tenant id reported by the client."""
        return self._tenant

    @property
    def models(self) -> RoleModels:
        """Return the model roles that were given, or the scripted defaults."""
        return self._models

    async def search(
        self,
        query: str,
        *,
        as_of: datetime | None = None,
        include_history: bool = False,
        limit: int = 10,
        budget_tokens: int = 2000,
    ) -> MemoryContext:
        """Record the query and budget, then return the scripted context.

        The time, history and limit arguments are accepted so that calls match
        the client signature, but they do not change the result.
        """
        self.queries.append(query)
        self.budgets.append(budget_tokens)
        context = self._context
        if isinstance(context, MemoryContext):
            return context
        if context is None:
            return MemoryContext(state="none", rendered="")
        return context(query)

    async def write_text(
        self,
        content: str,
        *,
        role: Role | None = None,
        speaker: str | None = None,
        session_id: str | None = None,
        thread: str | None = None,
        reference_time: datetime | None = None,
        idempotency_key: str | None = None,
        wait_seconds: float = 0.0,
    ) -> WriteReceipt:
        """Record an accepted text and return a queued receipt.

        The idempotency key and wait time are accepted so that calls match the
        client signature, but they are not recorded. The call selected by
        ``write_error_on`` raises ``RuntimeError`` and records nothing.
        """
        # Count failed calls too, so write_error_on names the nth call the test
        # made rather than the nth accepted write.
        self._write_calls += 1
        if self._write_calls == self._write_error_on:
            msg = "write failed"
            raise RuntimeError(msg)
        self.texts.append(
            AcceptedText(
                content=content,
                role=role,
                speaker=speaker,
                session_id=session_id,
                thread=thread,
                reference_time=reference_time,
            )
        )
        return self._queued_receipt()

    async def invalidate(self, fact_id: str, *, reason: str = "") -> WriteReceipt:
        """Record a closed fact and return a queued receipt.

        Invalidations share job numbers with accepted writes, so a test sees one
        sequence across both kinds of call.
        """
        self.invalidations.append(Invalidation(fact_id=fact_id, reason=reason))
        return self._queued_receipt()

    async def drain(self) -> None:
        """Mark the drain as started, wait for ``drain_gate`` if set, then finish.

        A test can read ``drained`` as False while the gate is held and True
        after the gate is released.
        """
        self.drain_started.set()
        if self.drain_gate is not None:
            await self.drain_gate.wait()
        self.drained = True

    def _queued_receipt(self) -> WriteReceipt:
        """Return a queued receipt numbered after every accepted write so far."""
        job_number = len(self.texts) + len(self.invalidations)
        return WriteReceipt(
            job_id=f"job-{job_number}",
            status=JobState.QUEUED,
            tenant=self._tenant,
        )


def fact_context(
    *facts: tuple[str, str],
    state: Literal["found", "none", "indexing", "degraded"] = "found",
) -> MemoryContext:
    """Build a memory context with one fact for each (fact id, statement) pair.

    Each fact's subject and object are entities named after their role, and
    ``rendered`` holds one ``[fact_id] statement`` line per fact.

    Args:
        *facts: Pairs of fact id and statement text, in the order to return.
        state: The state the context reports.

    Returns:
        A context whose facts belong to the tenant ``"tenant"``.
    """
    fact_outs = [
        FactOut(
            fact_id=fact_id,
            tenant="tenant",
            statement=statement,
            subject=EntityRef(entity_id=f"{fact_id}-subject", name="subject"),
            object=EntityRef(entity_id=f"{fact_id}-object", name="object"),
        )
        for fact_id, statement in facts
    ]
    rendered = "\n".join(f"[{fact_id}] {statement}" for fact_id, statement in facts)
    return MemoryContext(state=state, facts=fact_outs, rendered=rendered)
