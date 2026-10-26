"""Request context state stays local to its bound async task."""

import asyncio

import pytest

from ontogr_mem.data_models.tenant import InvalidTenantError
from ontogr_mem.request_context import (
    RequestContext,
    Usage,
    bind_context,
    current_context,
    record_usage,
)


class TestRequestContext:
    """Request context validation, scoping and usage accounting."""

    def test_remaining_seconds_is_none_without_a_budget(self):
        """An unbudgeted request has no deadline to report."""
        context = RequestContext(principal_id="owner", tenant="acme")

        assert context.remaining_seconds() is None

    def test_zero_budget_is_already_spent(self):
        """A zero budget cannot report a negative remainder."""
        context = RequestContext(
            principal_id="owner", tenant="acme", latency_budget_s=0.0
        )

        assert context.remaining_seconds() == 0.0

    def test_record_usage_without_a_bound_context_is_a_noop(self):
        """Usage outside a request does not create implicit context state."""
        record_usage(3, 1, 0.2)

        assert current_context() is None

    def test_priced_calls_accumulate_tokens_and_cost(self):
        """Priced calls add their token counts and costs to one snapshot."""
        usage = Usage()
        usage.add(4, 2, 0.1)
        usage.add(3, 1, 0.2)

        snapshot = usage.snapshot()

        assert (snapshot.tokens_in, snapshot.tokens_out) == (7, 3)
        assert snapshot.cost_usd == pytest.approx(0.3)

    def test_unknown_cost_makes_the_snapshot_cost_unknown(self):
        """An unpriced call does not make its cost look like zero."""
        usage = Usage()
        usage.add(4, 2, 0.1)
        usage.add(3, 1, None)

        snapshot = usage.snapshot()

        assert (snapshot.tokens_in, snapshot.tokens_out) == (7, 3)
        assert snapshot.cost_usd is None

    def test_nested_bind_restores_the_outer_context(self):
        """Exiting a nested bind restores the previous request context."""
        outer = RequestContext(principal_id="outer", tenant="acme")
        inner = RequestContext(principal_id="inner", tenant="beta")

        with bind_context(outer):
            assert current_context() is outer
            with bind_context(inner):
                assert current_context() is inner
            assert current_context() is outer

        assert current_context() is None

    async def test_concurrent_tasks_keep_separate_contexts_and_usage(self):
        """Concurrent tasks update only the context each task binds."""
        contexts = [
            RequestContext(principal_id="first", tenant="acme"),
            RequestContext(principal_id="second", tenant="beta"),
        ]

        async def use_context(context: RequestContext, tokens: int) -> RequestContext:
            with bind_context(context):
                record_usage(tokens, 1, 0.5)
                await asyncio.sleep(0)
                assert current_context() is context
                return context

        results = await asyncio.gather(
            use_context(contexts[0], 5), use_context(contexts[1], 9)
        )

        assert results == contexts
        assert [context.usage.snapshot().tokens_in for context in contexts] == [5, 9]
        assert [context.usage.snapshot().tokens_out for context in contexts] == [1, 1]

    @pytest.mark.parametrize("tenant", ["bad id", ""])
    def test_invalid_tenant_is_rejected(self, tenant):
        """Request contexts accept only valid Tenant ids when one is set."""
        with pytest.raises(InvalidTenantError):
            RequestContext(principal_id="owner", tenant=tenant)

    def test_empty_principal_id_is_rejected(self):
        """A request context requires an identifiable principal."""
        with pytest.raises(ValueError, match="principal"):
            RequestContext(principal_id="", tenant="acme")
