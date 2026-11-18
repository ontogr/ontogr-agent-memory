"""Candidate window widening behavior."""

from ontogr_mem.retrieval.lanes._shared import widen


class TestWiden:
    """Verify candidate windows widen until enough results survive."""

    async def test_doubles_until_enough_survivors(self):
        """Double the window until it returns the requested survivors."""
        windows = []

        async def run(window):
            windows.append(window)
            return window

        result = await widen(
            run,
            lambda value: max(0, value - 3),
            lambda value: value < 8,
            start=2,
            want=5,
            rechecks=3,
        )

        assert result == 8
        assert windows == [2, 4, 8]

    async def test_stops_at_recheck_limit(self):
        """Stop widening after the configured number of rechecks."""
        windows = []

        async def run(window):
            windows.append(window)
            return window

        await widen(run, lambda _: 0, lambda _: True, start=2, want=1, rechecks=2)

        assert windows == [2, 4, 8]
