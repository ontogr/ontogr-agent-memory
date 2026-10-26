"""Confirmation tokens bind a caller to one forget plan and expiry."""

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from ontogr_mem.data_models.clock import Clock
from ontogr_mem.service.errors import ConfirmationRequiredError
from ontogr_mem.service.forget.confirm import ConfirmTokens, target_digest
from ontogr_mem.service.forget.settings import ForgetSettings


class FixedClock:
    """Clock whose value can be advanced by a token test."""

    def __init__(self, now: datetime) -> None:
        """Set the clock value."""
        self.current = now

    def now(self) -> datetime:
        """Return the current test time."""
        return self.current


NOW = datetime(2026, 10, 23, 12, 0, tzinfo=UTC)


class TestConfirmTokens:
    """Tokens are short-lived and context-bound HMACs."""

    @pytest.fixture
    def clock(self) -> FixedClock:
        """Return a fixed clock that the test can move."""
        return FixedClock(NOW)

    def test_fresh_token_verifies_for_its_context(self, clock: Clock):
        """A newly issued token verifies for the same request context."""
        tokens = ConfirmTokens(b"secret-with-at-least-16-bytes", clock)
        token = tokens.issue("caller", "tenant", "target", "erase")

        tokens.verify(token, "caller", "tenant", "target", "erase")

    @pytest.mark.parametrize(
        "context",
        [
            ("other-caller", "tenant", "target", "erase"),
            ("caller", "other-tenant", "target", "erase"),
            ("caller", "tenant", "other-target", "erase"),
            ("caller", "tenant", "target", "delete"),
        ],
    )
    def test_rejects_a_different_context(self, clock: Clock, context):
        """Changing any bound field invalidates the token."""
        tokens = ConfirmTokens(b"secret-with-at-least-16-bytes", clock)
        token = tokens.issue("caller", "tenant", "target", "erase")

        with pytest.raises(ConfirmationRequiredError):
            tokens.verify(token, *context)

    def test_token_is_valid_until_its_expiry_second(self, clock: FixedClock):
        """Expiry is exclusive: the token is invalid at its encoded second."""
        tokens = ConfirmTokens(b"secret-with-at-least-16-bytes", clock)
        token = tokens.issue("caller", "tenant", "target", "erase")
        expires_at = tokens.expiry(token)

        clock.current = datetime.fromtimestamp(expires_at - 1, UTC)
        tokens.verify(token, "caller", "tenant", "target", "erase")
        clock.current = datetime.fromtimestamp(expires_at, UTC)
        with pytest.raises(ConfirmationRequiredError):
            tokens.verify(token, "caller", "tenant", "target", "erase")

    @pytest.mark.parametrize("token", ["no-dot", "x.mac", "1.", "1.invalid"])
    def test_rejects_malformed_or_tampered_tokens(self, clock: Clock, token: str):
        """Malformed expiry fields and MAC values fail closed."""
        tokens = ConfirmTokens(b"secret-with-at-least-16-bytes", clock)

        with pytest.raises(ConfirmationRequiredError):
            tokens.verify(token, "caller", "tenant", "target", "erase")

    def test_settings_reject_a_short_configured_secret(self):
        """Configured HMAC secrets must contain at least 16 UTF-8 bytes."""
        with pytest.raises(ValidationError):
            ForgetSettings(confirm_secret="x")

    def test_digest_is_independent_of_episode_order(self):
        """The same resolved episode set has the same target digest."""
        assert target_digest("episode_ids", ["e2", "e1"]) == target_digest(
            "episode_ids", ["e1", "e2"]
        )

    def test_caller_ids_with_separators_do_not_alias_other_contexts(self, clock):
        """The request encoding stays unambiguous for unrestricted principal ids."""
        tokens = ConfirmTokens(b"secret-with-at-least-16-bytes", clock)
        token = tokens.issue("a|b", "c", "target", "erase")

        with pytest.raises(ConfirmationRequiredError):
            tokens.verify(token, "a", "b|c", "target", "erase")

    def test_expiry_is_ten_minutes_after_issue(self):
        """The fixed confirmation lifetime is ten minutes."""
        tokens = ConfirmTokens(b"secret-with-at-least-16-bytes", FixedClock(NOW))
        token = tokens.issue("caller", "tenant", "target", "erase")

        assert tokens.expiry(token) == int((NOW + timedelta(minutes=10)).timestamp())
