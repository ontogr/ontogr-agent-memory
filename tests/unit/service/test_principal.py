"""Tests for the grant enum, the principal model and its three rules."""

import pytest
from pydantic import ValidationError

from ontogr_mem.service.contracts import ErrorInfo
from ontogr_mem.service.errors import ObjectNotFoundError
from ontogr_mem.service.principal import Grant, Principal


READ = Grant.READ
WRITE = Grant.WRITE
FORGET = Grant.FORGET
SHARE = Grant.SHARE
EXPORT = Grant.EXPORT
ADMIN = Grant.ADMIN


class TestHolds:
    """When a principal holds one grant on one tenant."""

    @pytest.mark.parametrize(
        ("grants", "tenant", "grant", "expected"),
        [
            ({"t1": {READ}}, "t1", READ, True),
            ({"t1": {READ}}, "t1", WRITE, False),
            ({"t1": {READ}}, "t2", READ, False),
            ({"t1": {ADMIN}}, "t1", EXPORT, True),
            ({"*": {WRITE}}, "t9", WRITE, True),
            ({"*": {ADMIN}}, "t9", FORGET, True),
            ({"t1": set()}, "t1", READ, False),
        ],
    )
    def test_holds_follows_the_grant_map(self, grants, tenant, grant, expected):
        """A grant holds for its tenant, its wildcard or admin over it."""
        principal = _principal(grants)

        assert principal.holds(tenant, grant) is expected

    def test_admin_on_one_tenant_implies_every_grant_on_it(self):
        """Admin covers all six grants on its own tenant only."""
        principal = _principal({"t1": {ADMIN}})

        assert all(principal.holds("t1", grant) for grant in Grant)
        assert not principal.holds("t2", READ)


class TestRequire:
    """The same error for a missing tenant and for a missing grant."""

    def test_no_grant_and_other_grant_give_the_same_error(self):
        """A caller cannot tell an absent tenant from a missing grant."""
        blind = _principal({"other": {READ}})
        read_only = _principal({"t1": {READ}})

        with pytest.raises(ObjectNotFoundError) as blind_info:
            blind.require("t1", READ)
        with pytest.raises(ObjectNotFoundError) as read_only_info:
            read_only.require("t1", WRITE)

        assert ErrorInfo.from_error(blind_info.value) == ErrorInfo.from_error(
            read_only_info.value
        )

    def test_a_tenant_that_does_not_exist_gives_the_same_error(self):
        """An unknown tenant id is indistinguishable from an ungranted one."""
        principal = _principal({"t1": {READ}})

        with pytest.raises(ObjectNotFoundError) as blind_info:
            _principal({"other": {READ}}).require("t1", READ)
        with pytest.raises(ObjectNotFoundError) as excinfo:
            principal.require("missing", READ)

        assert ErrorInfo.from_error(excinfo.value) == ErrorInfo.from_error(
            blind_info.value
        )

    def test_the_message_does_not_echo_the_tenant_id(self):
        """The refusal cannot be used to probe tenant ids."""
        principal = _principal({"t1": {READ}})

        with pytest.raises(ObjectNotFoundError) as excinfo:
            principal.require("secret-tenant", READ)

        assert "secret-tenant" not in excinfo.value.message
        assert excinfo.value.hint == (
            "Check the tenant id and the grants on this principal."
        )

    def test_a_held_grant_passes_and_returns_none(self):
        """The check is silent when the grant is held."""
        principal = _principal({"t1": {READ}})

        assert principal.require("t1", READ) is None


class TestIsWithin:
    """The subset rule an agent's grants must satisfy at creation."""

    @pytest.mark.parametrize(
        ("parent_grants", "child_grants", "expected"),
        [
            ({"t1": {ADMIN}}, {"t1": {READ, WRITE, FORGET, SHARE, EXPORT}}, True),
            ({"*": {READ}}, {"t1": {READ}}, True),
            ({"t1": {READ}}, {"*": {READ}}, False),
            ({"*": {ADMIN}}, {"*": {EXPORT}}, True),
            ({"t1": {READ, WRITE, FORGET, SHARE, EXPORT}}, {"t1": {ADMIN}}, False),
            ({"t1": {READ}}, {"t2": {READ}}, False),
            ({"t1": {READ}}, {}, True),
        ],
    )
    def test_child_grants_must_stay_within_the_parent(
        self, parent_grants, child_grants, expected
    ):
        """Every child grant must hold for the parent, or the child is wider."""
        child = _principal(child_grants)
        parent = _principal(parent_grants)

        assert child.is_within(parent) is expected


class TestPrincipalModel:
    """Validation, immutability and the wire form of a principal."""

    @pytest.mark.parametrize("key", ["bad key!", "", "a" * 61])
    def test_a_bad_grant_key_fails_validation(self, key):
        """A grant key is the wildcard or a valid tenant id."""
        with pytest.raises(ValidationError):
            _principal({key: {READ}})

    def test_the_wildcard_key_is_allowed(self):
        """The wildcard is the one key outside the tenant id format."""
        principal = _principal({"*": {READ}})

        assert principal.holds("t9", READ)

    def test_parent_id_marks_an_agent(self):
        """An agent principal names the principal that created it."""
        principal = Principal(id="agent-1", parent_id="root")

        assert principal.parent_id == "root"

    def test_a_principal_survives_a_json_round_trip(self):
        """The wire form keeps the wildcard, the tenant and the parent."""
        principal = Principal(
            id="agent-1",
            grants={"*": frozenset({READ}), "t1": frozenset({WRITE})},
            default_tenant="t1",
            parent_id="root",
        )

        assert Principal.model_validate_json(principal.model_dump_json()) == principal

    def test_a_principal_is_frozen(self):
        """Assigning a field after creation fails."""
        principal = _principal({"t1": {READ}})

        with pytest.raises(ValidationError):
            principal.id = "other"


def _principal(grants):
    """Build a principal over a plain grant map for the tables above."""
    return Principal(
        id="caller",
        grants={key: frozenset(value) for key, value in grants.items()},
    )
