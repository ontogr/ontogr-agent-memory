"""Ledger tests: canonical payload hashes and a tamper-evident chain."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone

from ontogr_mem.data_models.ledger import LedgerOp
from ontogr_mem.temporal.ledger import (
    GENESIS,
    checksum,
    link,
    payload_hash,
    verify,
)


TS = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
PAYLOAD = {"version": 1, "status": "active"}
PAYLOAD_HASH = "e5b6f7480b90285f20e1f99fd00000d8030c88de71cc0de36885b7c46fa768b8"
CHECKSUM = "266fa516d45769eb75669e9dac2fa8bce1c6e17a38571e32fd25bdac257fd522"
CHAIN = "slot-1"


def first_entry():
    """Build the first entry of a chain."""
    return link(
        CHAIN,
        (0, GENESIS),
        TS,
        LedgerOp.FACT_ADDED,
        "fact-1",
        "engine",
        PAYLOAD,
    )


class TestLedger:
    """Behavior of the pure hash-chain functions."""

    def test_payload_hash_is_canonical_json(self):
        """Key order does not change the payload hash."""
        assert payload_hash(PAYLOAD) == PAYLOAD_HASH
        assert payload_hash({"status": "active", "version": 1}) == PAYLOAD_HASH
        assert payload_hash({"version": 1}) != PAYLOAD_HASH

    def test_first_entry_links_to_genesis(self):
        """A first entry has sequence one and a pinned checksum."""
        entry = first_entry()
        assert entry.id == "slot-1:1"
        assert entry.prev_checksum == GENESIS
        assert entry.payload_hash == PAYLOAD_HASH
        assert entry.checksum == CHECKSUM
        assert verify([entry]) == []

    def test_an_intact_chain_verifies_in_any_order(self):
        """Two linked entries pass verification regardless of input order."""
        first = first_entry()
        second = link(
            CHAIN,
            (first.seq, first.checksum),
            TS + timedelta(hours=1),
            LedgerOp.FACT_VERSIONED,
            "fact-1",
            "engine",
            {"version": 2, "status": "active"},
        )
        assert second.seq == 2
        assert second.prev_checksum == first.checksum
        assert verify([first, second]) == []
        assert verify([second, first]) == []

    def test_a_tampered_entry_is_reported(self):
        """Changing a stored field breaks that entry's checksum."""
        first = first_entry()
        second = link(
            CHAIN,
            (first.seq, first.checksum),
            TS + timedelta(hours=1),
            LedgerOp.FACT_VERSIONED,
            "fact-1",
            "engine",
            {"version": 2, "status": "active"},
        )
        forged = replace(first, subject_id="forged")
        assert any("checksum" in problem for problem in verify([forged, second]))
        assert any("seq" in problem for problem in verify([second]))

    def test_checksum_depends_on_the_instant_not_the_timezone(self):
        """The hashed form is integer microseconds, so zones do not matter."""
        same_instant = datetime(
            2026,
            1,
            1,
            14,
            0,
            tzinfo=timezone(timedelta(hours=2)),
        )
        assert (
            checksum(
                1,
                same_instant,
                LedgerOp.FACT_ADDED,
                "fact-1",
                "engine",
                PAYLOAD_HASH,
                GENESIS,
            )
            == CHECKSUM
        )
