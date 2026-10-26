"""Ledger checksums keep field boundaries and detect tampering."""

from dataclasses import replace
from datetime import UTC, datetime

from ontogr_mem.data_models.ledger import LedgerOp
from ontogr_mem.temporal.ledger import GENESIS, link, verify


TS = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
CHAIN = "slot-1"
PAYLOAD = {"version": 1, "status": "active"}


class TestLedgerChecksum:
    """Checksums commit to each field with its boundaries intact."""

    def test_moving_a_pipe_between_fields_changes_the_checksum(self):
        """Entries whose fields differ only by where a pipe sits do not collide."""
        first = link(CHAIN, (0, GENESIS), TS, LedgerOp.FACT_ADDED, "a|b", "c", PAYLOAD)
        second = link(CHAIN, (0, GENESIS), TS, LedgerOp.FACT_ADDED, "a", "b|c", PAYLOAD)
        assert first.checksum != second.checksum

    def test_verify_accepts_a_chain_built_by_link(self):
        """Entries written by link pass verification in any order."""
        first = link(
            CHAIN, (0, GENESIS), TS, LedgerOp.FACT_ADDED, "fact-1", "engine", PAYLOAD
        )
        second = link(
            CHAIN,
            (first.seq, first.checksum),
            TS,
            LedgerOp.FACT_VERSIONED,
            "fact-1",
            "engine",
            PAYLOAD,
        )
        assert verify([second, first]) == []

    def test_verify_rejects_a_chain_with_a_tampered_field(self):
        """Changing a stored actor breaks that entry's checksum."""
        first = link(
            CHAIN, (0, GENESIS), TS, LedgerOp.FACT_ADDED, "fact-1", "engine", PAYLOAD
        )
        second = link(
            CHAIN,
            (first.seq, first.checksum),
            TS,
            LedgerOp.FACT_VERSIONED,
            "fact-1",
            "engine",
            PAYLOAD,
        )
        forged = replace(first, actor="intruder")
        problems = verify([forged, second])
        assert f"{CHAIN}#1: checksum does not match" in problems
