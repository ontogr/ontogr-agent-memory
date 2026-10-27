"""Map temporal domain rows to and from graph properties."""

from collections.abc import Sequence
from datetime import datetime
from typing import Any

from ontogr_mem.data_models.fact_versions import (
    Fact,
    FactStatus,
    Supersession,
    SupersessionKind,
)
from ontogr_mem.data_models.facts import Precision, TimeBasis
from ontogr_mem.data_models.ledger import LedgerEntry, LedgerOp
from ontogr_mem.data_models.observation import (
    Observation,
    ObservationKind,
    ProvenanceStamp,
)
from ontogr_mem.data_models.timestamps import from_micros, to_micros
from ontogr_mem.graphstore.ident import quote
from ontogr_mem.temporal.classify import ConflictClass, Disposition, RelationToTime
from ontogr_mem.temporal.reconcile import Conflict, ConflictKind
from ontogr_mem.temporal.verdicts import (
    ConflictDecision,
    DecisionAction,
    FactVerdict,
    VerdictKind,
    VerdictSource,
)


_OBSERVATION_COLUMNS = (
    "id",
    "episode_id",
    "slot_key",
    "subject_id",
    "object_id",
    "relation_type_id",
    "kind",
    "occurred_at",
    "ingested_at",
    "valid_from",
    "valid_from_precision",
    "valid_to",
    "valid_to_precision",
    "time_basis",
    "provenance_extractor",
    "provenance_model",
    "provenance_prompt_version",
    "statement",
    "quote",
    "inverse",
    "source_system",
    "connector",
    "actor_role",
)
_FACT_COLUMNS = (
    "fact_id",
    "version",
    "slot_key",
    "relation_type_id",
    "valid_from",
    "valid_to",
    "valid_from_precision",
    "valid_to_precision",
    "recorded_at",
    "expired_at",
    "status",
    "evidence",
    "last_confirmed_at",
    "closed_by",
    "close_kind",
    "inverse",
)
_LEDGER_COLUMNS = (
    "id",
    "chain",
    "seq",
    "ts",
    "op",
    "subject_id",
    "actor",
    "payload_hash",
    "prev_checksum",
    "checksum",
)
_VERDICT_COLUMNS = (
    "id",
    "pair_key",
    "fact_a_id",
    "fact_b_id",
    "verdict",
    "inputs_hash",
    "prompt_version",
    "source",
    "confidence",
    "kind",
    "reason_code",
    "reason_text",
    "quote_a",
    "quote_b",
    "model",
    "votes",
    "decided_at",
    "source_episode_ids",
)
_DECISION_COLUMNS = (
    "id",
    "conflict_id",
    "fact_ids",
    "action",
    "decided_by",
    "at",
    "chosen_fact_id",
    "wrong_fact_id",
    "actor",
    "reason",
    "trust_inputs",
    "carried_from",
)
_CONFLICT_COLUMNS = (
    "id",
    "slot_key",
    "kind",
    "obs_ids",
    "episode_ids",
    "reason",
    "cls",
    "relation_to_time",
    "disposition",
    "fact_ids",
    "pair_key",
)
_SUPERSESSION_COLUMNS = (
    "id",
    "slot_key",
    "old_fact_id",
    "kind",
    "source_obs_id",
    "source_episode_id",
    "prior_valid_to",
    "applied_valid_to",
    "created_at",
)


def _columns(alias: str, names: Sequence[str]) -> str:
    variable = quote(alias, "variable")
    return ", ".join(
        f"{variable}.{quote(name, 'property key')} AS {quote(name, 'column')}"
        for name in names
    )


def _drop_none(props: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in props.items() if value is not None}


def _enum_value(value: Any) -> str | None:
    return None if value is None else str(value.value)


def _instant(value: Any) -> datetime | None:
    return None if value is None else from_micros(int(value))


def _required(value: Any) -> datetime:
    instant = _instant(value)
    if instant is None:
        msg = "a stored time is the open end where a value is required"
        raise ValueError(msg)
    return instant


def _precision(value: Any) -> Precision | None:
    return None if value is None else Precision(value)


def _observation_props(observation: Observation) -> dict[str, Any]:
    stamp = observation.provenance
    return _drop_none(
        {
            "id": observation.obs_id,
            "episode_id": observation.episode_id,
            "slot_key": observation.slot_key,
            "subject_id": observation.subject_id,
            "object_id": observation.object_id,
            "relation_type_id": observation.relation_type_id,
            "kind": observation.kind.value,
            "occurred_at": to_micros(observation.occurred_at),
            "ingested_at": to_micros(observation.ingested_at),
            "valid_from": to_micros(observation.valid_from),
            "valid_from_precision": _enum_value(observation.valid_from_precision),
            "valid_to": to_micros(observation.valid_to),
            "valid_to_precision": _enum_value(observation.valid_to_precision),
            "time_basis": observation.time_basis.value,
            "provenance_extractor": stamp.extractor,
            "provenance_model": stamp.model,
            "provenance_prompt_version": stamp.prompt_version,
            "statement": observation.statement,
            "quote": observation.quote,
            "inverse": observation.inverse,
            "source_system": observation.source_system,
            "connector": observation.connector,
            "actor_role": observation.actor_role,
        }
    )


def _observation_from(row: dict[str, Any]) -> Observation:
    return Observation(
        obs_id=row["id"],
        slot_key=row["slot_key"],
        subject_id=row["subject_id"],
        object_id=row["object_id"],
        relation_type_id=int(row["relation_type_id"]),
        kind=ObservationKind(row["kind"]),
        episode_id=row["episode_id"],
        occurred_at=_required(row["occurred_at"]),
        ingested_at=_required(row["ingested_at"]),
        valid_from=_instant(row["valid_from"]),
        valid_from_precision=_precision(row["valid_from_precision"]),
        valid_to=_instant(row["valid_to"]),
        valid_to_precision=_precision(row["valid_to_precision"]),
        time_basis=TimeBasis(row["time_basis"]),
        provenance=ProvenanceStamp(
            extractor=row["provenance_extractor"] or "",
            model=row["provenance_model"] or "",
            prompt_version=row["provenance_prompt_version"] or "",
        ),
        statement=row["statement"] or "",
        quote=row["quote"] or "",
        inverse=bool(row["inverse"]),
        source_system=row["source_system"] or "",
        connector=row["connector"] or "",
        actor_role=row["actor_role"] or "",
    )


def _fact_props(fact: Fact) -> dict[str, Any]:
    return _drop_none(
        {
            "fact_id": fact.fact_id,
            "version": fact.version,
            "slot_key": fact.slot_key,
            "relation_type_id": fact.relation_type_id,
            "valid_from": to_micros(fact.valid_from),
            "valid_to": to_micros(fact.valid_to),
            "valid_from_precision": _enum_value(fact.valid_from_precision),
            "valid_to_precision": _enum_value(fact.valid_to_precision),
            "recorded_at": to_micros(fact.recorded_at),
            "expired_at": to_micros(fact.expired_at),
            "status": fact.status.value,
            "evidence": list(fact.evidence),
            "last_confirmed_at": to_micros(fact.last_confirmed_at),
            "closed_by": fact.closed_by,
            "close_kind": _enum_value(fact.close_kind),
            "inverse": fact.inverse,
        }
    )


def _fact_from(row: dict[str, Any]) -> Fact:
    close_kind = row["close_kind"]
    return Fact(
        fact_id=row["fact_id"],
        version=int(row["version"]),
        slot_key=row["slot_key"],
        subject_id=row["subject_id"],
        object_id=row["object_id"],
        relation_type_id=int(row["relation_type_id"]),
        valid_from=_required(row["valid_from"]),
        valid_to=_instant(row["valid_to"]),
        valid_from_precision=_precision(row["valid_from_precision"]),
        valid_to_precision=_precision(row["valid_to_precision"]),
        recorded_at=_required(row["recorded_at"]),
        expired_at=_instant(row["expired_at"]),
        status=FactStatus(row["status"]),
        evidence=tuple(row["evidence"] or ()),
        last_confirmed_at=_required(row["last_confirmed_at"]),
        closed_by=row["closed_by"],
        close_kind=None if close_kind is None else SupersessionKind(close_kind),
        inverse=bool(row["inverse"]),
    )


def _ledger_props(entry: LedgerEntry) -> dict[str, Any]:
    return {
        "id": entry.id,
        "chain": entry.chain,
        "seq": entry.seq,
        "ts": to_micros(entry.ts),
        "op": entry.op.value,
        "subject_id": entry.subject_id,
        "actor": entry.actor,
        "payload_hash": entry.payload_hash,
        "prev_checksum": entry.prev_checksum,
        "checksum": entry.checksum,
    }


def _ledger_from(row: dict[str, Any]) -> LedgerEntry:
    return LedgerEntry(
        chain=row["chain"],
        seq=int(row["seq"]),
        ts=_required(row["ts"]),
        op=LedgerOp(row["op"]),
        subject_id=row["subject_id"],
        actor=row["actor"],
        payload_hash=row["payload_hash"],
        prev_checksum=row["prev_checksum"],
        checksum=row["checksum"],
    )


def _supersession_props(row: Supersession) -> dict[str, Any]:
    return {
        "id": row.supersession_id,
        "slot_key": row.slot_key,
        "old_fact_id": row.old_fact_id,
        "kind": row.kind.value,
        "source_obs_id": row.source_obs_id,
        "source_episode_id": row.source_episode_id,
        "prior_valid_to": to_micros(row.prior_valid_to),
        "applied_valid_to": to_micros(row.applied_valid_to),
        "created_at": to_micros(row.created_at),
    }


def _supersession_from(row: dict[str, Any]) -> Supersession:
    return Supersession(
        supersession_id=row["id"],
        slot_key=row["slot_key"],
        old_fact_id=row["old_fact_id"],
        kind=SupersessionKind(row["kind"]),
        source_obs_id=row["source_obs_id"],
        source_episode_id=row["source_episode_id"],
        prior_valid_to=_instant(row["prior_valid_to"]),
        applied_valid_to=_instant(row["applied_valid_to"]),
        created_at=_required(row["created_at"]),
    )


def _verdict_props(verdict: FactVerdict) -> dict[str, Any]:
    return _drop_none(
        {
            "id": verdict.verdict_id,
            "pair_key": verdict.pair_key,
            "fact_a_id": verdict.fact_a_id,
            "fact_b_id": verdict.fact_b_id,
            "verdict": verdict.verdict.value,
            "inputs_hash": verdict.inputs_hash,
            "prompt_version": verdict.prompt_version,
            "source": verdict.source.value,
            "confidence": verdict.confidence,
            "kind": verdict.kind,
            "reason_code": verdict.reason_code,
            "reason_text": verdict.reason_text,
            "quote_a": verdict.quote_a,
            "quote_b": verdict.quote_b,
            "model": verdict.model,
            "votes": [f"{name}:{answer}" for name, answer in verdict.votes],
            "decided_at": to_micros(verdict.decided_at),
            "source_episode_ids": list(verdict.source_episode_ids),
        }
    )


def _verdict_from(row: dict[str, Any]) -> FactVerdict:
    return FactVerdict(
        pair_key=row["pair_key"],
        fact_a_id=row["fact_a_id"],
        fact_b_id=row["fact_b_id"],
        verdict=VerdictKind(row["verdict"]),
        inputs_hash=row["inputs_hash"],
        prompt_version=row["prompt_version"],
        source=VerdictSource(row["source"]),
        confidence=float(row["confidence"]),
        kind=row["kind"] or "",
        reason_code=row["reason_code"] or "",
        reason_text=row["reason_text"] or "",
        quote_a=row["quote_a"] or "",
        quote_b=row["quote_b"] or "",
        model=row["model"] or "",
        votes=tuple(tuple(item.split(":", 1)) for item in row["votes"] or ()),
        decided_at=_instant(row["decided_at"]),
        source_episode_ids=tuple(row["source_episode_ids"] or ()),
    )


def _decision_props(decision: ConflictDecision) -> dict[str, Any]:
    return _drop_none(
        {
            "id": decision.decision_id,
            "conflict_id": decision.conflict_id,
            "fact_ids": list(decision.fact_ids),
            "action": decision.action.value,
            "decided_by": decision.decided_by.value,
            "at": to_micros(decision.at),
            "chosen_fact_id": decision.chosen_fact_id,
            "wrong_fact_id": decision.wrong_fact_id,
            "actor": decision.actor,
            "reason": decision.reason,
            "trust_inputs": [f"{fact}:{rank}" for fact, rank in decision.trust_inputs],
            "carried_from": decision.carried_from,
        }
    )


def _decision_from(row: dict[str, Any]) -> ConflictDecision:
    return ConflictDecision(
        decision_id=row["id"],
        conflict_id=row["conflict_id"],
        fact_ids=tuple(row["fact_ids"] or ()),
        action=DecisionAction(row["action"]),
        decided_by=VerdictSource(row["decided_by"]),
        at=_required(row["at"]),
        chosen_fact_id=row["chosen_fact_id"] or "",
        wrong_fact_id=row["wrong_fact_id"] or "",
        actor=row["actor"] or "",
        reason=row["reason"] or "",
        trust_inputs=tuple(
            (fact, int(rank))
            for fact, rank in (
                item.rsplit(":", 1) for item in row["trust_inputs"] or ()
            )
        ),
        carried_from=row["carried_from"] or "",
    )


def _conflict_props(conflict: Conflict) -> dict[str, Any]:
    return _drop_none(
        {
            "id": conflict.conflict_id,
            "slot_key": conflict.slot_key,
            "kind": conflict.kind.value,
            "obs_ids": list(conflict.obs_ids),
            "episode_ids": list(conflict.episode_ids),
            "reason": conflict.reason,
            "cls": _enum_value(conflict.cls),
            "relation_to_time": _enum_value(conflict.relation_to_time),
            "disposition": _enum_value(conflict.disposition),
            "fact_ids": list(conflict.fact_ids),
            "pair_key": conflict.pair_key,
        }
    )


def _conflict_from(row: dict[str, Any]) -> Conflict:
    return Conflict(
        conflict_id=row["id"],
        slot_key=row["slot_key"],
        kind=ConflictKind(row["kind"]),
        obs_ids=tuple(row["obs_ids"] or ()),
        episode_ids=tuple(row["episode_ids"] or ()),
        reason=row["reason"] or "",
        cls=None if row["cls"] is None else ConflictClass(row["cls"]),
        relation_to_time=(
            None
            if row["relation_to_time"] is None
            else RelationToTime(row["relation_to_time"])
        ),
        disposition=(
            None if row["disposition"] is None else Disposition(row["disposition"])
        ),
        fact_ids=tuple(row["fact_ids"] or ()),
        pair_key=row["pair_key"],
    )
