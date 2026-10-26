"""Stable identities for verdicts about entity pairs."""

import hashlib
import json

from ontogr_mem.data_models.entities import Entity


def inputs_hash(a: Entity, b: Entity) -> str:
    """Return a hash of the data a verdict about ``(a, b)`` was based on."""
    first, second = sorted((a, b), key=lambda entity: entity.entity_id)
    payload = [
        [
            entity.entity_id,
            entity.canonical_name,
            entity.type_label,
            entity.summary,
            sorted(entity.identity_attrs.items()),
            entity.uri,
        ]
        for entity in (first, second)
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def verdict_id(a_id: str, b_id: str) -> tuple[str, str, str]:
    """Return the ordered entity pair and its deterministic verdict id."""
    lo, hi = sorted((a_id, b_id))
    return lo, hi, f"{lo}|{hi}"
