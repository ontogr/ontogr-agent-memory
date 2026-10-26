"""Fuzzy blocking: entropy floor, MinHash with LSH bands, exact Jaccard.

Hashing is blake2b with fixed seeds, so results do not depend on
``PYTHONHASHSEED``. Candidates come back sorted, never in set order.
"""

import hashlib
import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass


NAME_ENTROPY_THRESHOLD = 1.5
MIN_NAME_LENGTH = 6
MIN_TOKEN_COUNT = 2
MINHASH_PERMUTATIONS = 32
MINHASH_BAND_SIZE = 4
STRIPPED_SUFFIX = "#s"
"""Item id suffix of an entity indexed under its suffix-stripped key."""


def name_entropy(text: str) -> float:
    """Return the Shannon entropy over the characters of ``text``.

    Args:
        text: Name to measure. Spaces are removed first.

    Returns:
        Entropy in bits per character, or zero for an empty name.
    """
    chars = text.replace(" ", "")
    if not chars:
        return 0.0
    total = len(chars)
    return -sum((n / total) * math.log2(n / total) for n in Counter(chars).values())


def has_high_entropy(name: str) -> bool:
    """Return whether a name may take part in fuzzy matching.

    Short single-token names and names below the entropy floor are exact-only.

    Args:
        name: Name to check.

    Returns:
        True when the name clears both the length and entropy floors.
    """
    if len(name) < MIN_NAME_LENGTH and len(name.split()) < MIN_TOKEN_COUNT:
        return False
    return name_entropy(name) >= NAME_ENTROPY_THRESHOLD


def shingles(text: str) -> frozenset[str]:
    """Return the character 3-grams of ``text`` with spaces removed."""
    t = text.replace(" ", "")
    if len(t) < 2:
        return frozenset({t})
    return frozenset(t[i : i + 3] for i in range(max(1, len(t) - 2)))


def _h(seed: int, shingle: str) -> int:
    digest = hashlib.blake2b(f"{seed}:{shingle}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def minhash(sh: frozenset[str]) -> tuple[int, ...]:
    """Return the MinHash signature of a shingle set."""
    if not sh:
        return (0,) * MINHASH_PERMUTATIONS
    return tuple(min(_h(s, x) for x in sh) for s in range(MINHASH_PERMUTATIONS))


def bands(sig: tuple[int, ...]) -> list[tuple[int, tuple[int, ...]]]:
    """Return the LSH bucket keys of a signature."""
    return [
        (i // MINHASH_BAND_SIZE, sig[i : i + MINHASH_BAND_SIZE])
        for i in range(0, len(sig), MINHASH_BAND_SIZE)
    ]


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """Return the exact Jaccard similarity of two shingle sets."""
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


@dataclass(frozen=True, slots=True)
class BlockingRow:
    """What the blocking index needs to know about one active entity.

    Attributes:
        entity_id: Id of the entity.
        loose_key: Blocking key of the canonical name.
        stripped_key: Suffix-stripped key of the canonical name.
        alias_keys: Blocking keys of the entity's aliases.
    """

    entity_id: str
    loose_key: str
    stripped_key: str
    alias_keys: tuple[str, ...] = ()


class BlockIndex:
    """In-memory LSH index over the blocking keys of one tenant."""

    def __init__(self) -> None:
        """Create an empty index."""
        self._buckets: dict[tuple[int, tuple[int, ...]], set[str]] = defaultdict(set)
        self._shingles: dict[str, frozenset[str]] = {}
        self._bands: dict[str, list[tuple[int, tuple[int, ...]]]] = {}

    @classmethod
    def from_rows(cls, rows: Iterable[BlockingRow]) -> "BlockIndex":
        """Build an index from the store's blocking rows.

        An entity is indexed under its loose key, under one item per alias and,
        when the suffix-stripped key differs, under ``"<id>#s"``.

        Args:
            rows: Rows of the active entities.

        Returns:
            The filled index.
        """
        index = cls()
        for row in rows:
            index.add(row.entity_id, row.loose_key)
            for n, key in enumerate(row.alias_keys):
                index.add(f"{row.entity_id}#{n}", key)
            if row.stripped_key and row.stripped_key != row.loose_key:
                index.add(f"{row.entity_id}{STRIPPED_SUFFIX}", row.stripped_key)
        return index

    def __len__(self) -> int:
        """Return the number of indexed items."""
        return len(self._shingles)

    def add(self, item_id: str, key: str) -> None:
        """Index ``item_id`` under ``key``; re-adding replaces the old key."""
        self.remove(item_id)
        sh = shingles(key)
        bs = bands(minhash(sh))
        self._shingles[item_id] = sh
        self._bands[item_id] = bs
        for b in bs:
            self._buckets[b].add(item_id)

    def remove(self, item_id: str) -> None:
        """Drop ``item_id`` if present."""
        for b in self._bands.pop(item_id, []):
            self._buckets[b].discard(item_id)
        self._shingles.pop(item_id, None)

    def candidates(self, key: str) -> list[str]:
        """Return the sorted item ids sharing at least one band with ``key``."""
        found: set[str] = set()
        for b in bands(minhash(shingles(key))):
            found |= self._buckets.get(b, set())
        return sorted(found)

    def similarity(self, item_id: str, key: str) -> float:
        """Return the exact Jaccard between ``key`` and the item's key."""
        sh = self._shingles.get(item_id)
        return 0.0 if sh is None else jaccard(shingles(key), sh)
