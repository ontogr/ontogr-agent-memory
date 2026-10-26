"""Bottom-up bucket hierarchy over community summaries.

Items are placed into buckets of at most ``max_bucket_size`` in id order. While a
level has more buckets than fit under one parent, another level is built above it.
The last level is a single root. Placement by id keeps the result deterministic.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BucketPlan:
    """A planned bucket: its level, ordinal at that level and the child ids.

    Attributes:
        level: The level of the bucket. Level 1 holds the items themselves.
        index: The ordinal of the bucket within its level, starting at zero.
        children: The ids of the items or lower-level buckets in this bucket.
    """

    level: int
    index: int
    children: tuple[str, ...]


def plan_buckets(item_ids: list[str], max_bucket_size: int) -> list[BucketPlan]:
    """Plan every bucket level above the items.

    Args:
        item_ids: Ids of the items to place. Their order does not matter.
        max_bucket_size: Largest number of children in one bucket.

    Returns:
        The planned buckets, level by level. The last level holds one root bucket.
        Fewer than two items return an empty list.

    Raises:
        ValueError: If ``max_bucket_size`` is less than two.
    """
    if max_bucket_size < 2:
        raise ValueError("max_bucket_size must be at least 2")
    if len(item_ids) < 2:
        return []
    plans: list[BucketPlan] = []
    current = sorted(item_ids)
    level = 1
    while True:
        chunks = [
            tuple(current[i : i + max_bucket_size])
            for i in range(0, len(current), max_bucket_size)
        ]
        plans.extend(BucketPlan(level, i, c) for i, c in enumerate(chunks))
        if len(chunks) == 1:
            return plans
        current = [bucket_id(level, i) for i in range(len(chunks))]
        level += 1


def bucket_id(level: int, index: int) -> str:
    """Return the id of the bucket at ``index`` within ``level``.

    Args:
        level: The bucket level, starting at one.
        index: The ordinal of the bucket within its level, starting at zero.

    Returns:
        An id of the form ``bucket-<level>-<index>``.
    """
    return f"bucket-{level}-{index}"
