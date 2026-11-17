"""Pure helpers for length-bucketed batched seq2seq generation (no torch import)."""

from __future__ import annotations

from typing import Sequence


def plan_batches(lengths: Sequence[int], *, max_batch: int, square_budget: int) -> list[list[int]]:
    """Group request indices into batches, longest prompts first.

    Eager encoder attention costs O(batch * length**2) memory, so the batch size for a
    group is bounded by ``square_budget // longest_length**2`` (never below one) and by
    ``max_batch``. Every index appears in exactly one batch.
    """
    if max_batch < 1 or square_budget < 1:
        raise ValueError("max_batch and square_budget must be positive")
    if any(type(n) is not int or n < 1 for n in lengths):
        raise ValueError("prompt lengths must be positive integers")
    order = sorted(range(len(lengths)), key=lambda i: (-lengths[i], i))
    batches, position = [], 0
    while position < len(order):
        longest = lengths[order[position]]
        size = max(1, min(max_batch, square_budget // (longest * longest)))
        batches.append(order[position:position + size])
        position += size
    return batches


def split_generated(row: Sequence[int], eos_ids: Sequence[int]) -> tuple[list[int], bool]:
    """Cut one generated row (decoder start already removed) at its first end token.

    Returns ``(tokens, finished)``. A finished row keeps the end token, matching what
    unbatched generation returns; batch padding after it is dropped. A row with no end
    token is returned whole and reported unfinished.
    """
    ends = set(eos_ids)
    for index, token in enumerate(row):
        if token in ends:
            return list(row[:index + 1]), True
    return list(row), False
