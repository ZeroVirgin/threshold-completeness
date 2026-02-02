# tc/augment.py
from __future__ import annotations
from typing import List, Set, Tuple

def build_A_set(A: List[int]) -> Set[int]:
    """Hash set of A for O(1) membership."""
    return set(A)

def uncovered_list_from_coverage(B, start: int = 2) -> List[int]:
    """Extract uncovered indices from coverage bitset B, starting at `start`."""
    start = max(1, start)
    return [k for k in range(start, len(B)) if not B[k]]

def greedy_augment_to_cover(
    n: int,
    A: List[int],
    uncovered: List[int],
    halo: List[int],
    max_add: int | None = None,
    start: int = 2,
) -> Tuple[List[int], List[int]]:
    """
    Greedy augmentation:
      - A is the current y-smooth set (list).
      - uncovered are targets k not in A+A.
      - halo are candidate extra elements (e.g., y' -smooth with y'>y) we are allowed to ADD.
      - We choose candidates that cover the most still-uncovered k (i.e., for many k, k - a in A_set).

    Returns (added, remaining_uncovered).
    """
    A_set = build_A_set(A)
    remaining = list(uncovered)
    added: List[int] = []
    if not remaining or not halo:
        return added, remaining

    # Precompute for speed: map candidate a -> indices of k it can cover
    cover_map: List[Tuple[int, List[int]]] = []  # (a, k_list)
    rem_set = set(remaining)
    for a in halo:
        hits = []
        for k in remaining:
            b = k - a
            if 1 <= b <= n and b in A_set:
                hits.append(k)
        if hits:
            cover_map.append((a, hits))

    # Greedy loop: pick candidate covering the most remaining k at each step
    # Simple O(|halo| * |uncovered|) works fine for your scales
    picks = 0
    while rem_set:
        best_idx = -1
        best_gain = 0
        # recompute effective gains w.r.t. current rem_set
        for i, (a, hits) in enumerate(cover_map):
            gain = sum(1 for k in hits if k in rem_set)
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        if best_idx == -1 or best_gain == 0:
            break  # no candidate helps further

        a, hits = cover_map[best_idx]
        added.append(a)
        picks += 1
        # remove covered k from rem_set
        for k in hits:
            if k in rem_set:
                rem_set.remove(k)

        # Optional stop condition
        if max_add is not None and picks >= max_add:
            break

    return added, sorted(rem_set)
