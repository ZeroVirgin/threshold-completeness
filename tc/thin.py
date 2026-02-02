from __future__ import annotations
from typing import Iterable, List, Dict, Tuple
import random
from collections import defaultdict


def _residue_counts(A: List[int], q: int) -> Dict[int, int]:
    d: Dict[int, int] = defaultdict(int)
    for a in A:
        d[a % q] += 1
    return d


def residue_balanced_thin(
    A: List[int],
    qmax_thin: int = 64,
    keep_ratio: float = 1.0,
    seed: int | None = 12345,
) -> List[int]:
    """
    Residue-balanced thinning:
    - For each 2 <= q <= qmax_thin, we want residues to be roughly uniform.
    - Compute per-element weight as the minimum (over q) of (target_count / current_count_of_its_residue).
    - Then keep each element independently with probability lambda * weight, with lambda tuned
      so expected retained size ~= keep_ratio * |A|.

    Returns a new list A' (subset of A).
    """
    if seed is not None:
        random.seed(seed)

    if not A:
        return []

    A_sorted = list(A)
    A_sorted.sort()
    n = len(A_sorted)

    counts_by_q: Dict[int, Dict[int, int]] = {}
    for q in range(2, qmax_thin + 1):
        counts_by_q[q] = _residue_counts(A_sorted, q)

    weights: List[float] = []
    for a in A_sorted:
        w = 1.0
        for q in range(2, qmax_thin + 1):
            cnt = counts_by_q[q].get(a % q, 0)
            if cnt > 0:
                target = n / q
                w = min(w, float(target) / float(cnt))
        w = max(1e-6, min(w, 10.0))  # clip for stability
        weights.append(w)

    total_w = sum(weights)
    lam = (keep_ratio * n) / total_w if total_w > 0 else 1.0

    A_thin: List[int] = []
    for a, w in zip(A_sorted, weights):
        p = lam * w
        if p >= 1.0 or random.random() < p:
            A_thin.append(a)

    return A_thin
