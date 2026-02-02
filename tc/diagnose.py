from __future__ import annotations
from collections import defaultdict
from typing import Dict, List
from bitarray import bitarray


def uncovered_indices(B: bitarray, start: int = 2) -> list[int]:
    """
    Indices k (start..len(B)-1) for which B[k] == 0.
    By default we ignore 1 since A+A with A⊂Z_{>0} can't hit 1 unless 0 in A.
    """
    start = max(1, start)
    return [k for k in range(start, len(B)) if not B[k]]


def residue_hist(uncovered: List[int], qmax: int = 64) -> Dict[int, Dict[int, int]]:
    """
    Count uncovered residues up to small moduli.
    Returns {q: {a: count}} for 2 <= q <= qmax.
    """
    out: Dict[int, Dict[int, int]] = {}
    for q in range(2, qmax + 1):
        d: Dict[int, int] = defaultdict(int)
        for k in uncovered:
            d[k % q] += 1
        out[q] = dict(d)
    return out


def longest_uncovered_run(uncovered: List[int]) -> int:
    """
    Length of the longest consecutive run in the sorted uncovered list.
    """
    if not uncovered:
        return 0
    longest = 1
    curr = 1
    for i in range(1, len(uncovered)):
        if uncovered[i] == uncovered[i - 1] + 1:
            curr += 1
            if curr > longest:
                longest = curr
        else:
            curr = 1
    return longest
