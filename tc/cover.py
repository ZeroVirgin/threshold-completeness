from __future__ import annotations
import numpy as np
import numba as nb
from bitarray import bitarray


@nb.njit(parallel=True, fastmath=True, cache=True)
def _mark_pairs(A: np.ndarray, n: int, hits: np.ndarray) -> None:
    """
    Mark sums s = A[i] + A[j] <= n. hits is a uint8 array, hits[s] set to 1 if covered.
    Early-break on j-loop when sums exceed n (A is sorted non-decreasing).
    """
    m = A.size
    for i in nb.prange(m):
        ai = A[i]
        for j in range(m):
            s = ai + A[j]
            if s > n:
                break
            hits[s] = 1


def coverage_bitset(A_list: list[int], n: int) -> bitarray:
    """
    Return bitset B of length n+1, where B[k] == 1 iff k ∈ (A + A) and 0 <= k <= n.
    """
    if n < 1:
        B = bitarray(1)
        B.setall(False)
        return B

    # Ensure sorted, contiguous array for better cache behavior in JIT loop
    A = np.asarray(sorted(A_list), dtype=np.int32)
    hits = np.zeros(n + 1, dtype=np.uint8)

    _mark_pairs(A, n, hits)

    B = bitarray(n + 1)
    B.setall(False)
    # Vectorized pick of nonzero indices, then set bits
    idx = np.nonzero(hits)[0]
    for k in idx:
        B[k] = True
    return B
