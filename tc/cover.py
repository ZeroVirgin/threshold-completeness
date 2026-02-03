# tc/cover.py
from __future__ import annotations

from typing import List, Optional, Tuple
import os

import numpy as np
from bitarray import bitarray

# --- Optional: Numba path ----------------------------------------------------
# We prefer the Numba-accelerated implementation if available;
# otherwise we fall back to a multiprocessing implementation.
_NUMBA_AVAILABLE = False
try:
    import numba as nb  # type: ignore

    _NUMBA_AVAILABLE = True
except Exception:
    _NUMBA_AVAILABLE = False

# --- Multiprocessing fallback implementation (your original idea, cleaned) ----
from multiprocessing import Pool, cpu_count


def _cover_block(args: Tuple[np.ndarray, int, int, int]) -> np.ndarray:
    """
    Worker: mark sums s = A[i] + A[j] <= n for i in [lo, hi).
    Returns a uint8 vector 'hits' of length n+1.
    """
    A, n, lo, hi = args
    m = A.size
    hits = np.zeros(n + 1, dtype=np.uint8)
    for i in range(lo, hi):
        ai = A[i]
        # j-loop early break relies on A being sorted ascending
        for j in range(m):
            s = ai + A[j]
            if s > n:
                break
            hits[s] = 1
    return hits


def coverage_bitset_parallel(A_list: List[int], n: int, blocks: Optional[int] = None) -> bitarray:
    """
    Parallel coverage using multiprocessing.Pool.
    Returns bitset B of length n+1 where B[k] == 1 iff k in (A + A) and 0 <= k <= n.
    """
    if n < 1:
        B = bitarray(1)
        B.setall(False)
        return B

    # Ensure sorted array for early-break behavior and better locality
    A = np.asarray(sorted(A_list), dtype=np.int32)
    m = A.size
    if m == 0:
        B = bitarray(n + 1)
        B.setall(False)
        return B

    if blocks is None:
        # Good default on mixed P/E-core mobile CPUs
        blocks = min(cpu_count(), 8)

    # Evenly split the i-index across blocks
    splits = np.linspace(0, m, blocks + 1, dtype=int)
    tasks = [(A, n, int(splits[k]), int(splits[k + 1])) for k in range(blocks)]

    # Pool-Map and OR-reduce the hit vectors
    with Pool(processes=blocks) as p:
        parts = p.map(_cover_block, tasks)

    hits = np.bitwise_or.reduce(parts)

    # Convert to bitarray
    B = bitarray(n + 1)
    B.setall(False)
    idx = np.nonzero(hits)[0]
    for k in idx:
        B[k] = True
    return B


# --- Numba-accelerated implementation (preferred path) -----------------------
if _NUMBA_AVAILABLE:
    @nb.njit(parallel=True, fastmath=True, cache=True)
    def _mark_pairs(A: np.ndarray, n: int, hits: np.ndarray) -> None:
        """
        Mark sums s = A[i] + A[j] <= n. hits is a uint8 array; hits[s] set to 1 if covered.
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


def coverage_bitset(A_list: List[int], n: int) -> bitarray:
    """
    Return bitset B of length n+1, where B[k] == 1 iff k ∈ (A + A) and 0 <= k <= n.

    Strategy:
      * Prefer a Numba-jitted loop with early-break (fastest).
      * If Numba is unavailable or JIT errors at runtime, fall back to
        a well-parallelized multiprocessing implementation.

    Environment Override:
      * Set env var TC_COVER_IMPL='parallel' to force the multiprocessing path.
      * Set env var TC_COVER_IMPL='numba' to force the Numba path (if available).
    """
    if n < 1:
        B = bitarray(1)
        B.setall(False)
        return B

    # Normalize and sort A for monotonic increases in the inner loop
    A = np.asarray(sorted(A_list), dtype=np.int32)

    # Empty A ⇒ no sums
    if A.size == 0:
        B = bitarray(n + 1)
        B.setall(False)
        return B

    impl = os.environ.get("TC_COVER_IMPL", "").strip().lower()

    # If user explicitly wants the parallel path
    if impl == "parallel":
        return coverage_bitset_parallel(A.tolist(), n)

    # Try Numba first (unless explicitly disabled / unavailable)
    if _NUMBA_AVAILABLE and impl != "parallel":
        hits = np.zeros(n + 1, dtype=np.uint8)
        try:
            _mark_pairs(A, n, hits)  # JIT on first call; cached afterwards
            B = bitarray(n + 1)
            B.setall(False)
            idx = np.nonzero(hits)[0]
            for k in idx:
                B[k] = True
            return B
        except Exception:
            # Any JIT/runtime failure: fall back to parallel implementation
            pass

    # Fallback: multiprocessing
    return coverage_bitset_parallel(A.tolist(), n)


__all__ = [
    "coverage_bitset",
    "coverage_bitset_parallel",
]
