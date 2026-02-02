from __future__ import annotations
from typing import List, Tuple
import heapq


def primes_upto(m: int) -> list[int]:
    """
    Sieve of Eratosthenes: return all primes <= m.
    """
    if m < 2:
        return []
    sieve = bytearray(b"\x01") * (m + 1)
    sieve[:2] = b"\x00\x00"
    limit = int(m**0.5)
    for p in range(2, limit + 1):
        if sieve[p]:
            start = p * p
            step = p
            sieve[start : m + 1 : step] = b"\x00" * (((m - start) // step) + 1)
    return [i for i, b in enumerate(sieve) if b]


def generate_friables(n: int, y_primes: List[int]) -> list[int]:
    """
    Generate all y-smooth integers <= n using a multiplicative BFS over a min-heap.
    Includes 1 by convention. Output is sorted.
    """
    if n < 1:
        return []
    A: list[int] = []
    # Each heap item is (value, min_prime_index_allowed)
    heap: list[Tuple[int, int]] = [(1, 0)]
    seen = {1}
    while heap:
        val, idx = heapq.heappop(heap)
        A.append(val)
        # Multiply by any allowed prime at j >= idx (non-decreasing prime sequence)
        for j in range(idx, len(y_primes)):
            p = y_primes[j]
            nv = val * p
            if nv > n:
                break
            if nv not in seen:
                seen.add(nv)
                heapq.heappush(heap, (nv, j))
    A.sort()
    return A
