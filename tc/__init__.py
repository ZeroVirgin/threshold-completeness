"""
tc — Threshold Completeness core package.

Exports:
- primes_upto, generate_friables          (tc.smooth)
- coverage_bitset                         (tc.cover)
- uncovered_indices, residue_hist,
  longest_uncovered_run                   (tc.diagnose)
"""

from .smooth import primes_upto, generate_friables
from .cover import coverage_bitset
from .diagnose import uncovered_indices, residue_hist, longest_uncovered_run

__all__ = [
    "primes_upto",
    "generate_friables",
    "coverage_bitset",
    "uncovered_indices",
    "residue_hist",
    "longest_uncovered_run",
]
