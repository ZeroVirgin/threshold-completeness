from collections import defaultdict
from typing import Dict, List

def residue_hist_A(A: List[int], qmax: int = 64) -> Dict[int, Dict[int, int]]:
    out: Dict[int, Dict[int, int]] = {}
    for q in range(2, qmax + 1):
        d: Dict[int, int] = defaultdict(int)
        for a in A:
            d[a % q] += 1
        out[q] = dict(d)
    return out
