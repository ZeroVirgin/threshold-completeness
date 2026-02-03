# tc/augment_api.py
import math
from typing import Dict, Any
from tc.smooth import primes_upto, generate_friables
from tc.cover import coverage_bitset
from tc.diagnose import uncovered_indices
from tc.augment import greedy_augment_to_cover

def run_augment_once(n: int, C: float, Cbump: float, start: int = 2) -> Dict[str, Any]:
    yA = int((math.log(n)) ** C)
    yH = int((math.log(n)) ** (C + Cbump))
    A = generate_friables(n, primes_upto(yA))
    B = coverage_bitset(A, n)
    unc = uncovered_indices(B, start=start)
    Aset = set(A)
    H_all = generate_friables(n, primes_upto(yH))
    H = [h for h in H_all if h not in Aset]
    added, remaining = greedy_augment_to_cover(n=n, A=A, uncovered=unc, halo=H, max_add=None, start=start)
    A_prime = A if not added else sorted(A + added)
    Bp = coverage_bitset(A_prime, n)
    uncp = uncovered_indices(Bp, start=start)
    return {
        "n": n, "C": C, "Cbump": Cbump, "yA": yA, "yH": yH,
        "A_size": len(A), "unc_base": len(unc), "H_candidates": len(H),
        "added": len(added), "unc_after": len(uncp),
        "added_list": added[:10],
    }
