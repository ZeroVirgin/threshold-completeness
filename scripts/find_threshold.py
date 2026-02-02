# scripts/find_threshold.py
import argparse
import math
from tc.smooth import primes_upto, generate_friables
from tc.cover import coverage_bitset
from tc.diagnose import uncovered_indices

def zero_uncovered(n: int, C: float, start: int = 2) -> tuple[int, int]:
    y = int((math.log(n)) ** C)
    A = generate_friables(n, primes_upto(y))
    B = coverage_bitset(A, n)
    unc = uncovered_indices(B, start=start)
    return len(A), len(unc)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2_000_000)
    ap.add_argument("--Cmin", type=float, default=1.20)
    ap.add_argument("--Cmax", type=float, default=2.00)
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--start", type=int, default=2)
    args = ap.parse_args()

    lo, hi = args.Cmin, args.Cmax
    best = None
    while hi - lo > args.tol:
        mid = 0.5 * (lo + hi)
        Asize, unc = zero_uncovered(args.n, mid, start=args.start)
        print(f"[probe] C={mid:.4f}  |A|={Asize:,}  uncovered={unc}")
        if unc == 0:
            best = mid
            hi = mid
        else:
            lo = mid

    if best is None:
        print("[result] No C in range achieved full coverage.")
    else:
        print(f"[result] Minimal C≈{best:.4f} (tol={args.tol}) for n={args.n:,}")

if __name__ == "__main__":
    main()
