# scripts/run_augment.py
import argparse
import math
from tc.smooth import primes_upto, generate_friables
from tc.cover import coverage_bitset
from tc.diagnose import uncovered_indices
from tc.augment import greedy_augment_to_cover

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2_000_000)
    ap.add_argument("--C", type=float, default=1.4, help="Base smoothness exponent for A: y=(log n)^C")
    ap.add_argument("--Cbump", type=float, default=0.05, help="Tiny bump for halo: use C' = C + Cbump")
    ap.add_argument("--start", type=int, default=2)
    ap.add_argument("--max-add", type=int, default=None, help="Optional cap on number of added elements")
    args = ap.parse_args()

    n, C, Cbump = args.n, args.C, args.Cbump
    yA = int((math.log(n)) ** C)
    yH = int((math.log(n)) ** (C + Cbump))
    print(f"[config] n={n:,}  C={C:.3f}  yA={yA}  C'={C+Cbump:.3f}  yH={yH}  start={args.start}")

    # Base A (y-smooth)
    P_A = primes_upto(yA)
    A = generate_friables(n, P_A)
    B = coverage_bitset(A, n)
    unc = uncovered_indices(B, start=args.start)
    print(f"[base] |A|={len(A):,} uncovered={len(unc)}")
    if not unc:
        print("[base] Already fully covered. Nothing to augment.")
        return

    # Halo candidates H = yH-smooth minus yA-smooth (new elements permitted to add)
    P_H = primes_upto(yH)
    H_all = generate_friables(n, P_H)
    # remove A (i.e., only candidates that are new)
    A_set = set(A)
    H = [h for h in H_all if h not in A_set]
    print(f"[halo] |H_all|={len(H_all):,}  new_candidates=|H|={len(H):,}")

    # Greedy augment
    added, remaining = greedy_augment_to_cover(
        n=n, A=A, uncovered=unc, halo=H, max_add=args.max_add, start=args.start
    )
    print(f"[augment] added={len(added)} remaining_uncovered={len(remaining)}")
    if added:
        print(f"          first 10 added: {added[:10]}")

    # Verify coverage after adding
    if added:
        # Conceptually, we form A' = A ∪ added and re-check
        A_prime = sorted(A + added)
        Bp = coverage_bitset(A_prime, n)
        uncp = uncovered_indices(Bp, start=args.start)
        print(f"[verify] uncovered after augmentation: {len(uncp)}")

if __name__ == "__main__":
    main()
