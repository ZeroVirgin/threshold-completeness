import argparse
import math
import time
import numpy as np
import matplotlib.pyplot as plt

from tc.smooth import primes_upto, generate_friables
from tc.cover import coverage_bitset
from tc.diagnose import uncovered_indices, residue_hist, longest_uncovered_run
from tc.thin import residue_balanced_thin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1_000_000, help="Upper bound for A+A coverage test")
    ap.add_argument("--C", type=float, default=2.0, help="Smoothness exponent: y=(log n)^C (natural log)")
    ap.add_argument("--start", type=int, default=2, help="Lower bound of coverage range for uncovered reporting")
    ap.add_argument("--qmax", type=int, default=64, help="Residue histogram up to modulus qmax")
    ap.add_argument("--plot", action="store_true", help="Save a sampled coverage plot")
    ap.add_argument("--include-zero", action="store_true", help="Include 0 in A (treat 0 as smooth)")
    ap.add_argument("--thin", action="store_true", help="Apply residue-balanced thinning before coverage")
    ap.add_argument("--qmax-thin", type=int, default=64, help="Max modulus for thinning balances (2..qmax_thin)")
    ap.add_argument("--keep-ratio", type=float, default=1.0, help="Target retained fraction under thinning")
    args = ap.parse_args()

    n = args.n
    C = args.C
    y = int((math.log(n)) ** C)

    print(f"[config] n={n:,}  C={C:.2f}  y=(log n)^C={y}  start={args.start}")
    print(f"[config] include_zero={args.include_zero}  thin={args.thin} qmax_thin={args.qmax_thin} keep_ratio={args.keep_ratio}")

    t0 = time.time()
    y_primes = primes_upto(y)
    t1 = time.time()
    print(f"[stage] primes<=y: {len(y_primes)} (t={t1-t0:.2f}s)")

    friables = generate_friables(n, y_primes)
    if args.include_zero:
        if 0 not in friables:
            friables = [0] + friables
    t2 = time.time()
    print(f"[stage] |A| (y-smooth<=n): {len(friables):,} (t={t2-t1:.2f}s)")
    print(f"         first 10: {friables[:10]} ... last: {friables[-1]}")

    if args.thin:
        t_th0 = time.time()
        friables_thin = residue_balanced_thin(
            friables, qmax_thin=args.qmax_thin, keep_ratio=args.keep_ratio, seed=12345
        )
        t_th1 = time.time()
        print(f"[stage] thinning: |A'|={len(friables_thin):,} (t={t_th1 - t_th0:.2f}s)")
        A_used = friables_thin
    else:
        A_used = friables

    B = coverage_bitset(A_used, n)
    t3 = time.time()
    print(f"[stage] coverage computed (A+A) (t={t3-t2:.2f}s)")

    unc = uncovered_indices(B, start=args.start)
    print(f"[result] uncovered count: {len(unc):,} / {n:,}")
    if unc:
        print(f"         first 10 uncovered: {unc[:10]}")
        print(f"         longest uncovered run: {longest_uncovered_run(unc)}")
        rh = residue_hist(unc, qmax=args.qmax)
        for q in (8, 12):
            if q in rh:
                row = rh[q]
                ordered = ", ".join(f"{a}:{row.get(a,0)}" for a in range(q))
                print(f"         residue hist mod {q}: {ordered}")

    if args.plot:
        stride = max(1, n // 10000)
        xs = np.arange(1, n + 1, stride)
        ys = [1 if B[i] else 0 for i in xs]
        import os
        os.makedirs("plots", exist_ok=True)
        plt.figure(figsize=(10, 2.5))
        plt.plot(xs, ys, lw=0.7)
        plt.ylim(-0.1, 1.1)
        plt.title(f"Coverage (sampled) — n={n:,}, C={C:.2f}, y={(math.log(n))**C:.0f}")
        plt.xlabel("k")
        plt.ylabel("covered?")
        out = f"plots/coverage_n{n}_C{C:.2f}.png"
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        print(f"[plot] saved {out}")


if __name__ == "__main__":
    main()
