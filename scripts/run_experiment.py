# scripts/run_experiment.py
import argparse
import math
import time
import numpy as np
import matplotlib.pyplot as plt

from tc.smooth import primes_upto, generate_friables
# Coverage engines are imported conditionally based on --engine
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

    # New engine flag
    ap.add_argument(
        "--engine",
        choices=["njit", "tiled", "mp"],
        default="njit",
        help="njit=single-process; tiled=njit+cache-tiling; mp=multi-process (slower on Windows for large n)",
    )
    # Worker count for mp engine
    ap.add_argument("--blocks", type=int, default=8, help="Process count for --engine mp")
    # Legacy compatibility: --parallel maps to --engine mp (hidden in help)
    ap.add_argument("--parallel", action="store_true", help=argparse.SUPPRESS)

    args = ap.parse_args()

    # Map legacy --parallel to --engine mp
    if getattr(args, "parallel", False) and args.engine != "mp":
        print("[warn] --parallel is deprecated; using --engine=mp")
        args.engine = "mp"

    n = args.n
    C = args.C
    y = int((math.log(n)) ** C)

    print(f"[config] n={n:,}  C={C:.2f}  y=(log n)^C={y}  start={args.start}")
    print(f"[config] include_zero={args.include_zero}  thin={args.thin} qmax_thin={args.qmax_thin} keep_ratio={args.keep_ratio}")
    print(f"[config] engine={args.engine}" + (f" blocks={args.blocks}" if args.engine == "mp" else ""))

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

    # Coverage selection by engine
    if args.engine == "mp":
        from tc.cover import coverage_bitset_parallel
        B = coverage_bitset_parallel(A_used, n, blocks=args.blocks)
    elif args.engine == "tiled":
        from tc.cover import coverage_bitset_njit
        B = coverage_bitset_njit(A_used, n, tiled=True)
    else:  # "njit"
        from tc.cover import coverage_bitset_njit
        B = coverage_bitset_njit(A_used, n, tiled=False)

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
        # --- improved plotting block ---
        stride = max(1, n // 10000)
        xs = np.arange(2, n + 1, stride)  # sample from 2 upward
        ys = [1 if B[i] else 0 for i in xs]

        import os
        os.makedirs("plots", exist_ok=True)

        plt.figure(figsize=(10, 2.6))
        plt.plot(xs, ys, lw=0.8, color="#206eff", label="sampled coverage")

        unc_all = uncovered_indices(B, start=args.start)
        if unc_all:
            plt.scatter(unc_all, [0]*len(unc_all), s=8, color="#d62728", alpha=0.85, label="uncovered")

        plt.ylim(-0.1, 1.1)
        plt.xlim(0, n)
        plt.yticks([0, 1], ["no", "yes"])
        plt.xlabel("k")
        plt.ylabel("covered?")

        title = (
            f"Coverage (sampled) — n={n:,}, C={C:.2f}, y={(math.log(n))**C:.0f}, "
            f"|A|={len(A_used):,}, uncovered[{args.start}..n]={len(unc_all)}"
        )
        plt.title(title)
        plt.grid(alpha=0.25, linewidth=0.6)
        plt.legend(loc="lower right", frameon=False)

        out = f"plots/coverage_n{n}_C{C:.2f}_A{len(A_used)}_U{len(unc_all)}.png"
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        print(f"[plot] saved {out}")
        # --- end block ---


if __name__ == "__main__":
    main()
