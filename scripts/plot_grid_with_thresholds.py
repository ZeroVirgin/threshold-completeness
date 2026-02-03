# scripts/plot_grid_with_thresholds.py
import csv
import math
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt

# Lightweight probe used to refine C* lines (calls your core pipeline)
from tc.smooth import primes_upto, generate_friables
from tc.cover import coverage_bitset
from tc.diagnose import uncovered_indices

def uncovered_count(n: int, C: float, start: int = 2) -> int:
    y = int((math.log(n)) ** C)
    A = generate_friables(n, primes_upto(y))
    B = coverage_bitset(A, n)
    return len(uncovered_indices(B, start=start))

def refine_threshold(n: int, Cmin: float, Cmax: float, tol: float = 0.01, start: int = 2) -> float | None:
    """Binary search smallest C in [Cmin, Cmax] with 0 uncovered; None if none."""
    lo, hi = Cmin, Cmax
    best = None
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        u = uncovered_count(n, mid, start=start)
        print(f"[refine] n={n:,} C={mid:.4f} -> uncovered={u}")
        if u == 0:
            best = mid
            hi = mid
        else:
            lo = mid
    return best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results_1e6_2e6_5e6.csv", help="grid CSV from run_grid.py")
    ap.add_argument("--out", default="plots/uncovered_vs_C_annotated.png")
    ap.add_argument("--start", type=int, default=2)
    ap.add_argument("--tol", type=float, default=0.01)
    args = ap.parse_args()

    rows = []
    with open(args.csv, newline="") as f:
        for r in csv.DictReader(f):
            r["n"] = int(r["n"])
            r["C"] = float(r["C"])
            r["uncovered"] = int(r["uncovered"])
            rows.append(r)

    # Group by n and plot uncovered vs C
    by_n = defaultdict(list)
    for r in rows:
        by_n[r["n"]].append((r["C"], r["uncovered"]))
    for n in by_n:
        by_n[n].sort()

    plt.figure(figsize=(7.2, 4.2))
    for n, series in sorted(by_n.items()):
        Cs = [c for c, _ in series]
        U = [u for _, u in series]
        plt.plot(Cs, U, marker="o", label=f"n={n:,}")

    # Compute C* per n by finding smallest C with U=0 among sampled points,
    # then refine with a small binary search in its neighborhood.
    for n, series in sorted(by_n.items()):
        # find first sampled C with uncovered == 0
        C0 = None
        for c, u in series:
            if u == 0:
                C0 = c
                break
        if C0 is None:
            continue
        # refine between previous sampled point and C0
        left = max([c for c, u in series if c < C0], default=C0 - 0.1)
        Cstar = refine_threshold(n, left, C0, tol=args.tol, start=args.start)
        if Cstar is not None:
            plt.axvline(Cstar, color="#888", alpha=0.35, linestyle="--")
            plt.text(Cstar + 0.005, max(1, min(plt.ylim()[1]/15, 50)), f"C*≈{Cstar:.3f}\n(n={n:,})",
                     rotation=90, va="bottom", ha="left", fontsize=9, color="#444")

    plt.yscale("symlog", linthresh=1)
    plt.xlabel(r"C ($y=(\log n)^C$)")
    plt.ylabel("uncovered count (symlog)")
    plt.title("Uncovered vs C with empirical thresholds $C_*(n)$")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    import os
    os.makedirs("plots", exist_ok=True)
    plt.savefig(args.out, dpi=150)
    print(f"[plot] saved {args.out}")

if __name__ == "__main__":
    main()
