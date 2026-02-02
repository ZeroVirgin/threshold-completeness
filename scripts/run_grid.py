import csv
import math
import time
import argparse

from tc.smooth import primes_upto, generate_friables
from tc.cover import coverage_bitset
from tc.diagnose import uncovered_indices


def run_one(n: int, C: float, start: int = 2, include_zero: bool = False) -> dict:
    y = int((math.log(n)) ** C)

    t0 = time.time()
    y_primes = primes_upto(y)
    friables = generate_friables(n, y_primes)
    if include_zero and (0 not in friables):
        friables = [0] + friables
    B = coverage_bitset(friables, n)
    unc = uncovered_indices(B, start=start)
    t1 = time.time()

    return {
        "n": n,
        "C": C,
        "y": y,
        "A_size": len(friables),
        "uncovered": len(unc),
        "time_sec": round(t1 - t0, 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="grid_results.csv", help="CSV output path")
    ap.add_argument("--start", type=int, default=2, help="Coverage lower bound")
    ap.add_argument("--include-zero", action="store_true", help="Include 0 in A")
    args = ap.parse_args()

    # Customize your sweeps here
    Ns = [1_000_000, 2_000_000, 5_000_000]
    Cs = [1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0]

    rows = []
    for n in Ns:
        for C in Cs:
            print(f"Running n={n:,} C={C:.2f} ...")
            res = run_one(n, C, start=args.start, include_zero=args.include_zero)
            print(f" -> A_size={res['A_size']:,} uncovered={res['uncovered']} time={res['time_sec']}s")
            rows.append(res)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["n", "C", "y", "A_size", "uncovered", "time_sec"])
        w.writeheader()
        w.writerows(rows)
    print(f"[grid] wrote {args.out}")


if __name__ == "__main__":
    main()
