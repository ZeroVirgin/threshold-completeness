# scripts/plot_grid.py
import csv
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results_1e6_2e6_5e6.csv")
    ap.add_argument("--out", default="plots/uncovered_vs_C.png")
    args = ap.parse_args()

    rows = []
    with open(args.csv, newline="") as f:
        for r in csv.DictReader(f):
            r["n"] = int(r["n"])
            r["C"] = float(r["C"])
            r["uncovered"] = int(r["uncovered"])
            rows.append(r)

    by_n = defaultdict(list)
    for r in rows:
        by_n[r["n"]].append((r["C"], r["uncovered"]))
    for n in by_n:
        by_n[n].sort()

    plt.figure(figsize=(7,4))
    for n, series in sorted(by_n.items()):
        Cs = [c for c, _ in series]
        U  = [u for _, u in series]
        plt.plot(Cs, U, marker="o", label=f"n={n:,}")

    plt.yscale("symlog", linthresh=1)
    plt.xlabel("C (y=(log n)^C)")
    plt.ylabel("uncovered count (symlog)")
    plt.title("Uncovered vs C for each n")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"[plot] saved {args.out}")

if __name__ == "__main__":
    main()
