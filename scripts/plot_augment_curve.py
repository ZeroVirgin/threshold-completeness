# scripts/plot_augment_curve.py
import argparse
import subprocess
import re
import numpy as np
import matplotlib.pyplot as plt

def run_one(n, C, Cbump):
    cmd = ["python", "-m", "scripts.run_augment", "--n", str(n), "--C", str(C), "--Cbump", str(Cbump)]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    m = re.search(r"\[augment\]\s+added=(\d+)\s+remaining_uncovered=(\d+)", out)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2_000_000)
    ap.add_argument("--C", type=float, default=1.40)
    ap.add_argument("--bumps", default="0.02,0.03,0.04,0.05,0.06")
    ap.add_argument("--out", default="plots/augment_cost.png")
    args = ap.parse_args()

    bumps = [float(x) for x in args.bumps.split(",")]
    added = []
    remain = []

    print(f"[config] n={args.n:,} C={args.C} bumps={bumps}")
    for b in bumps:
        a, r = run_one(args.n, args.C, b)
        print(f"  Cbump={b:.3f} -> added={a} remaining={r}")
        added.append(a)
        remain.append(r)

    plt.figure(figsize=(6,4))
    plt.plot(bumps, added, marker="o")
    plt.xlabel("C bump (ΔC)")
    plt.ylabel("# added elements to achieve coverage")
    plt.title(f"Augmentation cost vs ΔC (n={args.n:,}, base C={args.C})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"[plot] saved {args.out}")

if __name__ == "__main__":
    main()
