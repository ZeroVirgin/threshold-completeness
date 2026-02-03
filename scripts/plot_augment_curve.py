# scripts/plot_augment_curve.py (replace contents)
import argparse
import matplotlib.pyplot as plt
from tc.augment_api import run_augment_once

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
        res = run_augment_once(args.n, args.C, b, start=2)
        print(f"  Cbump={b:.3f} -> added={res['added']} remaining={res['unc_after']}")
        added.append(res["added"])
        remain.append(res["unc_after"])

    import os; os.makedirs("plots", exist_ok=True)
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
