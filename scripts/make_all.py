# scripts/make_all.py
"""
One-shot artifact builder for the threshold-completeness project.

What it produces:
  - results_1e6_2e6_5e6.csv                      (grid results: uncovered vs C for each n)
  - plots/uncovered_vs_C_annotated.png           (Figure 2: phase transition with C*(n) lines)
  - augment_report.csv                           (CSV row for a representative augmentation run)
  - plots/augment_cost.png                       (Figure 3: augmentation cost vs ΔC)
  - Optional: two coverage plots (Figure 1 & a pre-threshold case) if --coverage-plots is set

Usage:
  python -m scripts.make_all
  python -m scripts.make_all --coverage-plots
  python -m scripts.make_all --engine tiled
"""

import argparse
import os
import sys
import subprocess
from shutil import which


def run(cmd: list[str]) -> None:
    """Run a command, stream output, fail fast on non-zero return."""
    print("\n>>", " ".join(cmd), flush=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"[make_all] ERROR: Command failed with exit code {rc}", file=sys.stderr)
        sys.exit(rc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--engine",
        choices=["njit", "tiled"],
        default="njit",
        help="Coverage engine to use for illustrative coverage plots (njit fastest on Windows).",
    )
    ap.add_argument(
        "--coverage-plots",
        action="store_true",
        help="Also generate two coverage plots (Figure 1 and a pre-threshold case).",
    )
    ap.add_argument(
        "--n-augment",
        type=int,
        default=2_000_000,
        help="n used for the augmentation demo.",
    )
    ap.add_argument(
        "--C-augment",
        type=float,
        default=1.40,
        help="Base C used for the augmentation demo.",
    )
    ap.add_argument(
        "--Cbump",
        type=float,
        default=0.05,
        help="ΔC bump for augmentation demo.",
    )
    args = ap.parse_args()

    # Resolve Python being used (inside venv)
    py = sys.executable
    print(f"[info] Using Python: {py}")

    # Ensure output folders exist
    os.makedirs("plots", exist_ok=True)

    # 1) Grid over (n, C) and CSV
    #    This uses run_grid.py which internally calls the fast coverage path;
    #    no plotting here (keeps the run quick).
    run([py, "-m", "scripts.run_grid", "--out", "results_1e6_2e6_5e6.csv"])

    # 2) Annotated thresholds (reads the CSV + refines C* with small binary searches)
    run([py, "-m", "scripts.plot_grid_with_thresholds", "--csv", "results_1e6_2e6_5e6.csv", "--tol", "0.01"])

    # 3) Augmentation report at a representative failing point
    #    Writes/append a row to augment_report.csv and prints a LaTeX row to console.
    run([
        py, "-m", "scripts.augment_report",
        "--n", str(args.n_augment),
        "--C", str(args.C_augment),
        "--Cbump", str(args.Cbump),
    ])

    # 4) Augmentation cost curve (calls augment API directly for robust numbers)
    run([
        py, "-m", "scripts.plot_augment_curve",
        "--n", str(args.n_augment),
        "--C", str(args.C_augment),
        "--bumps", "0.02,0.03,0.04,0.05,0.06",
    ])

    # 5) (Optional) Two illustrative coverage plots:
    #    - Figure 1: n=1e6, C=2.00 (flat coverage = 1 from 2..n)
    #    - A pre-threshold example: n=5e6, C=1.40 (a few uncovered)
    if args.coverage_plots:
        # Ensure matplotlib is available (it should be from earlier steps)
        if which("python") is None:
            print("[warn] Could not resolve python for coverage plots; skipping.", file=sys.stderr)
        else:
            # Figure 1 (showcase, full coverage)
            run([
                py, "-m", "scripts.run_experiment",
                "--n", "1000000",
                "--C", "2.0",
                "--plot",
                "--engine", args.engine,
            ])
            # Pre-threshold illustrative case
            run([
                py, "-m", "scripts.run_experiment",
                "--n", "5000000",
                "--C", "1.40",
                "--plot",
                "--engine", args.engine,
            ])

    print("\n[done] All artifacts generated:")
    print(" - results_1e6_2e6_5e6.csv")
    print(" - plots/uncovered_vs_C_annotated.png")
    print(" - augment_report.csv (appended)")
    print(" - plots/augment_cost.png")
    if args.coverage_plots:
        print(" - plots/coverage_n*_C*_A*_U*.png (two showcase cases)")


if __name__ == "__main__":
    main()
