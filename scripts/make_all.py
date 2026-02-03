# scripts/make_all.py
import subprocess

def run(cmd):
    print(">>", " ".join(cmd))
    # Use check=True to raise if the command fails
    subprocess.run(cmd, check=True)

def main():
    # 1) Grid and annotated threshold plot
    run(["python", "-m", "scripts.run_grid", "--out", "results_1e6_2e6_5e6.csv"])
    run(["python", "-m", "scripts.plot_grid_with_thresholds",
         "--csv", "results_1e6_2e6_5e6.csv", "--tol", "0.01"])

    # 2) Augmentation report at a failing point (n=2e6, C=1.40)
    run(["python", "-m", "scripts.augment_report", "--n", "2000000",
         "--C", "1.40", "--Cbump", "0.05"])

    # 3) Augmentation cost curve
    run(["python", "-m", "scripts.plot_augment_curve", "--n", "2000000", "--C", "1.40",
         "--bumps", "0.02,0.03,0.04,0.05,0.06"])

    print("\n[done] All artifacts generated:")
    print(" - results_1e6_2e6_5e6.csv")
    print(" - plots/uncovered_vs_C_annotated.png")
    print(" - augment_report.csv (appended)")
    print(" - plots/augment_cost.png")
    print(" - plots/coverage_n*_C*_A*_U*.png when you run run_experiment with --plot")

if __name__ == "__main__":
    main()
