# scripts/augment_report.py
import argparse
import math
import csv
from tc.smooth import primes_upto, generate_friables
from tc.cover import coverage_bitset
from tc.diagnose import uncovered_indices
from tc.augment import greedy_augment_to_cover

def run_one(n: int, C: float, Cbump: float, start: int = 2):
    yA = int((math.log(n)) ** C)
    yH = int((math.log(n)) ** (C + Cbump))

    # base set A and uncovered
    A = generate_friables(n, primes_upto(yA))
    B = coverage_bitset(A, n)
    unc = uncovered_indices(B, start=start)

    # halo candidates H
    Aset = set(A)
    H_all = generate_friables(n, primes_upto(yH))
    H = [h for h in H_all if h not in Aset]

    added, remaining = greedy_augment_to_cover(n=n, A=A, uncovered=unc, halo=H, max_add=None, start=start)
    A_prime = sorted(A + added) if added else A
    Bp = coverage_bitset(A_prime, n)
    uncp = uncovered_indices(Bp, start=start)

    return {
        "n": n,
        "C": C,
        "Cbump": Cbump,
        "yA": yA,
        "yH": yH,
        "|A|": len(A),
        "uncovered_base": len(unc),
        "halo_candidates": len(H),
        "added": len(added),
        "uncovered_after": len(uncp),
        "added_first10": added[:10],
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2_000_000)
    ap.add_argument("--C", type=float, default=1.40)
    ap.add_argument("--Cbump", type=float, default=0.05)
    ap.add_argument("--start", type=int, default=2)
    ap.add_argument("--out", default="augment_report.csv")
    args = ap.parse_args()

    res = run_one(args.n, args.C, args.Cbump, start=args.start)

    # CSV append
    header = ["n","C","Cbump","yA","yH","|A|","uncovered_base","halo_candidates","added","uncovered_after"]
    write_header = False
    try:
        with open(args.out, newline="") as _:
            pass
    except FileNotFoundError:
        write_header = True

    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if write_header:
            w.writeheader()
        w.writerow({k: res[k] for k in header})

    print(f"[csv] appended row to {args.out}")
    print(f"[added] first 10 added: {res['added_first10']}")

    # LaTeX row
    latex = (
        f"{args.n:,} & {args.C:.2f} & {args.Cbump:.2f} & {res['yA']} & {res['yH']} & "
        f"{res['|A|']:,} & {res['uncovered_base']:,} & {res['halo_candidates']:,} & "
        f"{res['added']:,} & {res['uncovered_after']:,} \\\\"
    )
    print("[latex row]")
    print(latex)

if __name__ == "__main__":
    main()
