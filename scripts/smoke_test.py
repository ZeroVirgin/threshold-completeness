import math
from tc.smooth import primes_upto, generate_friables
from tc.cover import coverage_bitset
from tc.diagnose import uncovered_indices, longest_uncovered_run

def main():
    n = 200_000
    C = 2.0
    y = int((math.log(n))**C)

    print(f\"n={n:,}  C={C}  y=(log n)^C ≈ {y}\")
    P = primes_upto(y)
    A = generate_friables(n, P)
    print(f\"|A|={len(A):,}  first={A[:10]}  last={A[-1]}\")
    B = coverage_bitset(A, n)
    unc = uncovered_indices(B)
    print(f\"Uncovered={len(unc):,}  Longest run={longest_uncovered_run(unc)}\")

if __name__ == \"__main__\":
    main()
