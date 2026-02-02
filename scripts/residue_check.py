import math
from tc.smooth import primes_upto, generate_friables
from tc.diagnose_A import residue_hist_A

def main():
    n, C = 2_000_000, 1.4
    y = int((math.log(n))**C)
    A = generate_friables(n, primes_upto(y))
    rh = residue_hist_A(A, qmax=12)
    print("Residue counts mod 8:",  [rh[8].get(a, 0)  for a in range(8)])
    print("Residue counts mod 12:", [rh[12].get(a, 0) for a in range(12)])

if __name__ == "__main__":
    main()
