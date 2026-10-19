"""Paired tests (McNemar + bootstrap) on the sweep, and the N=1 cap check."""
import csv
import random
import statistics
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "results/e2_sweep200.csv"
rows = list(csv.DictReader(open(P)))
by = {}
for r in rows:
    by.setdefault(int(r["id"]), {})[int(r["N"])] = int(r["correct"])
ids = sorted(by)
Ns = sorted({int(r["N"]) for r in rows})


def mcnemar(a, b, na, nb):
    A = [by[i].get(na, 0) for i in ids]
    B = [by[i].get(nb, 0) for i in ids]
    n01 = sum(1 for x, y in zip(A, B) if x == 0 and y == 1)   # a wrong, b right
    n10 = sum(1 for x, y in zip(A, B) if x == 1 and y == 0)   # a right, b wrong
    disc = n01 + n10
    chi2 = ((abs(n01 - n10) - 1) ** 2) / disc if disc else 0.0
    # exact binomial two-sided p
    p = 1.0
    if disc:
        from math import comb
        k = min(n01, n10)
        p = 2 * sum(comb(disc, j) for j in range(0, k + 1)) / (2 ** disc)
        p = min(1.0, p)
    return n01, n10, chi2, p


def boot(a, b, na, nb, reps=10000):
    random.seed(0)
    A = [by[i].get(na, 0) for i in ids]
    B = [by[i].get(nb, 0) for i in ids]
    d = []
    for _ in range(reps):
        idx = [random.randrange(len(ids)) for _ in ids]
        d.append(sum(B[i] - A[i] for i in idx) / len(ids))
    d.sort()
    return 100 * d[int(0.025 * reps)], 100 * d[int(0.975 * reps)]


print("paired comparisons (n=%d)" % len(ids))
for na, nb in [(1, 16), (4, 8), (8, 16), (4, 16)]:
    n01, n10, chi2, p = mcnemar(None, None, na, nb)
    lo, hi = boot(None, None, na, nb)
    diff = 100 * (sum(by[i].get(nb, 0) for i in ids) - sum(by[i].get(na, 0) for i in ids)) / len(ids)
    print(f"  N={na} vs N={nb}: diff {diff:+.1f} pts  boot CI [{lo:+.1f}, {hi:+.1f}]  "
          f"McNemar b={n01} c={n10} chi2={chi2:.2f} p={p:.3f}")

print("\nN=1 token distribution (cap check)")
t1 = [int(r["tokens"]) for r in rows if int(r["N"]) == 1]
print(f"  median {statistics.median(t1)}  p90 {sorted(t1)[int(0.9*len(t1))]}  max {max(t1)}")
for th in (400, 480, 500, 510):
    print(f"  tokens >= {th}: {sum(1 for t in t1 if t >= th)}/{len(t1)}")
