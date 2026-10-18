"""Test 1 (400 held-out): A1 vs A3 first-3-agree vs A1r loop-detect+restart."""
import csv
import random
import statistics
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "results/e2_400.csv"
rows = list(csv.DictReader(open(P)))
n = len(rows)


def col(k):
    return [int(r[k]) for r in rows]


def wcol(k):
    return [float(r[k]) for r in rows]


def pct(x):
    return 100 * sum(x) / n


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return 100 * (c - h), 100 * (c + h)


def paired_ci(a, b, reps=10000):
    random.seed(0)
    d = []
    for _ in range(reps):
        idx = [random.randrange(n) for _ in range(n)]
        d.append(sum(b[i] - a[i] for i in idx) / n)
    d.sort()
    return 100 * d[int(0.025 * reps)], 100 * d[int(0.975 * reps)]


arms = [("A1 N=1", "a1_correct", "a1_wall_ms"), ("A3 N=16 K=3", "a3_correct", "a3_wall_ms"), ("A1r loop+restart", "a1r_correct", "a1r_wall_ms")]
print(f"n={n}")
for name, ck, wk in arms:
    k = sum(col(ck))
    lo, hi = wilson(k, n)
    print(f"{name:>16}: acc {100*k/n:5.1f}%  CI {lo:.1f}-{hi:.1f}  wall median {statistics.median(wcol(wk)):.0f} ms")

b1 = statistics.median(wcol("a1_wall_ms"))
print()
for name, ck, wk in arms[1:]:
    dlo, dhi = paired_ci(col("a1_correct"), col(ck))
    print(f"{name:>16} vs A1: diff {100*(sum(col(ck))-sum(col('a1_correct')))/n:+.1f} pts  CI [{dlo:+.1f}, {dhi:+.1f}]  "
          f"wall {statistics.median(wcol(wk))/b1:.2f}x")
