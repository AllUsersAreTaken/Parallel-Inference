"""Test 1: N=1 vs N=16 first-3-agree, live on the 4060."""
import csv
import statistics
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "results/e2_100.csv"

rows = list(csv.DictReader(open(P)))
n = len(rows)
a1 = [int(r["a1_correct"]) for r in rows]
a3 = [int(r["a3_correct"]) for r in rows]
w1 = [float(r["a1_wall_ms"]) for r in rows]
w3 = [float(r["a3_wall_ms"]) for r in rows]


def pct(x):
    return 100 * sum(x) / n


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return 100 * (c - h), 100 * (c + h)


# paired bootstrap on the accuracy difference
import random
random.seed(0)
diffs = []
for _ in range(10000):
    idx = [random.randrange(n) for _ in range(n)]
    diffs.append(sum(a3[i] - a1[i] for i in idx) / n)
diffs.sort()
lo, hi = diffs[int(0.025 * 10000)], diffs[int(0.975 * 10000)]

print(f"n={n}")
print(f"A1 (N=1)  accuracy {pct(a1):.1f}%  CI {wilson(sum(a1),n)[0]:.1f}-{wilson(sum(a1),n)[1]:.1f}")
print(f"A3 (N=16,K=3) accuracy {pct(a3):.1f}%  CI {wilson(sum(a3),n)[0]:.1f}-{wilson(sum(a3),n)[1]:.1f}")
print(f"paired diff {pct(a3)-pct(a1):+.1f} pts  95% CI [{100*lo:+.1f}, {100*hi:+.1f}] pts")
print()
print(f"wall clock median: A1 {statistics.median(w1):.0f} ms  A3 {statistics.median(w3):.0f} ms  "
      f"ratio {statistics.median(w3)/statistics.median(w1):.2f}x")
print(f"wall clock mean:   A1 {statistics.mean(w1):.0f} ms  A3 {statistics.mean(w3):.0f} ms  "
      f"ratio {statistics.mean(w3)/statistics.mean(w1):.2f}x")
print(f"A3 finish median {statistics.median([int(r['a3_finish']) for r in rows]):.0f} tokens")
