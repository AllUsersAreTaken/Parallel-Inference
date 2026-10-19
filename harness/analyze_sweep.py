"""Accuracy vs N sweep (200 held-out): which N to recommend."""
import csv
import statistics
import sys

P = sys.argv[1] if len(sys.argv) > 1 else "results/e2_sweep200.csv"
rows = list(csv.DictReader(open(P)))
Ns = sorted({int(r["N"]) for r in rows})


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return 100 * (c - h), 100 * (c + h)


base_w = statistics.median(float(r["wall_ms"]) for r in rows if int(r["N"]) == 1)
print(f"{'N':>3} {'K':>2} {'acc%':>6} {'95% CI':>13} {'wall_ms':>8} {'ratio':>6} {'tokens':>8}")
for N in Ns:
    rs = [r for r in rows if int(r["N"]) == N]
    n = len(rs)
    k = sum(int(r["correct"]) for r in rs)
    lo, hi = wilson(k, n)
    w = statistics.median(float(r["wall_ms"]) for r in rs)
    t = statistics.median(int(r["tokens"]) for r in rs)
    K = rs[0]["K"]
    print(f"{N:>3} {K:>2} {100*k/n:>6.1f} {lo:>6.1f}-{hi:<6.1f} {w:>8.0f} {w/base_w:>6.2f} {t:>8.0f}")
