"""E0 summary: p50 step time vs N and unshared-tail depth, plus gate verdicts."""
import csv
import statistics
import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "results/e0_1p5b.csv"

rows = []
with open(SRC) as f:
    for r in csv.DictReader(f):
        rows.append((int(r["N"]), int(r["tail_depth"]), float(r["step_ms"]), int(r["clock_mhz"])))

maxt = max(r[1] for r in rows)
DEPTHS = [d for d in (0, 512, 1024) if d <= maxt] + [maxt - 15]

Ns = sorted({r[0] for r in rows})


def med(N, t0, w=16):
    v = [r[2] for r in rows if r[0] == N and t0 <= r[1] < t0 + w]
    return statistics.median(v) if v else float("nan")


print("p50 step ms (window=16)")
print("N  ", *[f"{d:>8}" for d in DEPTHS], "  per-token@1024  clock")
for N in Ns:
    ck = [r[3] for r in rows if r[0] == N and r[3] > 0]
    print(f"{N:<3}", *[f"{med(N,d):>8.2f}" for d in DEPTHS],
          f"  {med(N,1024)/N:>10.2f}", f"  {statistics.median(ck):.0f}")

base = {d: med(1, d) for d in DEPTHS}
print("\nratio vs N=1:")
for N in Ns:
    print(f"N={N:<3}", *[f"tail{d}={med(N,d)/base[d]:.2f}x" for d in DEPTHS])

print("\ngates:")
r8_1k = med(8, 1024) / base[1024]
r8_2k = med(8, 2032) / base[2032]
print(f"  stop gate (N=8 <= 2.0x at tail>=1024): tail1024={r8_1k:.2f}x "
      f"{'PASS' if r8_1k <= 2.0 else 'FAIL'}; tail2032={r8_2k:.2f}x "
      f"{'PASS' if r8_2k <= 2.0 else 'FAIL'}")
print(f"  E2 latency gate (N=8 <= 1.6x): tail1024={r8_1k:.2f}x "
      f"{'PASS' if r8_1k <= 1.6 else 'FAIL'} (peak step time; retirement/early-stop not yet applied)")
