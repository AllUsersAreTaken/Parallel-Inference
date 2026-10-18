"""Padding + retirement analysis (E0-derived scheduling rules)."""
import csv
import statistics

R = "S:/Agent/QSExplore/RecycleLM/results/"


def load(p, N=None):
    d = {}
    with open(p) as f:
        for r in csv.DictReader(f):
            if N is not None and int(r["N"]) != N:
                continue
            d[int(r["tail_depth"])] = (float(r["step_ms"]), int(r.get("active") or r["N"]))
    return d


def med(d, t0, w=16):
    v = [d[t][0] for t in range(t0, t0 + w) if t in d]
    return statistics.median(v) if v else float("nan")


e0 = lambda N: load(R + "e0_full.csv", N)
e8, e9, e16 = e0(8), e0(9), e0(16)

print("PADDING: 8 real streams topped to K with filler")
print("  tail    8real  pad->9  pad->12 pad->16   E0_N9   E0_N16")
p9 = load(R + "pad8_9.csv", 8)
p12 = load(R + "pad8_12.csv", 8)
p16 = load(R + "pad8_16.csv", 8)
for t in (0, 512, 1024, 2032):
    print(f"  {t:>5}  {med(e8,t):>6.2f}  {med(p9,t):>6.2f}  {med(p12,t):>7.2f}  {med(p16,t):>7.2f}   {med(e9,t):>6.2f}  {med(e16,t):>7.2f}")

print("\nRETIREMENT @1024 (16 -> 8 active, seq_rm on)")
r = load(R + "retire1024.csv", 16)
print("  tail    rm1(8act)   E0_N8")
for t in (1040, 1400, 1800, 2032):
    print(f"  {t:>5}  {med(r,t):>9.2f}  {med(e8,t):>8.2f}")

print("\ncliff: N=8 vs N=9 (tail 1024): "
      f"{med(e8,1024):.2f} vs {med(e9,1024):.2f} ms "
      f"({med(e9,1024)-med(e8,1024):+.2f} ms for +1 token)")
