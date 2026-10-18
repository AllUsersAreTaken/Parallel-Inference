"""Composition test: live step time vs the E0 cost model (acceptance: 5% at p50)."""
import csv
import statistics

R = "S:/Agent/QSExplore/RecycleLM/results/"

E0 = {}
with open(R + "e0_full.csv") as f:
    for r in csv.DictReader(f):
        E0.setdefault(int(r["N"]), {})[int(r["tail_depth"])] = float(r["step_ms"])
ks = sorted(E0)


def e0(N, t):
    N = min(max(N, ks[0]), ks[-1])
    if N in E0 and t in E0[N]:
        return E0[N][t]
    lo = max(k for k in ks if k <= N)
    hi = min(k for k in ks if k >= N)
    a, b = E0[lo].get(t), E0[hi].get(t)
    return a + (b - a) * (N - lo) / (hi - lo) if a and b else None


print(f"{'config':>16} {'active':>6} {'live_ms':>8} {'pred_ms':>8} {'resid':>7}")
for (t, p) in [(0, 0), (4, 0), (16, 0), (0, 64), (4, 64), (16, 64)]:
    with open(R + f"compose_t{t}_p{p}.csv") as f:
        rows = list(csv.DictReader(f))
    act = [int(r["active"]) for r in rows[20:]]
    ms = [float(r["step_ms"]) for r in rows[20:]]
    a_med = statistics.median(act)
    m_med = statistics.median(ms)
    tail = statistics.median([int(r["step"]) for r in rows[20:]])
    pred = e0(int(round(a_med)), int(tail))
    resid = 100 * (m_med - pred) / pred if pred else float("nan")
    print(f"traces={t:>2} p={p:>2}  {a_med:>6.0f} {m_med:>8.2f} {pred:>8.2f} {resid:>+6.1f}%")
