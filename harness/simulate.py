"""Replay simulator: stopping rules over fixed traces + E0 step-cost model.

Traces: results/dev200_s16.tsv (id, stream, fin, full, text), 16 sampled traces
per problem. Cost model: results/e0_1p5b.csv (step_ms for N in {1,2,4,8,16}).

For each rule (N streams, first-K-agree) it computes accuracy and a per-problem
wall clock = sum over steps of cost(active_streams(t), tail=t). The N=1 baseline
uses stream 0's single trace. Reports the stopping-length distribution and the
wall-clock ratio, which is what the latency gate depends on.
"""
import csv
import re
import statistics
import sys

TRACES = sys.argv[1] if len(sys.argv) > 1 else "results/dev200_s16.tsv"
COST = sys.argv[2] if len(sys.argv) > 2 else "results/e0_1p5b.csv"
GOLD = sys.argv[3] if len(sys.argv) > 3 else "data/gsm8k_dev200.tsv"

TE = "<" + "/" + "think" + ">"


def norm_num(s):
    if s is None:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", re.sub(r"[^0-9.\-]", "", s))
    return round(float(m.group(0)), 6) if m else None


def extract(text):
    if not text:
        return None
    b = re.findall(r"\\boxed\{([^{}]*)\}", text)
    if b:
        v = norm_num(b[-1])
        if v is not None:
            return v
    seg = text.rsplit(TE, 1)[-1] if TE in text else text
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", seg)
    return norm_num(nums[-1]) if nums else None


# ---- cost model ----
raw = {}
with open(COST) as f:
    for r in csv.DictReader(f):
        raw.setdefault(int(r["N"]), {})[int(r["tail_depth"])] = float(r["step_ms"])

def smooth(N, t, w=8):
    d = raw.get(N, {})
    if not d:
        return None
    v = [d[u] for u in range(t - w, t + w + 1) if u in d]
    return statistics.median(v) if v else None

Ns_known = sorted(raw)
def fN(N, t):
    if N in raw:
        return smooth(N, t)
    lo = max((n for n in Ns_known if n <= N), default=Ns_known[0])
    hi = min((n for n in Ns_known if n >= N), default=Ns_known[-1])
    if lo == hi:
        return smooth(lo, t)
    a, b = smooth(lo, t), smooth(hi, t)
    return a + (b - a) * (N - lo) / (hi - lo)

def wallclock(fins, stop_len, N_total):
    """sum step cost with active streams = #(fin > t), tails t=0..stop_len."""
    s = 0.0
    for t in range(stop_len + 1):
        active = sum(1 for x in fins if x > t)
        if active == 0:
            active = 1
        s += fN(active, t)
    return s

# ---- traces ----
prob = {}
with open(TRACES, encoding="utf-8") as f:
    for r in csv.reader(f, delimiter="\t"):
        if len(r) < 5:
            continue
        i, j, fin, full, text = int(r[0]), int(r[1]), int(r[2]), int(r[3]), r[4]
        prob.setdefault(i, {})[j] = (fin, extract(text))

gold = {}
with open(GOLD, encoding="utf-8") as f:
    for k, line in enumerate(f, 1):
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 2:
            gold[k] = norm_num(parts[1])

ids = sorted(i for i in prob if i in gold)
N1_len = [prob[i][0][0] for i in ids]
N1_wall = [wallclock([prob[i][0][0]], prob[i][0][0], 1) for i in ids]
print(f"problems={len(ids)}  N=1 median len={statistics.median(N1_len):.0f}  "
      f"N=1 median wall={statistics.median(N1_wall):.1f} ms")


def run(N, K):
    acc = 0
    stoplens = []
    walls = []
    for i in ids:
        streams = [prob[i][j] for j in sorted(prob[i])[:N]]
        order = sorted(streams, key=lambda x: x[0])
        ans = None
        stop = None
        seen = []
        for fin, a in order:
            seen.append(a)
            if len(seen) >= K:
                win = seen[-K:]
                if all(x is not None for x in win) and len(set(win)) == 1:
                    ans = win[-1]
                    stop = fin
                    break
        if ans is None:
            # majority vote of all, stop at last fin
            vals = [a for _, a in streams if a is not None]
            ans = statistics.mode(vals) if vals else None
            stop = max(f for f, _ in streams)
        stoplens.append(stop)
        walls.append(wallclock([f for f, _ in streams], stop, N))
        if ans is not None and ans == gold[i]:
            acc += 1
    return 100 * acc / len(ids), statistics.median(stoplens), statistics.median(walls)

print(f"\n{'N':>3} {'K':>2} {'acc%':>6} {'stop_len':>9} {'stop/N1':>8} {'wall_ms':>9} {'wall_ratio':>10}")
base_wall = statistics.median(N1_wall)
base_len = statistics.median(N1_len)
for N in (1, 4, 9, 12, 16):
    if N == 1:
        continue
    for K in (2, 3, 4, 5):
        if K > N:
            continue
        acc, sl, wl = run(N, K)
        print(f"{N:>3} {K:>2} {acc:>6.1f} {sl:>9.0f} {sl/base_len:>8.2f} {wl:>9.1f} {wl/base_wall:>10.2f}")
