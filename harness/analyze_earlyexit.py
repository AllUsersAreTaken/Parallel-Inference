"""Offline analysis for the early-exit probe experiment.

Compares, on the same traces/probes:
  - k-consecutive-agreeing-probe stop rules (the Dynasor-style method)
  - static early exit (fixed token budget)
  - logprob-confidence stop
and reports accuracy vs mean reasoning tokens, with Wilson CIs.
"""
import csv
import math
import re
import sys

PREFIX = sys.argv[1] if len(sys.argv) > 1 else "results/ee"
THINK_END = "<" + "/" + "think" + ">"


def norm_num(s):
    if s is None:
        return None
    s = s.replace(",", "").replace("$", "").strip()
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        return round(float(m.group(0)), 6)
    except ValueError:
        return None


def extract_answer(text):
    if not text:
        return None
    boxed = re.findall(r"\\boxed\{([^{}]*)\}", text)
    if boxed:
        v = norm_num(re.sub(r"[^0-9.\-]", "", boxed[-1]))
        if v is not None:
            return v
    seg = text.rsplit(THINK_END, 1)[-1] if THINK_END in text else text
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", seg)
    return norm_num(nums[-1]) if nums else None


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


traces = {}
with open(f"{PREFIX}.traces.tsv", encoding="utf-8") as f:
    for row in csv.reader(f, delimiter="\t"):
        if len(row) < 4:
            continue
        i, gold, tlen, text = int(row[0]), row[1], int(row[2]), row[3]
        traces[i] = {"gold": norm_num(gold), "len": tlen, "final": extract_answer(text)}

probes = {}
with open(f"{PREFIX}.probes.tsv", encoding="utf-8") as f:
    for row in csv.reader(f, delimiter="\t"):
        if len(row) < 4:
            continue
        i, N, gc, text = int(row[0]), int(row[1]), int(row[2]), row[3]
        probes.setdefault(i, {})[gc] = extract_answer(text)
for i in probes:
    probes[i] = dict(sorted(probes[i].items()))

logp = {}
with open(f"{PREFIX}.logprobs.tsv", encoding="utf-8") as f:
    for row in csv.reader(f, delimiter="\t"):
        if len(row) < 3:
            continue
        i, t, lp = int(row[0]), int(row[1]), float(row[2])
        logp.setdefault(i, {})[t] = lp

ids = sorted(traces)
n = len(ids)
full_acc = 100 * sum(1 for i in ids if traces[i]["final"] == traces[i]["gold"]) / n
mean_len = sum(traces[i]["len"] for i in ids) / n
print(f"problems={n}  full-trace accuracy={full_acc:.1f}%  mean trace tokens={mean_len:.0f}")


def evaluate(chooser):
    """chooser(id) -> (tokens_used, answer)."""
    correct = 0
    toks = 0
    for i in ids:
        used, ans = chooser(i)
        toks += used
        if ans is not None and ans == traces[i]["gold"]:
            correct += 1
    acc = 100 * correct / n
    lo, hi = wilson(correct, n)
    return acc, lo, hi, toks / n


print("\n== k-consecutive-agreeing probes ==")
print(f"{'N':>4} {'k':>2} {'acc%':>6} {'CI95':>14} {'tokens':>7} {'saved%':>7}")
rows = []
for N in (64, 128, 256):
    for k in (2, 3, 4):
        def chooser(i, N=N, k=k):
            seq = [(gc, a) for gc, a in probes.get(i, {}).items() if gc % N == 0]
            for j in range(len(seq)):
                win = seq[max(0, j - k + 1):j + 1]
                if len(win) == k and all(a is not None for _, a in win) and len({a for _, a in win}) == 1:
                    return win[-1][0], win[-1][1]
            return traces[i]["len"], traces[i]["final"]
        acc, lo, hi, t = evaluate(chooser)
        saved = 100 * (1 - t / mean_len)
        print(f"{N:>4} {k:>2} {acc:>6.1f} {lo:>6.1f}-{hi:<6.1f} {t:>7.0f} {saved:>6.1f}")
        rows.append(("kagree", N, k, acc, lo, hi, t, saved))

print("\n== static early exit (fixed budget) ==")
print(f"{'N':>4} {'probe#':>6} {'acc%':>6} {'tokens':>7} {'saved%':>7}")
for N in (64, 128, 256):
    maxj = max(len([gc for gc in probes.get(i, {}) if gc % N == 0]) for i in ids)
    for j in range(maxj):
        def chooser(i, N=N, j=j):
            seq = [(gc, a) for gc, a in probes.get(i, {}).items() if gc % N == 0]
            if j < len(seq):
                return seq[j][0], seq[j][1]
            return traces[i]["len"], traces[i]["final"]
        acc, lo, hi, t = evaluate(chooser)
        saved = 100 * (1 - t / mean_len)
        print(f"{N:>4} {j:>6} {acc:>6.1f} {t:>7.0f} {saved:>6.1f}")
        rows.append(("static", N, j, acc, lo, hi, t, saved))

print("\n== logprob-confidence stop (mean logprob of last chunk) ==")
print(f"{'N':>4} {'thr':>7} {'acc%':>6} {'tokens':>7} {'saved%':>7}")
for N in (64, 128, 256):
    for thr in (-0.02, -0.05, -0.1, -0.2, -0.4, -0.8):
        def chooser(i, N=N, thr=thr):
            seq = [gc for gc in probes.get(i, {}) if gc % N == 0]
            lp = logp.get(i, {})
            for gc in seq:
                chunk = [lp[t] for t in range(max(0, gc - N), gc) if t in lp]
                if chunk and sum(chunk) / len(chunk) > thr:
                    return gc, probes[i][gc]
            return traces[i]["len"], traces[i]["final"]
        acc, lo, hi, t = evaluate(chooser)
        saved = 100 * (1 - t / mean_len)
        print(f"{N:>4} {thr:>7.2f} {acc:>6.1f} {t:>7.0f} {saved:>6.1f}")
        rows.append(("conf", N, thr, acc, lo, hi, t, saved))

with open(f"{PREFIX}.pareto.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["method", "N", "param", "acc", "ci_lo", "ci_hi", "tokens", "saved_pct"])
    w.writerows(rows)
print(f"\nwrote {PREFIX}.pareto.csv")
