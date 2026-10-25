#!/usr/bin/env python3
"""Summarize the fair HumanEval baseline at --actor 512 (stdlib only).

Reads the evaluated per-seed CSVs produced by scripts/eval_humaneval.py, writes
a combined CSV with an explicit `seed` column, and reports pass@1 per seed and
pooled, latency percentiles, truncation, and syntax/eval status counts.

Usage:
    python scripts/analyze_humaneval_baseline.py \
        --seeds 1234 5678 9012 --actor 512
"""

import argparse
import csv
import hashlib
import math
import os
import statistics


def pct(values, p):
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p / 100.0
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return s[int(k)]
    return s[lo] * (hi - k) + s[hi] * (k - lo)


def sha256(path):
    if not path or not os.path.exists(path):
        return "n/a"
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in ("idx", "pass1", "syntax_ok", "actor_pass1", "actor_syntax_ok",
                  "actor_n", "final_n", "stitch", "actor_stitch"):
            r[k] = int(r.get(k) or 0)
        for k in ("actor_ms", "total_ms", "prefill_ms", "eval_ms"):
            r[k] = float(r.get(k) or 0.0)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1234, 5678, 9012])
    ap.add_argument("--actor", type=int, default=512)
    ap.add_argument("--pattern", default="results/humaneval_baseline_512_seed{seed}.csv")
    ap.add_argument("--raw-pattern", default="results/humaneval_baseline_512_seed{seed}_raw.csv")
    ap.add_argument("--out", default="results/humaneval_baseline_512.csv")
    ap.add_argument("--summary", default="results/humaneval_baseline_512_summary.txt")
    ap.add_argument("--model", default="models/Llama-3.2-3B-Instruct-Q4_K_M.gguf")
    ap.add_argument("--data", default="data/humaneval_100.jsonl")
    ap.add_argument("--tsv", default="data/humaneval_100.tsv")
    args = ap.parse_args()

    per_seed = {}
    all_rows = []
    fieldnames = None
    for seed in args.seeds:
        path = args.pattern.format(seed=seed)
        rows = load(path)
        for r in rows:
            r["seed"] = seed
        per_seed[seed] = rows
        all_rows.extend(rows)
        if fieldnames is None:
            fieldnames = ["seed"] + [k for k in rows[0].keys() if k != "seed"]
        print(f"seed {seed}: {len(rows)} rows from {path}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    n_all = len(all_rows)
    pooled_pass = sum(r["pass1"] for r in all_rows)

    L = []
    w = L.append
    w("RecycleLM HumanEval fair baseline at --actor %d" % args.actor)
    w("=" * 72)
    for seed in args.seeds:
        w("input seed %-5d: %s  sha256=%s" % (
            seed, args.pattern.format(seed=seed),
            sha256(args.pattern.format(seed=seed))))
        w("       raw      : %s  sha256=%s" % (
            args.raw_pattern.format(seed=seed),
            sha256(args.raw_pattern.format(seed=seed))))
    w("combined: %s  sha256=%s" % (args.out, sha256(args.out)))
    w("model:   %s  sha256=%s" % (args.model, sha256(args.model)))
    w("data:    %s  sha256=%s" % (args.data, sha256(args.data)))
    w("tsv:     %s  sha256=%s" % (args.tsv, sha256(args.tsv)))
    w("")
    w("protocol: official HumanEval. stitch==0 -> prompt + \"\\n\" + code + test +")
    w("check(entry_point); stitch==1/2 -> code already contains the prompt/preamble.")
    w("model: Llama-3.2-3B-Instruct Q4_K_M. prompt: system \"You are a Python")
    w("programmer. Complete the function. Output only one ```python code block.\" +")
    w("HumanEval prompt + \"\\nComplete the function.\" sampling: temp 0.6, top_k 20,")
    w("top_p 0.95, seed = seed + task index. No retry.")
    w("")
    w("pass@1")
    hdr = f"{'seed':>6} {'n':>4} {'pass':>5} {'pass@1':>8} {'median_ms':>10} {'p95_ms':>8}"
    w(hdr)
    w("-" * len(hdr))
    for seed in args.seeds:
        rows = per_seed[seed]
        p = sum(r["pass1"] for r in rows)
        w(f"{seed:>6} {len(rows):>4} {p:>5} {100.0*p/len(rows):>7.1f}% "
          f"{pct([r['total_ms'] for r in rows], 50):>10.0f} "
          f"{pct([r['total_ms'] for r in rows], 95):>8.0f}")
    w(f"{'pooled':>6} {n_all:>4} {pooled_pass:>5} "
      f"{100.0*pooled_pass/n_all:>7.1f}% "
      f"{pct([r['total_ms'] for r in all_rows], 50):>10.0f} "
      f"{pct([r['total_ms'] for r in all_rows], 95):>8.0f}")
    w("")
    w("pass@1 per-seed mean = %.2f%%  min = %.2f%%  max = %.2f%%" % (
        100.0 * statistics.mean(sum(r["pass1"] for r in per_seed[s]) / len(per_seed[s])
                                for s in args.seeds),
        100.0 * min(sum(r["pass1"] for r in per_seed[s]) / len(per_seed[s])
                    for s in args.seeds),
        100.0 * max(sum(r["pass1"] for r in per_seed[s]) / len(per_seed[s])
                    for s in args.seeds)))
    w("")
    w("latency (all %d rows, ms)" % n_all)
    tots = [r["total_ms"] for r in all_rows]
    acts = [r["actor_ms"] for r in all_rows]
    prefs = [r["prefill_ms"] for r in all_rows]
    w("  total:   mean=%.1f median=%.1f p95=%.1f max=%.1f" % (
        statistics.mean(tots), pct(tots, 50), pct(tots, 95), max(tots)))
    w("  actor:   mean=%.1f median=%.1f p95=%.1f" % (
        statistics.mean(acts), pct(acts, 50), pct(acts, 95)))
    w("  prefill: median=%.1f" % pct(prefs, 50))
    w("")
    w("truncation at actor cap %d" % args.actor)
    n_trunc = sum(1 for r in all_rows if r["final_n"] >= args.actor)
    n_trunc_actor = sum(1 for r in all_rows if r["actor_n"] >= args.actor)
    w("  final_n >= %d: %d/%d (%.1f%%)" % (
        args.actor, n_trunc, n_all, 100.0 * n_trunc / n_all))
    w("  actor_n >= %d: %d/%d (%.1f%%)" % (
        args.actor, n_trunc_actor, n_all, 100.0 * n_trunc_actor / n_all))
    if n_trunc:
        by = {}
        for r in all_rows:
            if r["final_n"] >= args.actor:
                by.setdefault(r["seed"], []).append(r["task_id"])
        for s in sorted(by):
            w("    seed %d: %s" % (s, ", ".join(by[s])))
    w("")
    w("syntax / eval status (pooled)")
    status = {}
    for r in all_rows:
        status[r.get("eval_status", "")] = status.get(r.get("eval_status", ""), 0) + 1
    for k in sorted(status):
        w("  eval_status %-10s %d" % (k, status[k]))
    n_syntax = sum(1 for r in all_rows if r["syntax_ok"] == 0)
    w("  syntax_ok==0: %d/%d (%.1f%%)" % (
        n_syntax, n_all, 100.0 * n_syntax / n_all))
    stitched = sum(1 for r in all_rows if r["stitch"])
    w("  stitched artifacts (stitch=1/2): %d" % stitched)
    w("")
    w("failure modes (final artifact, pooled)")
    kinds = {}
    for r in all_rows:
        if r["pass1"]:
            continue
        e = r.get("eval_error", "")
        kind = e.split(":")[0].strip() if e else r.get("eval_status", "?")
        kinds[kind] = kinds.get(kind, 0) + 1
    for k, v in sorted(kinds.items(), key=lambda kv: -kv[1]):
        w("  %-16s %d" % (k, v))
    w("")
    w("per-seed failures on HumanEval/38 and HumanEval/50 (evaluator fix targets)")
    for s in args.seeds:
        for r in per_seed[s]:
            if r["task_id"] in ("HumanEval/38", "HumanEval/50"):
                w("  seed %d %s: pass1=%d stitch=%d status=%s %s" % (
                    s, r["task_id"], r["pass1"], r["stitch"],
                    r.get("eval_status", ""), r.get("eval_error", "")[:80]))
    w("")
    w("Commands")
    for s in args.seeds:
        w("  humaneval.exe --model %s --data data\\humaneval_100.tsv "
          "--out results\\humaneval_baseline_512_seed%d_raw.csv --limit 100 "
          "--mode baseline --actor %d --seed %d" % (args.model, s, args.actor, s))
        w("  python scripts\\eval_humaneval.py --csv "
          "results\\humaneval_baseline_512_seed%d_raw.csv --data data\\humaneval_100.jsonl "
          "--out results\\humaneval_baseline_512_seed%d.csv" % (s, s))
    w("  python scripts\\analyze_humaneval_baseline.py --seeds %s --actor %d" % (
        " ".join(str(s) for s in args.seeds), args.actor))

    text = "\n".join(L) + "\n"
    with open(args.summary, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
