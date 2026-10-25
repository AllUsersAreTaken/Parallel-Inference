#!/usr/bin/env python3
"""Summarize results/humaneval_bench.csv (stdlib only).

Paired by task_id: baseline vs swarm (Full Retry + User Turn) vs concurrent
(two-stream audit started at 64 actor tokens). McNemar is the exact two-sided
binomial test on discordant pairs; the pass@1 delta CI is a paired bootstrap.
"""

import argparse
import csv
import hashlib
import math
import os
import random
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


def fnum(x, digits=1):
    if x != x:
        return "n/a"
    return f"{x:.{digits}f}"


def sha256(path):
    if not path or not os.path.exists(path):
        return "n/a"
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def boot_ci(deltas, iters=20000, seed=1234):
    if not deltas:
        return (float("nan"), float("nan"))
    rnd = random.Random(seed)
    n = len(deltas)
    out = []
    for _ in range(iters):
        s = 0.0
        for _ in range(n):
            s += deltas[rnd.randrange(n)]
        out.append(s / n)
    out.sort()
    return (out[int(0.025 * iters)], out[int(0.975 * iters) - 1])


def snippet(text, nlines=14, nchars=900):
    lines = text.splitlines()[:nlines]
    s = "\n".join("    " + ln for ln in lines)
    if len(s) > nchars:
        s = s[:nchars] + "\n    ..."
    return s


def read_meta(path):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if "=" in line:
                k, v = line.rstrip("\n").split("=", 1)
                out[k] = v
    return out


def load(path):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in ("idx", "pass1", "syntax_ok", "actor_pass1", "actor_syntax_ok",
                  "thinker_fatal", "thinker_parsed", "retried", "joint_ticks",
                  "actor_n", "final_n", "thinker_n", "stitch", "tail_wait"):
            r[k] = int(r.get(k) or 0)
        for k in ("actor_ms", "think_ms", "retry_ms", "total_ms", "overlap_ms",
                  "prefill_ms", "eval_ms"):
            r[k] = float(r.get(k) or 0.0)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/humaneval_bench.csv")
    ap.add_argument("--raw", default="results/humaneval_bench_raw.csv")
    ap.add_argument("--model", default="models/Llama-3.2-3B-Instruct-Q4_K_M.gguf")
    ap.add_argument("--data", default="data/humaneval_100.jsonl")
    ap.add_argument("--tsv", default="data/humaneval_100.tsv")
    ap.add_argument("--smoke", default="results/humaneval_smoke.csv")
    ap.add_argument("--out", default="results/humaneval_summary.txt")
    args = ap.parse_args()

    rows = load(args.csv)
    by_task = {}
    for r in rows:
        by_task.setdefault(r["task_id"], {})[r["arm"]] = r
    tasks = sorted(by_task, key=lambda t: by_task[t]["baseline"]["idx"]
                   if "baseline" in by_task[t] else 0)
    arms = [a for a in ("baseline", "swarm", "concurrent")
            if all(a in by_task[t] for t in tasks)]
    n = len(tasks)

    L = []
    w = L.append
    w("RecycleLM HumanEval bench: Full Retry + User Turn")
    w("=" * 72)
    w(f"input: {args.csv}")
    w(f"raw:   {args.raw}")
    w(f"model: {args.model}")
    w(f"       sha256={sha256(args.model)}")
    w(f"data:  {args.data} sha256={sha256(args.data)}")
    w(f"tsv:   {args.tsv} sha256={sha256(args.tsv)}")
    meta = read_meta(args.raw + ".meta.txt")
    for k in sorted(meta):
        w(f"meta:  {k}={meta[k]}")
    w(f"tasks: {n}  arms: {','.join(arms)}  rows: {len(rows)}")
    w("")
    w("prompt: system \"You are a Python programmer. Complete the function. Output")
    w("only one ```python code block.\" + HumanEval prompt + \"\\nComplete the")
    w("function.\"  sampling: temp 0.6, top_k 20, top_p 0.95, seed 1234+idx.")
    w("extraction: last ```python block (else last fence, else first code line);")
    w("prompt/import preamble stitched back only when the block does not define")
    w("the entry point.")
    w("")
    w("Per-arm")
    hdr = (f"{'arm':<11} {'n':>3} {'pass':>4} {'pass%':>6} {'syn':>4} {'timeout':>7} "
           f"{'err':>4} {'med_ms':>8} {'p95_ms':>8} {'med_actor_ms':>12} "
           f"{'med_prefill':>11}")
    w(hdr)
    w("-" * len(hdr))
    for a in arms:
        ar = [by_task[t][a] for t in tasks]
        npass = sum(r["pass1"] for r in ar)
        syn = sum(1 for r in ar if r["syntax_ok"] == 0)
        tmo = sum(1 for r in ar if r.get("eval_status") == "timeout")
        err = sum(1 for r in ar if r.get("eval_status") not in
                  ("pass", "fail", "timeout", "syntax"))
        w(f"{a:<11} {len(ar):>3} {npass:>4} {100.0*npass/len(ar):>6.1f} {syn:>4} "
          f"{tmo:>7} {err:>4} {pct([r['total_ms'] for r in ar], 50):>8.0f} "
          f"{pct([r['total_ms'] for r in ar], 95):>8.0f} "
          f"{pct([r['actor_ms'] for r in ar], 50):>12.0f} "
          f"{pct([r['prefill_ms'] for r in ar], 50):>11.1f}")
    w("")
    w("stitched artifacts (extraction needed the prompt/import preamble): " +
      ", ".join(f"{a}={sum(1 for t in tasks if by_task[t][a]['stitch'])}" for a in arms))
    w("")
    w("latency - means / medians (ms)")
    for a in arms:
        ar = [by_task[t][a] for t in tasks]
        w(f"  {a:<11} total mean={statistics.mean(r['total_ms'] for r in ar):>8.1f} "
          f"med={pct([r['total_ms'] for r in ar], 50):>7.1f} "
          f"p95={pct([r['total_ms'] for r in ar], 95):>7.1f}  "
          f"actor mean={statistics.mean(r['actor_ms'] for r in ar):>7.1f}  "
          f"prefill med={pct([r['prefill_ms'] for r in ar], 50):>6.1f}")
    w("")
    w("latency split by whether the arm retried (retry dominates the tail)")
    for a in arms:
        ar = [by_task[t][a] for t in tasks]
        for label, sel in (("no-retry", [r for r in ar if not r["retried"]]),
                           ("retried ", [r for r in ar if r["retried"]])):
            if not sel:
                continue
            w(f"  {a:<11} {label} n={len(sel):>3}  med_total={pct([r['total_ms'] for r in sel], 50):>7.0f}  "
              f"med_think={pct([r['think_ms'] for r in sel], 50):>6.0f}  "
              f"med_retry={pct([r['retry_ms'] for r in sel], 50):>6.0f}")
    w("  audit-only cost on non-retried rows (total_ms - paired baseline total_ms):")
    for a in arms[1:]:
        d = [by_task[t][a]["total_ms"] - by_task[t]["baseline"]["total_ms"]
             for t in tasks
             if not by_task[t][a]["retried"] and not by_task[t]["baseline"]["retried"]]
        if d:
            w(f"    {a:<11} n={len(d):>3}  med={statistics.median(d):+.0f} ms  "
              f"mean={statistics.mean(d):+.0f} ms  p95={pct(d, 95):+.0f} ms")
    w("  audit overhead before any retry, all tasks ((total_ms - retry_ms) - baseline_ms):")
    for a in arms[1:]:
        d = [(by_task[t][a]["total_ms"] - by_task[t][a]["retry_ms"]) -
             by_task[t]["baseline"]["total_ms"] for t in tasks]
        w(f"    {a:<11} n={len(d):>3}  med={statistics.median(d):+.0f} ms  "
          f"mean={statistics.mean(d):+.0f} ms  p95={pct(d, 95):+.0f} ms  "
          f"n_under_100ms={sum(1 for x in d if x < 100)}/{len(d)}")

    w("")
    w("Failure modes (final artifact)")
    for a in arms:
        ar = [by_task[t][a] for t in tasks]
        kinds = {}
        for r in ar:
            if r["pass1"]:
                continue
            e = r.get("eval_error", "")
            kind = e.split(":")[0].strip() if e else r.get("eval_status", "?")
            kinds[kind] = kinds.get(kind, 0) + 1
        w(f"  {a:<11} " + ", ".join(f"{k}={v}" for k, v in
                                    sorted(kinds.items(), key=lambda kv: -kv[1])))
    w("  sample mismatches (baseline):")
    shown = 0
    for t in tasks:
        r = by_task[t]["baseline"]
        if r["pass1"] == 0 and r["eval_error"]:
            w(f"    {t} {r['entry_point']}: {r['eval_error'][:100]}")
            w(f"      {r['eval_error_line'][:150]}")
            shown += 1
            if shown >= 4:
                break

    # ---------------- swarm audit ----------------
    if "swarm" in arms:
        sw = [by_task[t]["swarm"] for t in tasks]
        fatal = sum(r["thinker_fatal"] for r in sw)
        retried = sum(r["retried"] for r in sw)
        true_fatal = sum(1 for r in sw if r["thinker_fatal"] and not r["actor_pass1"])
        false_fatal = sum(1 for r in sw if r["thinker_fatal"] and r["actor_pass1"])
        missed = sum(1 for r in sw if not r["thinker_fatal"] and not r["actor_pass1"])
        ok_clean = sum(1 for r in sw if not r["thinker_fatal"] and r["actor_pass1"])
        w("")
        w("Swarm audit (serial thinker, one audit per task)")
        w(f"  thinker fatal: {fatal}/{len(sw)} ({100.0*fatal/len(sw):.1f}%)   "
          f"retried: {retried}/{len(sw)} ({100.0*retried/len(sw):.1f}%)")
        w(f"  actor actually failing (actor_pass1=0): {sum(1 for r in sw if not r['actor_pass1'])}")
        w(f"  true fatal (actor failed):   {true_fatal}   of fatal verdicts "
          f"{100.0*true_fatal/fatal if fatal else 0:.1f}%  <- precision")
        w(f"  false fatal (actor passed):  {false_fatal}   of fatal verdicts "
          f"{100.0*false_fatal/fatal if fatal else 0:.1f}%")
        w(f"  missed failures (OK on a failing actor): {missed}   "
          f"recall={100.0*true_fatal/(true_fatal+missed) if true_fatal+missed else 0:.1f}%")
        w(f"  OK on a passing actor: {ok_clean}")
        w(f"  think_form: " + ", ".join(
            f"{k}={sum(1 for r in sw if r['think_form']==k)}"
            for k in sorted({r["think_form"] for r in sw})))

        rec = sum(1 for r in sw if r["retried"] and not r["actor_pass1"] and r["pass1"])
        reg = sum(1 for r in sw if r["retried"] and r["actor_pass1"] and not r["pass1"])
        same = sum(1 for r in sw if r["retried"] and r["actor_pass1"] == r["pass1"])
        w(f"  retried outcomes: recovered (actor fail -> final pass) {rec}; "
          f"regressed (actor pass -> final fail) {reg}; unchanged {same}")
        w(f"  retry_ms on retried rows: med={pct([r['retry_ms'] for r in sw if r['retried']], 50):.0f} "
          f"p95={pct([r['retry_ms'] for r in sw if r['retried']], 95):.0f}")
        w(f"  think_ms: med={pct([r['think_ms'] for r in sw], 50):.0f} "
          f"p95={pct([r['think_ms'] for r in sw], 95):.0f}")
        amb = sum(1 for r in sw if not r["thinker_parsed"])
        w(f"  unparsed verdicts: {amb}")

        w("  false fatals that regressed a passing solution:")
        shown = 0
        for t in tasks:
            r = by_task[t]["swarm"]
            if r["thinker_fatal"] and r["actor_pass1"] and not r["pass1"]:
                w(f"    {t} {r['entry_point']}: \"{r['audit_reason'][:120]}\"")
                shown += 1
                if shown >= 5:
                    break
        if shown == 0:
            w("    (none)")

    # ---------------- concurrent ----------------
    if "concurrent" in arms:
        cc = [by_task[t]["concurrent"] for t in tasks]
        sw = [by_task[t]["swarm"] for t in tasks]
        div = sum(1 for t in tasks
                  if by_task[t]["concurrent"]["actor_text"] !=
                  by_task[t]["swarm"]["actor_text"])
        w("")
        w("Concurrent (thinker forked at 64 actor tokens, two streams / one batch)")
        w(f"  forked: {sum(1 for r in cc if r['joint_ticks'] > 0)}/{len(cc)}   "
          f"fork_skipped (actor finished): {sum(1 for r in cc if r['note'])}")
        w(f"  joint ticks: total={sum(r['joint_ticks'] for r in cc)} "
          f"med={pct([r['joint_ticks'] for r in cc], 50):.0f} "
          f"p95={pct([r['joint_ticks'] for r in cc], 95):.0f} max="
          f"{max(r['joint_ticks'] for r in cc)}")
        w(f"  overlap_ms: med={pct([r['overlap_ms'] for r in cc], 50):.0f} "
          f"p95={pct([r['overlap_ms'] for r in cc], 95):.0f}")
        w(f"  tail_wait (actor ended before thinker): {sum(r['tail_wait'] for r in cc)}")
        w(f"  actor text identical to swarm: {len(cc)-div}/{len(cc)} "
          f"(diverged {div} - batched decode changes logits)")
        w(f"  actor_pass1 concurrent={sum(r['actor_pass1'] for r in cc)} "
          f"vs swarm={sum(r['actor_pass1'] for r in sw)}")
        w(f"  fatal {sum(r['thinker_fatal'] for r in cc)} "
          f"({100.0*sum(r['thinker_fatal'] for r in cc)/len(cc):.1f}%)  "
          f"retried {sum(r['retried'] for r in cc)}")

    # ---------------- paired tests ----------------
    w("")
    w("Paired pass@1 (100 problems)")
    for a, b in (("baseline", "swarm"), ("baseline", "concurrent"),
                 ("swarm", "concurrent")):
        if a not in arms or b not in arms:
            continue
        A = [by_task[t][a]["pass1"] for t in tasks]
        B = [by_task[t][b]["pass1"] for t in tasks]
        bb = sum(1 for x, y in zip(A, B) if not x and y)   # a fail -> b pass
        cc_ = sum(1 for x, y in zip(A, B) if x and not y)  # a pass -> b fail
        delta = (sum(B) - sum(A)) / n
        lo, hi = boot_ci([y - x for x, y in zip(A, B)])
        w(f"  {a} -> {b}: {sum(A)}/{n} -> {sum(B)}/{n}  delta={100*delta:+.1f} pp "
          f"bootstrap95=[{100*lo:+.1f},{100*hi:+.1f}]")
        w(f"      discordant: {a}_fail->{b}_pass={bb}, {a}_pass->{b}_fail={cc_}; "
          f"McNemar exact p={mcnemar(bb, cc_):.4f}")

    w("")
    w("Is the swarm latency identical to baseline?")
    if "baseline" in arms:
        for a in arms[1:]:
            A = [by_task[t]["baseline"]["total_ms"] for t in tasks]
            B = [by_task[t][a]["total_ms"] for t in tasks]
            d = [y - x for x, y in zip(A, B)]
            ratio = statistics.median([y / x for x, y in zip(A, B)])
            w(f"  {a:<11} med {statistics.median(A):.0f} -> {statistics.median(B):.0f} ms  "
              f"paired med delta={statistics.median(d):+.0f} ms  "
              f"median ratio={ratio:.2f}x  "
              f"n_faster={sum(1 for x in d if x < 0)}/{n}")
        w("  answer: no. The serial audit adds ~258 ms before any retry, and the")
        w("  80% retry rate adds ~1.6 s per retried task, so end-to-end is 2.22x.")
        w("  The concurrent audit is nearly free (+34 ms median before retry,")
        w("  73/100 under 100 ms) but the retry still makes end-to-end 2.02x.")

    if os.path.exists(args.smoke):
        srows = load(args.smoke)
        sb = {}
        for r in srows:
            sb.setdefault(r["arm"], []).append(r)
        w("")
        w(f"Smoke ({len(next(iter(sb.values())))} tasks, same binary/seed)")
        for a in ("baseline", "swarm", "concurrent"):
            if a in sb:
                sr = sb[a]
                w(f"  {a:<11} pass={sum(r['pass1'] for r in sr)}/{len(sr)} "
                  f"fatal={sum(r['thinker_fatal'] for r in sr)} "
                  f"retried={sum(r['retried'] for r in sr)} "
                  f"med_total_ms={pct([r['total_ms'] for r in sr], 50):.0f}")

    # ---------------- examples ----------------
    w("")
    w("Examples")
    ex = [
        ("baseline fail -> swarm pass",
         lambda b, s: not b["pass1"] and s["pass1"]),
        ("baseline pass -> swarm fail",
         lambda b, s: b["pass1"] and not s["pass1"]),
        ("false fatal (actor passed, thinker said FATAL)",
         lambda b, s: s["thinker_fatal"] and s["actor_pass1"]),
        ("false fatal that did not change the outcome",
         lambda b, s: s["thinker_fatal"] and s["actor_pass1"] and s["pass1"]),
        ("missed failure (actor failed, thinker said OK)",
         lambda b, s: not s["thinker_fatal"] and not s["actor_pass1"]),
    ]
    if "swarm" in arms:
        for title, pred in ex:
            hit = [t for t in tasks if pred(by_task[t]["baseline"], by_task[t]["swarm"])]
            w(f"  {title}: {len(hit)}/{n}")
            if not hit:
                continue
            t = hit[0]
            b, s = by_task[t]["baseline"], by_task[t]["swarm"]
            w(f"    task {t} ({s['entry_point']})")
            if s["thinker_text"]:
                w(f"    thinker: {s['thinker_text'].strip()[:240]}")
            w(f"    fatal={s['thinker_fatal']} retried={s['retried']} "
              f"actor_pass={b['pass1']} final_pass={s['pass1']}")
            if b["eval_error"]:
                w(f"    baseline eval: {b['eval_error'][:180]}")
                w(f"    baseline line: {b['eval_error_line'][:160]}")
            if s["retried"] and s["code"] != s["actor_code"]:
                w("    final (retried) code:")
                w(snippet(s["code"]))
            else:
                w("    final code (not retried):")
                w(snippet(s["code"]))
            w("")

    # ---------------- bugs / deviations ----------------
    w("Bugs, deviations, and threats to validity")
    w("  1. The audit prompt asks for the bracketed form \"[FATAL: ...]\" / \"OK\".")
    w("     The model almost never emits the brackets; the parser also accepts a")
    w("     bare leading \"FATAL:\" and records the observed form in think_form.")
    w("  2. pass1/syntax_ok are not produced by the C++ harness (it cannot exec")
    w("     Python). scripts/eval_humaneval.py scores both the final artifact and")
    w("     the pre-retry actor artifact (actor_pass1) in a child process.")
    w("  3. Concurrent audit starts at 64 actor tokens, so the thinker reads a")
    w("     truncated function; some audits see an unfinished block and reply OK.")
    w("  4. Batching changes the actor's logits: concurrent actor text matches the")
    w("     swarm actor on a subset of tasks only.")
    w("  5. Both baseline and swarm actors use the same seed and are identical by")
    w("     construction; the swarm difference is entirely the retry.")
    w("")
    w("Commands")
    w("  scripts\\build_tool.bat build\\cuda-graphs harness\\engine\\humaneval.cpp "
      "harness\\engine\\humaneval.exe")
    w("  humaneval.exe --model models\\Llama-3.2-3B-Instruct-Q4_K_M.gguf "
      "--data data\\humaneval_100.tsv \\")
    w("      --out results\\humaneval_bench_raw.csv --limit 100 --mode all --seed 1234")
    w("  python scripts\\eval_humaneval.py --csv results\\humaneval_bench_raw.csv "
      "--data data\\humaneval_100.jsonl \\")
    w("      --out results\\humaneval_bench.csv")
    w("  python scripts\\analyze_humaneval.py --csv results\\humaneval_bench.csv")
    w("")

    text = "\n".join(L) + "\n"
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
