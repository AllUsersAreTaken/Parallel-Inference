#!/usr/bin/env python3
"""Full Retry + User Turn on IFEval with the official deterministic checker.

Protocol
--------
Baseline: generate one response per official IFEval prompt (541). Sampling is
temp 0.6, top_k 20, top_p 0.95, seed 1234 + prompt index, <=256 new tokens.
Evaluate with the official strict checker from
vendor/ifeval/instruction_following_eval.
Retry: for every prompt whose strict `follow_all` is false, append the violated
instruction ids/descriptions to the original prompt and generate once
(seed 1234 + index + 10000). The retry response replaces the baseline response.
Final: retry response if a retry was produced, otherwise baseline.

Writes (default --tag ifeval):
    results/ifeval_baseline_raw.csv
    results/ifeval_retry_input.tsv
    results/ifeval_retry_raw.csv
    results/ifeval_bench.csv
    results/ifeval_summary.txt
    results/ifeval_examples.txt
    results/ifeval_baseline.log / ifeval_retry.log

Usage:
    python scripts/run_ifeval.py
    python scripts/run_ifeval.py --smoke --no-retry
    python scripts/run_ifeval.py --reuse-baseline --reuse-retry
"""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDOR = os.path.join(ROOT, "vendor", "ifeval")
sys.path.insert(0, VENDOR)

from instruction_following_eval import evaluation_lib  # noqa: E402
from instruction_following_eval import instructions_registry  # noqa: E402
from langdetect import DetectorFactory  # noqa: E402

# The official checker has random fallbacks when a kwarg is missing or invalid
# (e.g. keywords:letter_frequency with letter="#"). Seed both RNGs so reruns on
# the same responses agree. Per-prompt seeds are applied in check_strict().
DetectorFactory.seed = 0


# ---------------- helpers ----------------

def sha256(path):
    if not path or not os.path.exists(path):
        return "missing"
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def esc(s):
    out = []
    for ch in s:
        if ch == "\\":
            out.append("\\\\")
        elif ch == "\t":
            out.append("\\t")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        else:
            out.append(ch)
    return "".join(out)


def pct(x, digits=1):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{100.0 * x:.{digits}f}%"


def fnum(x, digits=1):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.{digits}f}"


def pctl(values, p):
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p / 100.0
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return s[lo]
    return s[lo] * (hi - k) + s[hi] * (k - lo)


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (center - half, center + half)


def boot_delta(base, final, iters=5000, seed=1234):
    n = len(base)
    rnd = random.Random(seed)
    obs = sum(final) / n - sum(base) / n
    vals = []
    for _ in range(iters):
        s = 0
        t = 0
        for _ in range(n):
            i = rnd.randrange(n)
            s += base[i]
            t += final[i]
        vals.append((t - s) / n)
    vals.sort()
    return obs, vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]


def mcnemar_exact(base, final):
    n01 = sum(1 for b, f in zip(base, final) if b == 0 and f == 1)
    n10 = sum(1 for b, f in zip(base, final) if b == 1 and f == 0)
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return n01, n10, min(1.0, 2.0 * tail)


def check_strict(inp, response):
    """Official strict check, with per-prompt seeds for the official RNG fallbacks."""
    random.seed(inp.key)
    return evaluation_lib.test_instruction_following_strict(
        inp, {inp.prompt: response})


def build_descriptions(inp):
    """Build every instruction description in official order, same RNG stream."""
    random.seed(inp.key)
    out = []
    for idx, iid in enumerate(inp.instruction_id_list):
        kw = inp.kwargs[idx]
        cls = instructions_registry.INSTRUCTION_DICT[iid]
        ins = cls(iid)
        desc = ins.build_description(**kw)
        args = ins.get_instruction_args()
        if args and "prompt" in args:
            desc2 = ins.build_description(prompt=inp.prompt)
            if isinstance(desc2, str) and desc2:
                desc = desc2
        if not isinstance(desc, str) or not desc:
            desc = iid
        out.append((iid, desc))
    return out


def violated(inp, follow_list):
    descs = build_descriptions(inp)
    return [descs[i] for i, ok in enumerate(follow_list) if not ok]


def run_exe(cmd, log_path, env):
    print("[run] " + " ".join(cmd), flush=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    t0 = time.perf_counter()
    with open(log_path, "w", encoding="utf-8", newline="\n") as log:
        p = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log,
                           stderr=subprocess.STDOUT, text=True, timeout=4 * 3600)
    wall = time.perf_counter() - t0
    if p.returncode != 0:
        raise SystemExit(f"command failed ({p.returncode}); see {log_path}")
    print(f"[run] done in {wall:.1f}s -> {log_path}", flush=True)
    return wall


def load_raw(path, expected_keys):
    rows = {}
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
        for r in csv.DictReader(fh):
            rows[str(r["key"])] = r
    if len(rows) != len(expected_keys):
        raise SystemExit(f"{path}: {len(rows)} rows, expected {len(expected_keys)}")
    missing = [k for k in expected_keys if k not in rows]
    extra = [k for k in rows if k not in set(expected_keys)]
    if missing or extra:
        raise SystemExit(f"{path}: key mismatch missing={missing[:3]} extra={extra[:3]}")
    return rows


def log_wall_seconds(path):
    """Read the harness's final `rows=... wall_s=...` line."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            last = None
            for line in fh:
                if line.startswith("rows=") and "wall_s=" in line:
                    last = line
        if last is None:
            return None
        return float(last.split("wall_s=")[1].split()[0])
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default=os.path.join(ROOT, "harness", "engine", "ifeval.exe"))
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "Llama-3.2-3B-Instruct-Q4_K_M.gguf"))
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "ifeval_541.tsv"))
    ap.add_argument("--meta", default=os.path.join(ROOT, "data", "ifeval_541_meta.json"))
    ap.add_argument("--official-data", default=os.path.join(
        ROOT, "vendor", "ifeval", "instruction_following_eval", "data", "input_data.jsonl"))
    ap.add_argument("--build-dir", default=os.path.join(ROOT, "build", "cuda-graphs"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "results"))
    ap.add_argument("--tag", default="ifeval")
    ap.add_argument("--limit", type=int, default=541)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-retry", action="store_true")
    ap.add_argument("--reuse-baseline", action="store_true")
    ap.add_argument("--reuse-retry", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--bootstrap-seed", type=int, default=1234)
    args = ap.parse_args()

    if args.smoke and args.limit == 541:
        args.limit = 10
    tag = args.tag
    if args.smoke and tag == "ifeval":
        tag = "ifeval_smoke"

    os.makedirs(args.out_dir, exist_ok=True)
    baseline_raw = os.path.join(args.out_dir, f"{tag}_baseline_raw.csv")
    retry_input = os.path.join(args.out_dir, f"{tag}_retry_input.tsv")
    retry_raw = os.path.join(args.out_dir, f"{tag}_retry_raw.csv")
    bench_csv = os.path.join(args.out_dir, f"{tag}_bench.csv")
    summary_txt = os.path.join(args.out_dir, f"{tag}_summary.txt")
    examples_txt = os.path.join(args.out_dir, f"{tag}_examples.txt")
    baseline_log = os.path.join(args.out_dir, f"{tag}_baseline.log")
    retry_log = os.path.join(args.out_dir, f"{tag}_retry.log")

    inputs_all = evaluation_lib.read_prompt_list(args.official_data)
    if len(inputs_all) != 541:
        raise SystemExit(f"official data has {len(inputs_all)} prompts, expected 541")
    inputs = inputs_all[: args.limit]
    keys = [str(inp.key) for inp in inputs]

    with open(args.meta, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    for inp in inputs:
        ent = meta["entries"].get(str(inp.key))
        if ent is None:
            raise SystemExit(f"meta missing key {inp.key}")
        if list(ent["instruction_id_list"]) != list(inp.instruction_id_list):
            raise SystemExit(f"meta instruction list mismatch for key {inp.key}")
        if list(ent["kwargs"]) != list(inp.kwargs):
            raise SystemExit(f"meta kwargs mismatch for key {inp.key}")

    env = dict(os.environ)
    env["PATH"] = os.path.join(args.build_dir, "bin") + os.pathsep + env.get("PATH", "")

    base_cmd = [
        args.exe, "--model", args.model, "--data", args.data,
        "--out", baseline_raw, "--limit", str(len(inputs)),
        "--max-tokens", str(args.max_tokens), "--seed", str(args.seed),
    ]
    base_wall = None
    if args.reuse_baseline and os.path.exists(baseline_raw):
        print(f"[reuse] {baseline_raw}", flush=True)
    else:
        base_wall = run_exe(base_cmd, baseline_log, env)

    raw = load_raw(baseline_raw, keys)

    # ---------------- baseline evaluation (official strict) ----------------
    base_out = [check_strict(inp, raw[str(inp.key)]["response"]) for inp in inputs]
    base_pass = [1 if o.follow_all_instructions else 0 for o in base_out]
    base_follow = [list(o.follow_instruction_list) for o in base_out]

    violated_by_key = {}
    retry_prompts = {}
    for inp, o in zip(inputs, base_out):
        if not o.follow_all_instructions:
            vio = violated(inp, o.follow_instruction_list)
            violated_by_key[str(inp.key)] = vio
            body = "; ".join(f"{iid} ({desc})" for iid, desc in vio)
            retry_prompts[str(inp.key)] = (
                f"{inp.prompt}\n\nYour previous response did not satisfy these "
                f"instructions: {body}. Rewrite your response so that it strictly "
                f"satisfies all instructions."
            )

    retry_raw_rows = {}
    retry_pass = {}
    retry_follow = {}
    retry_wall = None
    retry_cmd = []
    if not args.no_retry and retry_prompts:
        with open(retry_input, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("key\tretry_prompt\tidx\n")
            for idx, inp in enumerate(inputs):
                k = str(inp.key)
                if k in retry_prompts:
                    fh.write(f"{k}\t{esc(retry_prompts[k])}\t{idx}\n")
        retry_cmd = [
            args.exe, "--model", args.model, "--retry-data", retry_input,
            "--out", retry_raw, "--limit", str(len(retry_prompts)),
            "--max-tokens", str(args.max_tokens), "--seed", str(args.seed),
            "--seed-add", "10000",
        ]
        if args.reuse_retry and os.path.exists(retry_raw):
            print(f"[reuse] {retry_raw}", flush=True)
        else:
            retry_wall = run_exe(retry_cmd, retry_log, env)

        retry_raw_rows = load_raw(retry_raw, list(retry_prompts.keys()))
        for inp in inputs:
            k = str(inp.key)
            if k in retry_raw_rows:
                o = check_strict(inp, retry_raw_rows[k]["response"])
                retry_pass[k] = 1 if o.follow_all_instructions else 0
                retry_follow[k] = list(o.follow_instruction_list)

    # ---------------- final = retry if produced else baseline ----------------
    final_pass = []
    final_follow = []
    final_source = []
    final_response = []
    final_violated = []
    for inp, bpass, bfollow in zip(inputs, base_pass, base_follow):
        k = str(inp.key)
        if k in retry_raw_rows:
            final_source.append("retry")
            final_response.append(retry_raw_rows[k]["response"])
            final_pass.append(retry_pass[k])
            final_follow.append(retry_follow[k])
            final_violated.append(violated(inp, retry_follow[k]))
        else:
            final_source.append("baseline")
            final_response.append(raw[k]["response"])
            final_pass.append(bpass)
            final_follow.append(bfollow)
            final_violated.append(violated(inp, bfollow))

    n = len(inputs)
    n_retried = len(retry_raw_rows)

    # ---------------- bench CSV ----------------
    with open(bench_csv, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "key", "idx", "prompt", "baseline_response", "baseline_follow_all",
            "baseline_follow_instruction_list", "baseline_violated_ids",
            "retried", "retry_prompt", "retry_response", "retry_follow_all",
            "retry_follow_instruction_list", "retry_violated_ids",
            "final_source", "final_response", "final_follow_all",
            "final_follow_instruction_list", "final_violated_ids",
            "baseline_gen_ms", "baseline_total_ms", "retry_gen_ms",
            "retry_total_ms", "total_gen_ms",
        ])
        for idx, inp in enumerate(inputs):
            k = str(inp.key)
            br = raw[k]
            rr = retry_raw_rows.get(k)
            w.writerow([
                k, idx, inp.prompt, br["response"], base_pass[idx],
                json.dumps(base_follow[idx]),
                json.dumps([iid for iid, _ in violated_by_key.get(k, [])]),
                1 if k in retry_raw_rows else 0,
                retry_prompts.get(k, ""),
                rr["response"] if rr else "",
                retry_pass.get(k, ""),
                json.dumps(retry_follow[k]) if k in retry_follow else "",
                json.dumps([iid for iid, _ in final_violated[idx]]) if k in retry_raw_rows else "",
                final_source[idx], final_response[idx], final_pass[idx],
                json.dumps(final_follow[idx]),
                json.dumps([iid for iid, _ in final_violated[idx]]),
                br.get("gen_ms", ""), br.get("total_ms", ""),
                rr.get("gen_ms", "") if rr else "",
                rr.get("total_ms", "") if rr else "",
                (float(br["gen_ms"]) + (float(rr["gen_ms"]) if rr else 0.0)),
            ])

    # ---------------- statistics ----------------
    b_acc = sum(base_pass) / n
    f_acc = sum(final_pass) / n
    delta = f_acc - b_acc
    b_ci = wilson(sum(base_pass), n)
    f_ci = wilson(sum(final_pass), n)
    obs_delta, d_lo, d_hi = boot_delta(base_pass, final_pass,
                                       iters=args.bootstrap,
                                       seed=args.bootstrap_seed)
    n01, n10, mcn_p = mcnemar_exact(base_pass, final_pass)

    def instr_stats(follow_lists):
        tot = {}
        cor = {}
        for inp, fl in zip(inputs, follow_lists):
            for iid, ok in zip(inp.instruction_id_list, fl):
                tot[iid] = tot.get(iid, 0) + 1
                cor[iid] = cor.get(iid, 0) + (1 if ok else 0)
        return tot, cor

    b_tot, b_cor = instr_stats(base_follow)
    _, f_cor = instr_stats(final_follow)
    n_instr = sum(b_tot.values())
    b_instr_acc = sum(b_cor.values()) / n_instr
    f_instr_acc = sum(f_cor.values()) / n_instr

    latency = {
        "baseline": [],
        "retry": [],
        "total": [],
    }
    for idx, inp in enumerate(inputs):
        k = str(inp.key)
        br = raw[k]
        latency["baseline"].append((float(br["gen_ms"]), float(br["total_ms"])))
        if k in retry_raw_rows:
            rr = retry_raw_rows[k]
            latency["retry"].append((float(rr["gen_ms"]), float(rr["total_ms"])))
    for idx, inp in enumerate(inputs):
        k = str(inp.key)
        t = float(raw[k]["total_ms"])
        if k in retry_raw_rows:
            t += float(retry_raw_rows[k]["total_ms"])
        latency["total"].append(t)

    def lat_stats(vals):
        if not vals:
            return None
        return {
            "n": len(vals),
            "mean": statistics.mean(vals),
            "median": statistics.median(vals),
            "p95": pctl(vals, 95),
            "max": max(vals),
            "sum": sum(vals),
        }

    b_gen_st = lat_stats([x[0] for x in latency["baseline"]])
    b_tot_st = lat_stats([x[1] for x in latency["baseline"]])
    r_gen_st = lat_stats([x[0] for x in latency["retry"]])
    r_tot_st = lat_stats([x[1] for x in latency["retry"]])
    total_st = lat_stats(latency["total"])
    total_gen_st = lat_stats([
        x[0] + (float(retry_raw_rows[str(inp.key)]["gen_ms"])
                if str(inp.key) in retry_raw_rows else 0.0)
        for x, inp in zip(latency["baseline"], inputs)
    ])

    # ---------------- examples ----------------
    pass_examples = []
    fail_examples = []
    for idx, inp in enumerate(inputs):
        k = str(inp.key)
        if not base_pass[idx] and final_pass[idx] and k in retry_raw_rows:
            pass_examples.append(idx)
        elif not base_pass[idx] and not final_pass[idx]:
            fail_examples.append(idx)

    def clip(s, n=1400):
        s = s.replace("\r\n", "\n")
        if len(s) <= n:
            return s
        return s[:n] + f"\n...[truncated {len(s) - n} chars]"

    def example_block(idx):
        k = str(inp_key[idx])
        inp = inputs[idx]
        br = raw[k]
        rr = retry_raw_rows.get(k)
        lines = []
        lines.append(f"key={k} idx={idx}")
        lines.append(f"prompt:\n{clip(inp.prompt)}")
        lines.append(f"baseline_follow_all=0 violated={[i for i, _ in violated_by_key[k]]}")
        lines.append(f"baseline_response:\n{clip(br['response'])}")
        if rr:
            lines.append(f"retry_prompt:\n{clip(retry_prompts[k])}")
            lines.append(f"retry_response:\n{clip(rr['response'])}")
            lines.append(f"retry_follow_all={retry_pass[k]} violated={[i for i, _ in final_violated[idx]]}")
        else:
            lines.append("retry: none")
        lines.append(f"final={final_source[idx]} final_follow_all={final_pass[idx]}")
        return "\n".join(lines)

    inp_key = [str(inp.key) for inp in inputs]
    with open(examples_txt, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("IFEval retry examples (official strict checker)\n")
        fh.write("=" * 72 + "\n")
        fh.write("BASELINE FAIL -> RETRY PASS\n")
        fh.write("-" * 72 + "\n")
        if pass_examples:
            for idx in pass_examples[:3]:
                fh.write(example_block(idx) + "\n\n")
        else:
            fh.write("none\n\n")
        fh.write("BASELINE FAIL -> " +
                 ("RETRY FAIL\n" if retry_raw_rows else "NO RETRY PRODUCED\n"))
        fh.write("-" * 72 + "\n")
        if fail_examples:
            for idx in fail_examples[:3]:
                fh.write(example_block(idx) + "\n\n")
        else:
            fh.write("none\n\n")

    # ---------------- summary ----------------
    def git_head():
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        except Exception:
            return "unknown"

    def stat_line(name, st):
        if not st:
            return f"  {name:<10} n=0"
        return (f"  {name:<10} n={st['n']:<3} mean={st['mean']:8.1f}  "
                f"median={st['median']:8.1f}  p95={st['p95']:8.1f}  "
                f"max={st['max']:8.1f}  sum={st['sum']/1000.0:8.1f}s")

    lines = []
    lines.append("RecycleLM IFEval Full Retry + User Turn")
    lines.append("=" * 78)
    lines.append("Verifier: deterministic verifier (stronger than an LLM thinker)")
    lines.append("          official IFEval strict checker, "
                 "google-research instruction_following_eval")
    lines.append("          (vendor/ifeval, upstream commit "
                 "3ef52111c1fabc301880818f5a547570e926a76b)")
    lines.append("")
    lines.append(f"model: {os.path.relpath(args.model, ROOT)}")
    lines.append(f"  sha256: {sha256(args.model)}")
    lines.append(f"data:  {os.path.relpath(args.data, ROOT)}")
    lines.append(f"  sha256: {sha256(args.data)}")
    lines.append(f"meta:  {os.path.relpath(args.meta, ROOT)}")
    lines.append(f"  sha256: {sha256(args.meta)}")
    lines.append(f"official input_data.jsonl sha256: {sha256(args.official_data)}")
    lines.append(f"harness source sha256: {sha256(os.path.join(ROOT, 'harness', 'engine', 'ifeval.cpp'))}")
    lines.append(f"exe sha256: {sha256(args.exe)}")
    lines.append(f"git HEAD: {git_head()}")
    lines.append(f"n prompts: {n}")
    n_repl = sum(r["response"].count("\ufffd") for r in raw.values())
    n_repl += sum(r["response"].count("\ufffd") for r in retry_raw_rows.values())
    lines.append(f"utf-8 replacement chars in raw responses: {n_repl} "
                 f"(incomplete trailing sequences are dropped by the harness)")
    lines.append(f"sampling: temp 0.6, top_k 20, top_p 0.95, max_tokens {args.max_tokens}, "
                 f"seed {args.seed}+idx (retry +10000)")
    n_base_max = sum(r.get("finish") == "max" for r in raw.values())
    n_retry_max = sum(r.get("finish") == "max" for r in retry_raw_rows.values())
    retry_trunc = (f"{n_retry_max}/{n_retried} ({pct(n_retry_max / n_retried)})"
                   if n_retried else "0/0 (n/a)")
    lines.append(f"truncation at max_tokens: baseline {n_base_max}/{n} "
                 f"({pct(n_base_max / n)}), retry {retry_trunc}")
    lines.append("")

    lines.append("Prompt-level strict accuracy")
    lines.append("-" * 78)
    lines.append(f"  baseline follow_all : {pct(b_acc)}  ({sum(base_pass)}/{n})  "
                 f"Wilson95 [{pct(b_ci[0])}, {pct(b_ci[1])}]")
    lines.append(f"  final follow_all    : {pct(f_acc)}  ({sum(final_pass)}/{n})  "
                 f"Wilson95 [{pct(f_ci[0])}, {pct(f_ci[1])}]")
    lines.append(f"  retry gain          : {delta*100:+.1f} pp")
    lines.append(f"  bootstrap 95% CI    : [{d_lo*100:+.1f}, {d_hi*100:+.1f}] pp "
                 f"({args.bootstrap} resamples, seed {args.bootstrap_seed})")
    lines.append(f"  McNemar exact       : recoveries n01={n01}, regressions n10={n10}, "
                 f"two-sided p={mcn_p:.3g}")
    if n10 == 0:
        lines.append("  note: only baseline failures were retried, so regressions are "
                     "impossible by construction; p reflects n01.")
    lines.append("")

    n_candidates = len(retry_prompts)
    lines.append("Retry accounting")
    lines.append("-" * 78)
    lines.append(f"  baseline failures (retry trigger rate): {n_candidates}/{n} "
                 f"({pct(n_candidates / n)})")
    lines.append(f"  retry rows produced                   : {n_retried}/{n} "
                 f"({pct(n_retried / n)})")
    if n_retried:
        rec = sum(1 for idx in range(n) if not base_pass[idx] and final_pass[idx])
        lines.append(f"  retry success on triggered prompts     : {rec}/{n_retried} "
                     f"({pct(rec / n_retried)})")
    lines.append("")

    lines.append("Instruction-level strict accuracy")
    lines.append("-" * 78)
    lines.append(f"  baseline: {pct(b_instr_acc)}  ({sum(b_cor.values())}/{n_instr})")
    lines.append(f"  final   : {pct(f_instr_acc)}  ({sum(f_cor.values())}/{n_instr})")
    lines.append(f"  delta   : {(f_instr_acc - b_instr_acc)*100:+.1f} pp")
    i_rec = sum(1 for bf, ff in zip(base_follow, final_follow)
                for b, f in zip(bf, ff) if not b and f)
    i_reg = sum(1 for bf, ff in zip(base_follow, final_follow)
                for b, f in zip(bf, ff) if b and not f)
    lines.append(f"  instruction flips: fail->pass={i_rec}, pass->fail={i_reg} "
                 f"(a retry can break instructions that the baseline already satisfied)")
    lines.append("")

    rows_i = []
    for iid in sorted(b_tot):
        tot = b_tot[iid]
        bc = b_cor.get(iid, 0)
        fc = f_cor.get(iid, 0)
        rows_i.append((iid, tot, bc / tot, fc / tot, (fc - bc) / tot))
    rows_i.sort(key=lambda r: (r[2], -r[1], r[0]))
    lines.append("Per-instruction breakdown (sorted by baseline accuracy)")
    lines.append("-" * 78)
    lines.append(f"  {'instruction':<52} {'n':>4} {'base':>8} {'final':>8} {'delta':>8}")
    for iid, tot, ba, fa, dd in rows_i:
        lines.append(f"  {iid:<52} {tot:>4} {pct(ba):>8} {pct(fa):>8} {dd*100:>+7.1f}pp")
    lines.append("")
    bottom5 = sorted(rows_i, key=lambda r: (r[2], -r[1], r[0]))[:5]
    top5 = sorted(rows_i, key=lambda r: (-r[2], -r[1], r[0]))[:5]
    lines.append("  bottom 5 (baseline)")
    for iid, tot, ba, fa, dd in bottom5:
        lines.append(f"    {iid:<52} {pct(ba):>8} -> {pct(fa):>8}  (n={tot})")
    lines.append("  top 5 (baseline)")
    for iid, tot, ba, fa, dd in top5:
        lines.append(f"    {iid:<52} {pct(ba):>8} -> {pct(fa):>8}  (n={tot})")
    lines.append("")

    fam_tot, fam_cor_b, fam_cor_f = {}, {}, {}
    for iid, tot, ba, fa, dd in rows_i:
        fam = iid.split(":")[0]
        fam_tot[fam] = fam_tot.get(fam, 0) + tot
        fam_cor_b[fam] = fam_cor_b.get(fam, 0) + round(ba * tot)
        fam_cor_f[fam] = fam_cor_f.get(fam, 0) + round(fa * tot)
    rows_f = [(f, fam_tot[f], fam_cor_b[f] / fam_tot[f], fam_cor_f[f] / fam_tot[f])
              for f in fam_tot]
    rows_f.sort(key=lambda r: r[2])
    lines.append("Per-family breakdown")
    lines.append("-" * 78)
    for fam, tot, ba, fa in rows_f:
        lines.append(f"  {fam:<30} n={tot:<4} baseline={pct(ba):>8} final={pct(fa):>8} "
                     f"delta={(fa-ba)*100:>+7.1f}pp")
    lines.append("")

    lines.append("Latency (ms, per prompt)")
    lines.append("-" * 78)
    lines.append(stat_line("baseline gen", b_gen_st))
    lines.append(stat_line("baseline total", b_tot_st))
    lines.append(stat_line("retry gen", r_gen_st))
    lines.append(stat_line("retry total", r_tot_st))
    lines.append(stat_line("total gen", total_gen_st))
    lines.append(stat_line("total prompt", total_st))
    base_wall_obs = base_wall if base_wall is not None else log_wall_seconds(baseline_log)
    retry_wall_obs = retry_wall if retry_wall is not None else log_wall_seconds(retry_log)
    wall_b = "n/a" if base_wall_obs is None else f"{base_wall_obs:.1f}s"
    wall_r = "n/a" if retry_wall_obs is None else f"{retry_wall_obs:.1f}s"
    lines.append(f"  wall clock: baseline generation {wall_b}, retry generation {wall_r}")
    if r_tot_st:
        amort = total_st["mean"] - b_tot_st["mean"] if b_tot_st else float("nan")
        lines.append("")
        lines.append("  Claimed +34 ms check")
        lines.append(f"    retry mean total per retried prompt : {r_tot_st['mean']:.1f} ms")
        lines.append(f"    retry mean gen   per retried prompt : {r_gen_st['mean']:.1f} ms")
        lines.append(f"    amortized over all {n} prompts        : {amort:.1f} ms/prompt")
        verdict = "COSTS MORE" if r_tot_st["mean"] > 34 else "within 34 ms"
        lines.append(f"    verdict: {verdict} (a retry is a full extra generation, not 34 ms)")
    lines.append("")

    lines.append("Examples")
    lines.append("-" * 78)
    lines.append(f"  baseline fail -> retry pass: {len(pass_examples)}")
    for idx in pass_examples[:2]:
        k = inp_key[idx]
        lines.append(f"    key={k} violated={[i for i, _ in violated_by_key[k]]}")
        lines.append(f"      baseline: {clip(raw[k]['response'], 220).replace(chr(10), ' ')}")
        lines.append(f"      retry   : {clip(retry_raw_rows[k]['response'], 220).replace(chr(10), ' ')}")
    fail_label = "retry fail" if retry_raw_rows else "no retry produced"
    lines.append(f"  baseline fail -> {fail_label}: {len(fail_examples)}")
    for idx in fail_examples[:2]:
        k = inp_key[idx]
        lines.append(f"    key={k} residual violated={[i for i, _ in final_violated[idx]]}")
        text = retry_raw_rows[k]["response"] if k in retry_raw_rows else raw[k]["response"]
        label = "retry" if k in retry_raw_rows else "baseline"
        lines.append(f"      {label}: {clip(text, 220).replace(chr(10), ' ')}")
    lines.append("")
    lines.append(f"Full examples: {os.path.relpath(examples_txt, ROOT)}")

    lines.append("")
    lines.append("Commands")
    lines.append("-" * 78)
    lines.append("  build: scripts\\build_tool.bat build\\cuda-graphs "
                 "harness\\engine\\ifeval.cpp harness\\engine\\ifeval.exe")
    lines.append("  prep:  python scripts\\prep_ifeval.py")
    lines.append("  baseline: " + " ".join(base_cmd))
    if retry_cmd:
        lines.append("  retry:    " + " ".join(retry_cmd))
    lines.append("  report:   python scripts\\run_ifeval.py" +
                 (" --reuse-baseline --reuse-retry" if not args.no_retry else " --no-retry"))
    lines.append("")
    lines.append("Hashes")
    lines.append("-" * 78)
    for label, p in [
        ("baseline_raw", baseline_raw),
        ("retry_input", retry_input),
        ("retry_raw", retry_raw),
        ("bench_csv", bench_csv),
        ("examples", examples_txt),
    ]:
        lines.append(f"  {label:<14} {sha256(p)}")
    lines.append("")
    lines.append("Notes / bugs")
    lines.append("-" * 78)
    lines.append("  - The requested repo URL https://github.com/google-research/"
                 "instruction_following_eval does not exist (404).")
    lines.append("    Official code/data were sparse-checked out from "
                 "google-research/google-research/instruction_following_eval.")
    lines.append("  - nltk punkt_tab data was downloaded; langdetect 1.0.9 installed.")
    lines.append("  - The retry TSV has an optional third `idx` column so the retry "
                 "seed can be 1234+original_index+10000 even though the file contains "
                 "only failures.")
    lines.append("  - Determinism: the official checker has random fallbacks when a "
                 "kwarg is missing/invalid (e.g. keywords:letter_frequency with "
                 "letter='#'). We seed Python's random per prompt (seed=key) and "
                 "DetectorFactory.seed=0, then call the official strict function "
                 "unchanged. Reruns on the same responses agree exactly.")
    lines.append("  - Official-data bug: 2/33 keywords:letter_frequency prompts use a "
                 "non-letter (`#`, `!`), so the checker falls back to a random ASCII "
                 "letter instead of counting the requested character.")
    lines.append("  - The mandated 256-token cap makes some instructions unachievable: "
                 "`length_constraints:number_words` with at least 300 words needs far "
                 "more than 256 tokens. Those failures are not retry-fixable.")
    lines.append("  - No instruction checkers were skipped; all 25 instruction ids "
                 "evaluated with the official code.")
    lines.append("")

    with open(summary_txt, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")

    # console digest
    print("\n".join(lines[:52]), flush=True)
    print(f"\n[summary] {summary_txt}", flush=True)
    print(f"[bench]   {bench_csv}", flush=True)
    print(f"[examples]{examples_txt}", flush=True)


if __name__ == "__main__":
    main()
