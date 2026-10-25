#!/usr/bin/env python3
"""IFEval 512: baseline vs blind retry vs targeted retry.

This driver reuses the existing IFEval harness (harness/engine/ifeval.cpp),
the official strict checker in vendor/ifeval, and the evaluation/statistics
helpers in scripts/run_ifeval.py.

Protocol
--------
Baseline: all 541 official IFEval prompts, --max-tokens 512.
Retry trigger: any prompt whose baseline strict `follow_all` is false.
Blind retry: original prompt + the failed instruction descriptions only.
Targeted retry: original prompt + satisfied descriptions + failed
descriptions. Both retries use the same closing sentence, so the only
difference is the satisfied list.
Retry sampling: temp 0.6, top_k 20, top_p 0.95, max_tokens 512,
seed 1234 + prompt index + 10000 for both retry variants.
Final: the retry response for triggered prompts, baseline otherwise.

Outputs (default --tag ifeval_512):
    results/<tag>_baseline_raw.csv
    results/<tag>_blind_input.tsv
    results/<tag>_blind_raw.csv
    results/<tag>_targeted_input.tsv
    results/<tag>_targeted_raw.csv
    results/<tag>_retry_bench.csv
    results/<tag>_retry_summary.txt
    results/<tag>_retry_examples.txt
    results/<tag>_{baseline,blind,targeted}.log/.err

Usage:
    python scripts/run_ifeval_retry_compare.py --max-tokens 512
    python scripts/run_ifeval_retry_compare.py --reuse-baseline
    python scripts/run_ifeval_retry_compare.py --reuse-baseline --reuse-blind --reuse-targeted
    python scripts/run_ifeval_retry_compare.py --dry-run --reuse-baseline
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
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "vendor", "ifeval"))

import run_ifeval as ri  # noqa: E402
from instruction_following_eval import evaluation_lib  # noqa: E402
from instruction_following_eval import instructions_registry  # noqa: E402
from langdetect import DetectorFactory  # noqa: E402

DetectorFactory.seed = 0

FOCUS = [
    "language:response_language",
    "keywords:forbidden_words",
    "detectable_format:json_format",
    "length_constraints:number_paragraphs",
    "length_constraints:number_words",
]


def sha256(path):
    if not path or not os.path.exists(path):
        return "missing"
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def stat_line(name, st):
    if not st:
        return f"  {name:<18} n=0"
    return (f"  {name:<18} n={st['n']:<3} mean={st['mean']:8.1f}  "
            f"median={st['median']:8.1f}  p95={st['p95']:8.1f}  "
            f"max={st['max']:8.1f}  sum={st['sum']/1000.0:8.1f}s")


def git_head():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def log_wall_seconds(path):
    try:
        last = None
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith("rows=") and "wall_s=" in line:
                    last = line
        if last is None:
            return None
        return float(last.split("wall_s=")[1].split()[0])
    except Exception:
        return None


def describe_all(inp):
    """Official descriptions in instruction order, with a deterministic fallback."""
    random.seed(inp.key)
    out = []
    for idx, iid in enumerate(inp.instruction_id_list):
        kw = inp.kwargs[idx] or {}
        desc = None
        try:
            cls = instructions_registry.INSTRUCTION_DICT[iid]
            ins = cls(iid)
            desc = ins.build_description(**kw)
            args = ins.get_instruction_args()
            if args and "prompt" in args:
                desc2 = ins.build_description(prompt=inp.prompt)
                if isinstance(desc2, str) and desc2.strip():
                    desc = desc2
        except Exception:
            desc = None
        if not isinstance(desc, str) or not desc.strip():
            desc = (f"{iid} with kwargs {json.dumps(kw, sort_keys=True)} "
                    f"(original prompt: {inp.prompt})")
        out.append((iid, " ".join(desc.replace("\r", " ").replace("\n", " ").split())))
    return out


def make_retry_prompt(inp, descs, follow, include_satisfied):
    sat = [d for (_, d), ok in zip(descs, follow) if ok]
    fail = [d for (_, d), ok in zip(descs, follow) if not ok]
    parts = [inp.prompt.rstrip(), ""]
    if include_satisfied and sat:
        parts.append("Your previous response satisfied these instructions:")
        parts.extend(f"- {d}" for d in sat)
        parts.append("It failed these instructions:")
    else:
        parts.append("It failed these instructions:")
    parts.extend(f"- {d}" for d in fail)
    parts.append("Rewrite the response so it strictly satisfies the failed "
                 "instructions, WITHOUT breaking the satisfied ones.")
    return "\n".join(parts), sat, fail


def write_retry_input(path, inputs, base_follow, include_satisfied):
    prompts = {}
    meta = {}
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("key\tretry_prompt\tidx\n")
        for idx, (inp, fl) in enumerate(zip(inputs, base_follow)):
            if all(fl):
                continue
            descs = describe_all(inp)
            prompt, sat, fail = make_retry_prompt(inp, descs, fl, include_satisfied)
            k = str(inp.key)
            prompts[k] = prompt
            meta[k] = {
                "satisfied": sat,
                "failed": fail,
                "same_as_blind": not sat,
            }
            fh.write(f"{k}\t{ri.esc(prompt)}\t{idx}\n")
    return prompts, meta


def eval_responses(inputs, rows):
    out = []
    for inp in inputs:
        out.append(ri.check_strict(inp, rows[str(inp.key)]["response"]))
    return out


def follow_lists(outs):
    return [list(o.follow_instruction_list) for o in outs]


def pass_vector(outs):
    return [1 if o.follow_all_instructions else 0 for o in outs]


def instruction_totals(inputs, follow):
    tot, cor = {}, {}
    for inp, fl in zip(inputs, follow):
        for iid, ok in zip(inp.instruction_id_list, fl):
            tot[iid] = tot.get(iid, 0) + 1
            cor[iid] = cor.get(iid, 0) + (1 if ok else 0)
    return tot, cor


def flip_counts(inputs, base_follow, final_follow):
    rec = reg = 0
    by_id = {}
    for inp, bf, ff in zip(inputs, base_follow, final_follow):
        for iid, b, f in zip(inp.instruction_id_list, bf, ff):
            ent = by_id.setdefault(iid, {"fail_to_pass": 0, "pass_to_fail": 0})
            if not b and f:
                rec += 1
                ent["fail_to_pass"] += 1
            elif b and not f:
                reg += 1
                ent["pass_to_fail"] += 1
    return rec, reg, by_id


def run_exe(cmd, log_path, err_path, env):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    print("[run] " + " ".join(cmd), flush=True)
    t0 = time.perf_counter()
    with open(log_path, "w", encoding="utf-8", newline="\n") as out, \
            open(err_path, "w", encoding="utf-8", newline="\n") as err:
        p = subprocess.run(cmd, cwd=ROOT, env=env, stdout=out, stderr=err,
                           text=True, timeout=4 * 3600)
    wall = time.perf_counter() - t0
    if p.returncode != 0:
        raise SystemExit(f"command failed ({p.returncode}); see {log_path} / {err_path}")
    print(f"[run] done in {wall:.1f}s -> {log_path}", flush=True)
    return wall


def load_raw(path, keys):
    return ri.load_raw(path, keys)


def retry_cmd(exe, model, retry_input, raw_out, max_tokens, seed, seed_add):
    return [
        exe, "--model", model, "--retry-data", retry_input,
        "--out", raw_out, "--limit", str(len(
            [ln for ln in open(retry_input, encoding="utf-8") if ln.strip()]) - 1),
        "--max-tokens", str(max_tokens), "--seed", str(seed),
        "--seed-add", str(seed_add),
    ]


def baseline_cmd(exe, model, data, raw_out, max_tokens, seed):
    return [
        exe, "--model", model, "--data", data,
        "--out", raw_out, "--limit", "541",
        "--max-tokens", str(max_tokens), "--seed", str(seed),
    ]


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
    ap.add_argument("--tag", default="ifeval_512")
    ap.add_argument("--baseline-256-raw", default=os.path.join(
        ROOT, "results", "ifeval_baseline_raw.csv"))
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--retry-seed-add", type=int, default=10000)
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--bootstrap-seed", type=int, default=1234)
    ap.add_argument("--reuse-baseline", action="store_true")
    ap.add_argument("--reuse-blind", action="store_true")
    ap.add_argument("--reuse-targeted", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tag = args.tag
    os.makedirs(args.out_dir, exist_ok=True)
    baseline_raw = os.path.join(args.out_dir, f"{tag}_baseline_raw.csv")
    blind_input = os.path.join(args.out_dir, f"{tag}_blind_input.tsv")
    targeted_input = os.path.join(args.out_dir, f"{tag}_targeted_input.tsv")
    blind_raw = os.path.join(args.out_dir, f"{tag}_blind_raw.csv")
    targeted_raw = os.path.join(args.out_dir, f"{tag}_targeted_raw.csv")
    bench_csv = os.path.join(args.out_dir, f"{tag}_retry_bench.csv")
    summary_txt = os.path.join(args.out_dir, f"{tag}_retry_summary.txt")
    examples_txt = os.path.join(args.out_dir, f"{tag}_retry_examples.txt")
    baseline_log = os.path.join(args.out_dir, f"{tag}_baseline.log")
    blind_log = os.path.join(args.out_dir, f"{tag}_blind.log")
    targeted_log = os.path.join(args.out_dir, f"{tag}_targeted.log")
    blind_err = os.path.join(args.out_dir, f"{tag}_blind.err")
    targeted_err = os.path.join(args.out_dir, f"{tag}_targeted.err")

    inputs_all = evaluation_lib.read_prompt_list(args.official_data)
    if len(inputs_all) != 541:
        raise SystemExit(f"official data has {len(inputs_all)} prompts, expected 541")
    inputs = inputs_all
    keys = [str(inp.key) for inp in inputs]
    n = len(inputs)

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

    base_cmd = baseline_cmd(args.exe, args.model, args.data, baseline_raw,
                            args.max_tokens, args.seed)
    base_wall = None
    if args.reuse_baseline and os.path.exists(baseline_raw):
        print(f"[reuse] {baseline_raw}", flush=True)
    else:
        base_wall = run_exe(base_cmd, baseline_log, os.path.join(args.out_dir, f"{tag}_baseline.err"), env)

    raw = load_raw(baseline_raw, keys)
    base_out = eval_responses(inputs, raw)
    base_pass = pass_vector(base_out)
    base_follow = follow_lists(base_out)

    triggered = [i for i, ok in enumerate(base_pass) if not ok]
    triggered_keys = [keys[i] for i in triggered]
    print(f"[eval] baseline strict {sum(base_pass)}/{n}; triggered {len(triggered)}", flush=True)

    blind_prompts, blind_meta = write_retry_input(blind_input, inputs, base_follow, False)
    targeted_prompts, targeted_meta = write_retry_input(targeted_input, inputs, base_follow, True)
    if set(blind_prompts) != set(triggered_keys) or set(targeted_prompts) != set(triggered_keys):
        raise SystemExit("retry input key mismatch")
    n_same = sum(1 for k in targeted_meta if targeted_meta[k]["same_as_blind"])
    print(f"[retry-input] blind={len(blind_prompts)} targeted={len(targeted_prompts)} "
          f"targeted==blind (no satisfied instructions)={n_same}", flush=True)

    if args.dry_run:
        print("[dry-run] inputs written; not generating", flush=True)
        return

    blind_wall = targeted_wall = None
    if args.reuse_blind and os.path.exists(blind_raw):
        print(f"[reuse] {blind_raw}", flush=True)
    else:
        cmd = retry_cmd(args.exe, args.model, blind_input, blind_raw,
                        args.max_tokens, args.seed, args.retry_seed_add)
        blind_wall = run_exe(cmd, blind_log, blind_err, env)

    if args.reuse_targeted and os.path.exists(targeted_raw):
        print(f"[reuse] {targeted_raw}", flush=True)
    else:
        cmd = retry_cmd(args.exe, args.model, targeted_input, targeted_raw,
                        args.max_tokens, args.seed, args.retry_seed_add)
        targeted_wall = run_exe(cmd, targeted_log, targeted_err, env)

    blind_rows = load_raw(blind_raw, triggered_keys)
    targeted_rows = load_raw(targeted_raw, triggered_keys)

    blind_out = eval_responses([inputs[i] for i in triggered], blind_rows)
    targeted_out = eval_responses([inputs[i] for i in triggered], targeted_rows)
    blind_pass_trig = pass_vector(blind_out)
    targeted_pass_trig = pass_vector(targeted_out)
    blind_follow_trig = follow_lists(blind_out)
    targeted_follow_trig = follow_lists(targeted_out)

    blind_pass = list(base_pass)
    targeted_pass = list(base_pass)
    blind_follow = [list(x) for x in base_follow]
    targeted_follow = [list(x) for x in base_follow]
    for j, idx in enumerate(triggered):
        blind_pass[idx] = blind_pass_trig[j]
        targeted_pass[idx] = targeted_pass_trig[j]
        blind_follow[idx] = blind_follow_trig[j]
        targeted_follow[idx] = targeted_follow_trig[j]

    # ---------------- 256 baseline (optional, for the cap comparison) ----------------
    base256_pass = base256_follow = None
    if os.path.exists(args.baseline_256_raw):
        raw256 = load_raw(args.baseline_256_raw, keys)
        out256 = eval_responses(inputs, raw256)
        base256_pass = pass_vector(out256)
        base256_follow = follow_lists(out256)

    # ---------------- bench CSV ----------------
    with open(bench_csv, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "key", "idx", "prompt",
            "baseline_response", "baseline_follow_all", "baseline_follow_instruction_list",
            "baseline_finish", "baseline_n_tokens", "baseline_gen_ms", "baseline_total_ms",
            "triggered",
            "blind_prompt", "blind_response", "blind_follow_all",
            "blind_follow_instruction_list", "blind_finish", "blind_n_tokens",
            "blind_gen_ms", "blind_total_ms",
            "targeted_prompt", "targeted_response", "targeted_follow_all",
            "targeted_follow_instruction_list", "targeted_finish", "targeted_n_tokens",
            "targeted_gen_ms", "targeted_total_ms",
            "blind_final_follow_all", "targeted_final_follow_all",
            "blind_instruction_flips", "targeted_instruction_flips",
        ])
        for idx, inp in enumerate(inputs):
            k = str(inp.key)
            br = raw[k]
            bf = base_follow[idx]
            rr_b = blind_rows.get(k)
            rr_t = targeted_rows.get(k)
            b_flips = []
            t_flips = []
            if rr_b:
                for iid, b, f in zip(inp.instruction_id_list, bf, blind_follow[idx]):
                    if b != f:
                        b_flips.append(f"{iid}:{'pass->fail' if b else 'fail->pass'}")
                for iid, b, f in zip(inp.instruction_id_list, bf, targeted_follow[idx]):
                    if b != f:
                        t_flips.append(f"{iid}:{'pass->fail' if b else 'fail->pass'}")
            w.writerow([
                k, idx, inp.prompt,
                br["response"], base_pass[idx], json.dumps(bf),
                br.get("finish", ""), br.get("n_tokens", ""),
                br.get("gen_ms", ""), br.get("total_ms", ""),
                1 if rr_b else 0,
                blind_prompts.get(k, ""),
                rr_b["response"] if rr_b else "",
                blind_pass_trig[triggered.index(idx)] if rr_b else "",
                json.dumps(blind_follow[idx]),
                rr_b.get("finish", "") if rr_b else "",
                rr_b.get("n_tokens", "") if rr_b else "",
                rr_b.get("gen_ms", "") if rr_b else "",
                rr_b.get("total_ms", "") if rr_b else "",
                targeted_prompts.get(k, ""),
                rr_t["response"] if rr_t else "",
                targeted_pass_trig[triggered.index(idx)] if rr_t else "",
                json.dumps(targeted_follow[idx]),
                rr_t.get("finish", "") if rr_t else "",
                rr_t.get("n_tokens", "") if rr_t else "",
                rr_t.get("gen_ms", "") if rr_t else "",
                rr_t.get("total_ms", "") if rr_t else "",
                blind_pass[idx], targeted_pass[idx],
                json.dumps(b_flips), json.dumps(t_flips),
            ])

    # ---------------- statistics ----------------
    n_trig = len(triggered)
    b_acc = sum(base_pass) / n
    blind_acc = sum(blind_pass) / n
    targeted_acc = sum(targeted_pass) / n
    blind_trig_acc = sum(blind_pass_trig) / n_trig if n_trig else float("nan")
    targeted_trig_acc = sum(targeted_pass_trig) / n_trig if n_trig else float("nan")

    def prompt_compare(label, base_v, final_v):
        obs, lo, hi = ri.boot_delta(base_v, final_v, iters=args.bootstrap,
                                    seed=args.bootstrap_seed)
        n01, n10, p = ri.mcnemar_exact(base_v, final_v)
        return {
            "label": label,
            "base_acc": sum(base_v) / len(base_v),
            "final_acc": sum(final_v) / len(final_v),
            "delta": obs, "lo": lo, "hi": hi,
            "n01": n01, "n10": n10, "p": p,
            "base_ci": wilson(sum(base_v), len(base_v)),
            "final_ci": wilson(sum(final_v), len(final_v)),
        }

    cmp_blind_base = prompt_compare("blind vs baseline", base_pass, blind_pass)
    cmp_target_base = prompt_compare("targeted vs baseline", base_pass, targeted_pass)
    cmp_target_blind = prompt_compare("targeted vs blind", blind_pass, targeted_pass)

    treated_idx = [i for i in triggered
                   if targeted_prompts[keys[i]] != blind_prompts[keys[i]]]
    identical_idx = [i for i in triggered if i not in set(treated_idx)]
    treated_blind = sum(blind_pass[i] for i in treated_idx)
    treated_targ = sum(targeted_pass[i] for i in treated_idx)
    identical_pass = sum(blind_pass[i] for i in identical_idx)

    b_tot, b_cor = instruction_totals(inputs, base_follow)
    _, blind_cor = instruction_totals(inputs, blind_follow)
    _, targeted_cor = instruction_totals(inputs, targeted_follow)
    n_instr = sum(b_tot.values())
    b_instr = sum(b_cor.values()) / n_instr
    blind_instr = sum(blind_cor.values()) / n_instr
    targeted_instr = sum(targeted_cor.values()) / n_instr

    blind_rec, blind_reg, blind_by_id = flip_counts(inputs, base_follow, blind_follow)
    targ_rec, targ_reg, targ_by_id = flip_counts(inputs, base_follow, targeted_follow)
    tb_rec, tb_reg, tb_by_id = flip_counts(inputs, blind_follow, targeted_follow)

    # instruction slot-level McNemar (secondary)
    def slot_vectors(follow):
        b, f = [], []
        for inp, bf, ff in zip(inputs, base_follow, follow):
            for x, y in zip(bf, ff):
                b.append(1 if x else 0)
                f.append(1 if y else 0)
        return b, f

    blind_slots = slot_vectors(blind_follow)
    targeted_slots = slot_vectors(targeted_follow)
    blind_slot_mcn = ri.mcnemar_exact(blind_slots[0], blind_slots[1])
    targeted_slot_mcn = ri.mcnemar_exact(targeted_slots[0], targeted_slots[1])

    # truncation
    def n_max(rows_iter):
        return sum(1 for r in rows_iter if r.get("finish") == "max")

    base_max = n_max(raw.values())
    blind_max = n_max(blind_rows.values())
    targeted_max = n_max(targeted_rows.values())
    base256_max = None
    if os.path.exists(args.baseline_256_raw):
        raw256 = load_raw(args.baseline_256_raw, keys)
        base256_max = n_max(raw256.values())

    # latency
    base_tot = [float(raw[k]["total_ms"]) for k in keys]
    base_gen = [float(raw[k]["gen_ms"]) for k in keys]
    blind_gen = [float(blind_rows[k]["gen_ms"]) for k in triggered_keys]
    blind_tot = [float(blind_rows[k]["total_ms"]) for k in triggered_keys]
    targ_gen = [float(targeted_rows[k]["gen_ms"]) for k in triggered_keys]
    targ_tot = [float(targeted_rows[k]["total_ms"]) for k in triggered_keys]
    blind_pipeline = list(base_tot)
    targeted_pipeline = list(base_tot)
    for j, k in enumerate(triggered_keys):
        blind_pipeline[triggered[j]] += float(blind_rows[k]["total_ms"])
        targeted_pipeline[triggered[j]] += float(targeted_rows[k]["total_ms"])

    b_gen_st = lat_stats(base_gen)
    b_tot_st = lat_stats(base_tot)
    blind_gen_st = lat_stats(blind_gen)
    blind_tot_st = lat_stats(blind_tot)
    targ_gen_st = lat_stats(targ_gen)
    targ_tot_st = lat_stats(targ_tot)
    blind_pipe_st = lat_stats(blind_pipeline)
    targeted_pipe_st = lat_stats(targeted_pipeline)
    extra_blind = sum(blind_pipeline) - sum(base_tot)
    extra_targeted = sum(targeted_pipeline) - sum(base_tot)
    amort_blind = extra_blind / n
    amort_targeted = extra_targeted / n

    # ---------------- examples ----------------
    def clip(s, limit=1800):
        s = s.replace("\r\n", "\n")
        if len(s) <= limit:
            return s
        return s[:limit] + f"\n...[truncated {len(s) - limit} chars]"

    targ_pass_idx = [i for i in range(n) if not base_pass[i] and targeted_pass[i]]
    targ_fail_idx = [i for i in range(n) if not base_pass[i] and not targeted_pass[i]]
    target_only_pass = [i for i in targ_pass_idx if not blind_pass[i]]
    target_only_fail = [i for i in range(n) if blind_pass[i] and not targeted_pass[i]]
    target_reg_idx = []
    target_only_reg_idx = []
    for i in range(n):
        for iid, b, bl, t in zip(inputs[i].instruction_id_list, base_follow[i],
                                 blind_follow[i], targeted_follow[i]):
            if b and not t:
                target_reg_idx.append((i, iid))
                if bl:
                    target_only_reg_idx.append((i, iid))
                break
    blind_pass_target_fail = target_only_fail
    target_only_reg_slots = sum(
        1 for i in range(n)
        for b, bl, t in zip(base_follow[i], blind_follow[i], targeted_follow[i])
        if b and bl and not t)
    trunc_idx = [i for i in range(n) if raw[keys[i]].get("finish") == "max"]

    def example_block(i, title, extra=""):
        k = keys[i]
        inp = inputs[i]
        br = raw[k]
        rr_b = blind_rows.get(k)
        rr_t = targeted_rows.get(k)
        lines = [title, f"key={k} idx={i}"]
        lines.append(f"prompt:\n{clip(inp.prompt)}")
        lines.append(f"instructions={inp.instruction_id_list}")
        lines.append(f"baseline_follow_all={base_pass[i]} follow={base_follow[i]} "
                     f"finish={br.get('finish')} n_tokens={br.get('n_tokens')}")
        lines.append(f"baseline_response:\n{clip(br['response'])}")
        if rr_b:
            lines.append(f"blind_follow_all={blind_pass[i]} follow={blind_follow[i]} "
                         f"finish={rr_b.get('finish')} n_tokens={rr_b.get('n_tokens')}")
            lines.append(f"blind_response:\n{clip(rr_b['response'])}")
        if rr_t:
            lines.append(f"targeted_follow_all={targeted_pass[i]} follow={targeted_follow[i]} "
                         f"finish={rr_t.get('finish')} n_tokens={rr_t.get('n_tokens')}")
            lines.append(f"targeted_response:\n{clip(rr_t['response'])}")
        if extra:
            lines.append(extra)
        return "\n".join(lines)

    with open(examples_txt, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("IFEval 512 retry examples (official strict checker)\n")
        fh.write("=" * 78 + "\n\n")
        ex_pass = (target_only_pass[0] if target_only_pass
                   else (targ_pass_idx[0] if targ_pass_idx else None))
        ex_reg = (target_only_reg_idx[0] if target_only_reg_idx
                  else (target_reg_idx[0] if target_reg_idx else None))
        used = set()
        fh.write("TARGETED FAIL -> PASS (targeted-only where available)\n")
        fh.write("-" * 78 + "\n")
        if ex_pass is not None:
            used.add(keys[ex_pass])
            extra = ("blind_also_fails=1" if not blind_pass[ex_pass]
                     else "blind_also_passes=1")
            fh.write(example_block(ex_pass, "targeted fail -> pass", extra) + "\n\n")
        else:
            fh.write("none\n\n")
        fh.write("TARGETED PASS -> FAIL (instruction-level regression)\n")
        fh.write("-" * 78 + "\n")
        if ex_reg is not None:
            i, iid = ex_reg
            used.add(keys[i])
            blind_preserved = ex_reg in target_only_reg_idx
            extra = (f"regressed_instruction={iid}\n"
                     f"blind_preserved_instruction={1 if blind_preserved else 0}")
            fh.write(example_block(i, f"targeted regression: {iid}", extra) + "\n\n")
        else:
            fh.write("none\n\n")
        fh.write("TARGETED FAIL -> FAIL (residual)\n")
        fh.write("-" * 78 + "\n")
        if targ_fail_idx:
            fh.write(example_block(targ_fail_idx[0], "targeted fail -> fail") + "\n\n")
        else:
            fh.write("none\n\n")
        fh.write("BLIND PASS -> TARGETED FAIL\n")
        fh.write("-" * 78 + "\n")
        if blind_pass_target_fail:
            fh.write(example_block(blind_pass_target_fail[0],
                                   "blind pass -> targeted fail") + "\n\n")
        else:
            fh.write("none\n\n")
        fh.write("BASELINE TRUNCATION AT 512\n")
        fh.write("-" * 78 + "\n")
        trunc_choice = next((i for i in trunc_idx if keys[i] not in used),
                            (trunc_idx[0] if trunc_idx else None))
        if trunc_choice is not None:
            fh.write(example_block(trunc_choice, "baseline truncation at 512") + "\n\n")
        else:
            fh.write("none\n\n")

    # ---------------- summary ----------------
    lines = []
    lines.append("RecycleLM IFEval 512: baseline vs blind retry vs targeted retry")
    lines.append("=" * 78)
    lines.append("Verifier: official IFEval strict checker "
                 "(google-research instruction_following_eval)")
    lines.append(f"vendor/ifeval upstream commit 3ef52111c1fabc301880818f5a547570e926a76b")
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
    n_repl += sum(r["response"].count("\ufffd") for r in blind_rows.values())
    n_repl += sum(r["response"].count("\ufffd") for r in targeted_rows.values())
    lines.append(f"utf-8 replacement chars in raw responses: {n_repl} "
                 f"(incomplete trailing sequences are dropped by the harness)")
    lines.append(f"sampling: temp 0.6, top_k 20, top_p 0.95, max_tokens {args.max_tokens}, "
                 f"seed {args.seed}+idx (both retries +{args.retry_seed_add})")
    lines.append("")
    lines.append("Protocol")
    lines.append("-" * 78)
    lines.append("  baseline: one response per prompt, strict-checked on the original prompt.")
    lines.append("  retry trigger: baseline prompt strict `follow_all` is false.")
    lines.append("  blind retry: original prompt + failed instruction descriptions only.")
    lines.append("  targeted retry: original prompt + satisfied descriptions + failed descriptions.")
    lines.append("  both retries use the same closing sentence; the satisfied list is the only")
    lines.append("  treatment difference. The retry response replaces the baseline for the")
    lines.append("  final score on triggered prompts.")
    lines.append("")

    lines.append("256 vs 512 baseline")
    lines.append("-" * 78)
    if base256_pass is not None:
        b256 = sum(base256_pass) / n
        p256 = sum(base256_pass)
        p512 = sum(base_pass)
        d, lo, hi = ri.boot_delta(base256_pass, base_pass, iters=args.bootstrap,
                                  seed=args.bootstrap_seed)
        n01, n10, p = ri.mcnemar_exact(base256_pass, base_pass)
        lines.append(f"  prompt strict 256 : {pct(b256)} ({p256}/{n})  "
                     f"Wilson95 [{pct(wilson(p256, n)[0])}, {pct(wilson(p256, n)[1])}]")
        lines.append(f"  prompt strict 512 : {pct(b_acc)} ({p512}/{n})  "
                     f"Wilson95 [{pct(wilson(p512, n)[0])}, {pct(wilson(p512, n)[1])}]")
        lines.append(f"  delta 512-256     : {(b_acc-b256)*100:+.1f} pp  "
                     f"bootstrap 95% CI [{lo*100:+.1f}, {hi*100:+.1f}] pp")
        lines.append(f"  McNemar 512 vs 256: n01={n01} (256 fail -> 512 pass), "
                     f"n10={n10} (256 pass -> 512 fail), p={p:.3g}")
        _, c256 = instruction_totals(inputs, base256_follow)
        i256 = sum(c256.values()) / n_instr
        lines.append(f"  instruction strict 256: {pct(i256)} ({sum(c256.values())}/{n_instr})")
        lines.append(f"  instruction strict 512: {pct(b_instr)} ({sum(b_cor.values())}/{n_instr})")
        lines.append(f"  instruction delta     : {(b_instr-i256)*100:+.1f} pp")
        lines.append(f"  truncation at 256     : {base256_max}/{n} ({pct(base256_max/n)})")
        lines.append(f"  truncation at 512     : {base_max}/{n} ({pct(base_max/n)})")
        lines.append(f"  truncation drop       : {base_max-base256_max:+d} prompts "
                     f"({(base_max-base256_max)/n*100:+.1f} pp)")
        pref_total = sum(1 for k in keys if raw256[k].get("finish") == "max")
        pref_same = sum(1 for k in keys
                        if raw256[k].get("finish") == "max"
                        and raw[k]["response"].startswith(raw256[k]["response"]))
        lines.append(f"  prefix check          : {pref_same}/{pref_total} prompts truncated at "
                     f"256 have the 512 response starting with the 256 response exactly")
        lines.append("")
        lines.append("  Per-instruction baseline: 256 -> 512")
        lines.append(f"  {'instruction':<52} {'n':>4} {'256':>8} {'512':>8} {'delta':>8}")
        rows = []
        for iid in sorted(b_tot):
            tot = b_tot[iid]
            c2 = c256.get(iid, 0)
            c5 = b_cor.get(iid, 0)
            rows.append((iid, tot, c2 / tot, c5 / tot))
        rows.sort(key=lambda r: (r[2], -r[1], r[0]))
        for iid, tot, a2, a5 in rows:
            lines.append(f"  {iid:<52} {tot:>4} {pct(a2):>8} {pct(a5):>8} "
                         f"{(a5-a2)*100:>+7.1f}pp")
    else:
        lines.append(f"  256 baseline raw not found: {args.baseline_256_raw}")
    lines.append("")

    lines.append("Retry accounting")
    lines.append("-" * 78)
    lines.append(f"  baseline prompt failures (retry rate): {n_trig}/{n} "
                 f"({pct(n_trig/n)})")
    lines.append(f"  blind retry rows                     : {len(blind_rows)}/{n_trig}")
    lines.append(f"  targeted retry rows                  : {len(targeted_rows)}/{n_trig}")
    n_multi = sum(1 for inp in inputs if len(inp.instruction_id_list) > 1)
    n_multi_trig = sum(1 for i in triggered if len(inputs[i].instruction_id_list) > 1)
    lines.append(f"  multi-instruction prompts            : {n_multi}/{n} "
                 f"({pct(n_multi/n)}); triggered {n_multi_trig}/{n_trig}")
    lines.append(f"  targeted == blind (no satisfied list): {n_same}/{n_trig} "
                 f"({pct(n_same/n_trig) if n_trig else 'n/a'})")
    lines.append(f"  blind retry success                  : {sum(blind_pass_trig)}/{n_trig} "
                 f"({pct(blind_trig_acc)})")
    lines.append(f"  targeted retry success               : {sum(targeted_pass_trig)}/{n_trig} "
                 f"({pct(targeted_trig_acc)})")
    lines.append(f"  retry-success difference (targeted-blind): "
                 f"{(targeted_trig_acc-blind_trig_acc)*100:+.1f} pp")
    lines.append(f"  targeted-only prompt recoveries      : {len(target_only_pass)} "
                 f"(blind fail -> targeted pass)")
    lines.append(f"  targeted-only prompt regressions     : {len(target_only_fail)} "
                 f"(blind pass -> targeted fail)")
    lines.append(f"  blind truncation at cap              : {blind_max}/{len(blind_rows)} "
                 f"({pct(blind_max/len(blind_rows))})")
    lines.append(f"  targeted truncation at cap           : {targeted_max}/{len(targeted_rows)} "
                 f"({pct(targeted_max/len(targeted_rows))})")
    lines.append("")

    lines.append("Headline")
    lines.append("-" * 78)
    lines.append(f"  {'variant':<16} {'prompt strict':>22} {'triggered only':>18} "
                 f"{'instr strict':>18}")
    lines.append(f"  {'baseline':<16} {pct(b_acc) + f' ({sum(base_pass)}/{n})':>22} "
                 f"{pct(0.0) + f' (0/{n_trig})':>18} "
                 f"{pct(b_instr) + f' ({sum(b_cor.values())}/{n_instr})':>18}")
    lines.append(f"  {'blind final':<16} {pct(blind_acc) + f' ({sum(blind_pass)}/{n})':>22} "
                 f"{pct(blind_trig_acc) + f' ({sum(blind_pass_trig)}/{n_trig})':>18} "
                 f"{pct(blind_instr) + f' ({sum(blind_cor.values())}/{n_instr})':>18}")
    lines.append(f"  {'targeted final':<16} {pct(targeted_acc) + f' ({sum(targeted_pass)}/{n})':>22} "
                 f"{pct(targeted_trig_acc) + f' ({sum(targeted_pass_trig)}/{n_trig})':>18} "
                 f"{pct(targeted_instr) + f' ({sum(targeted_cor.values())}/{n_instr})':>18}")
    lines.append(f"  targeted - blind : {(targeted_acc-blind_acc)*100:+.1f} pp prompt, "
                 f"{(targeted_instr-blind_instr)*100:+.1f} pp instruction")
    lines.append("")

    lines.append("Prompt-level strict accuracy")
    lines.append("-" * 78)
    lines.append(f"  baseline          : {pct(b_acc)}  ({sum(base_pass)}/{n})  "
                 f"Wilson95 [{pct(wilson(sum(base_pass), n)[0])}, "
                 f"{pct(wilson(sum(base_pass), n)[1])}]")
    for c in (cmp_blind_base, cmp_target_base, cmp_target_blind):
        lines.append("")
        lines.append(f"  {c['label']}")
        lines.append(f"    base             : {pct(c['base_acc'])}  "
                     f"Wilson95 [{pct(c['base_ci'][0])}, {pct(c['base_ci'][1])}]")
        lines.append(f"    final            : {pct(c['final_acc'])}  "
                     f"({sum(blind_pass) if c['label'].startswith('blind') else sum(targeted_pass)}/{n})  "
                     f"Wilson95 [{pct(c['final_ci'][0])}, {pct(c['final_ci'][1])}]")
        lines.append(f"    delta vs base    : {c['delta']*100:+.1f} pp  "
                     f"bootstrap 95% CI [{c['lo']*100:+.1f}, {c['hi']*100:+.1f}] pp "
                     f"({args.bootstrap} resamples, seed {args.bootstrap_seed})")
        lines.append(f"    McNemar exact    : n01={c['n01']}, n10={c['n10']}, "
                     f"two-sided p={c['p']:.3g}")
    lines.append("")
    lines.append("  note: retries were generated only for baseline failures, so prompt-level")
    lines.append("  pass->fail against the baseline is impossible by construction. Instruction-")
    lines.append("  level pass->fail is still possible and is reported below.")
    lines.append("")

    lines.append("Targeted vs blind on the treated subset")
    lines.append("-" * 78)
    lines.append(f"  prompts with a satisfied list (treated): {len(treated_idx)}/{n_trig}")
    lines.append(f"  prompts with no satisfied list (identical prompts): "
                 f"{len(identical_idx)}/{n_trig}")
    lines.append(f"  treated subset, blind prompt strict    : {pct(treated_blind/len(treated_idx))} "
                 f"({treated_blind}/{len(treated_idx)})")
    lines.append(f"  treated subset, targeted prompt strict : {pct(treated_targ/len(treated_idx))} "
                 f"({treated_targ}/{len(treated_idx)})")
    lines.append(f"  treated subset delta                   : "
                 f"{(treated_targ-treated_blind)/len(treated_idx)*100:+.1f} pp")
    tbn01 = sum(1 for i in treated_idx if not blind_pass[i] and targeted_pass[i])
    tbn10 = sum(1 for i in treated_idx if blind_pass[i] and not targeted_pass[i])
    _, _, tbp = ri.mcnemar_exact([blind_pass[i] for i in treated_idx],
                                 [targeted_pass[i] for i in treated_idx])
    lines.append(f"  treated subset McNemar                 : n01={tbn01}, n10={tbn10}, "
                 f"p={tbp:.3g}")
    lines.append(f"  identical-prompt subset                : both "
                 f"{pct(identical_pass/len(identical_idx))} "
                 f"({identical_pass}/{len(identical_idx)}), zero discordant pairs")
    lines.append("")

    lines.append("Instruction-level strict accuracy")
    lines.append("-" * 78)
    lines.append(f"  baseline : {pct(b_instr)}  ({sum(b_cor.values())}/{n_instr})")
    lines.append(f"  blind    : {pct(blind_instr)}  ({sum(blind_cor.values())}/{n_instr})  "
                 f"delta {(blind_instr-b_instr)*100:+.1f} pp")
    lines.append(f"  targeted : {pct(targeted_instr)}  ({sum(targeted_cor.values())}/{n_instr})  "
                 f"delta {(targeted_instr-b_instr)*100:+.1f} pp")
    lines.append(f"  flips blind vs baseline    : fail->pass={blind_rec}, pass->fail={blind_reg}")
    lines.append(f"  flips targeted vs baseline : fail->pass={targ_rec}, pass->fail={targ_reg}")
    lines.append(f"  flips targeted vs blind    : fail->pass={tb_rec}, pass->fail={tb_reg}")
    lines.append(f"  targeted-only instruction regressions (baseline+blind pass -> targeted fail): "
                 f"{target_only_reg_slots} slots in {len(target_only_reg_idx)} prompts")
    lines.append(f"  slot McNemar blind vs base : n01={blind_slot_mcn[0]}, "
                 f"n10={blind_slot_mcn[1]}, p={blind_slot_mcn[2]:.3g}")
    lines.append(f"  slot McNemar targ vs base  : n01={targeted_slot_mcn[0]}, "
                 f"n10={targeted_slot_mcn[1]}, p={targeted_slot_mcn[2]:.3g}")
    lines.append("")
    lines.append("  Did the satisfied list reduce pass->fail regressions?")
    red = blind_reg - targ_reg
    lines.append(f"    blind pass->fail={blind_reg}, targeted pass->fail={targ_reg}, "
                 f"reduction={red} ({'yes' if red > 0 else 'no'})")
    lines.append("")
    lines.append("  Regressions still present (targeted vs baseline, pass->fail > 0)")
    any_reg = False
    for iid in sorted(targ_by_id):
        ent = targ_by_id[iid]
        if ent["pass_to_fail"]:
            any_reg = True
            lines.append(f"    {iid:<52} pass->fail={ent['pass_to_fail']} "
                         f"fail->pass={ent['fail_to_pass']}")
    if not any_reg:
        lines.append("    none")
    lines.append("")
    lines.append("  Regressions present (blind vs baseline, pass->fail > 0)")
    any_reg = False
    for iid in sorted(blind_by_id):
        ent = blind_by_id[iid]
        if ent["pass_to_fail"]:
            any_reg = True
            lines.append(f"    {iid:<52} pass->fail={ent['pass_to_fail']} "
                         f"fail->pass={ent['fail_to_pass']}")
    if not any_reg:
        lines.append("    none")
    lines.append("")
    lines.append("  Regressions targeted vs blind (pass->fail > 0)")
    any_reg = False
    for iid in sorted(tb_by_id):
        ent = tb_by_id[iid]
        if ent["pass_to_fail"]:
            any_reg = True
            lines.append(f"    {iid:<52} pass->fail={ent['pass_to_fail']} "
                         f"fail->pass={ent['fail_to_pass']}")
    if not any_reg:
        lines.append("    none")
    lines.append("")

    rows_i = []
    for iid in sorted(b_tot):
        tot = b_tot[iid]
        bc = b_cor.get(iid, 0)
        blc = blind_cor.get(iid, 0)
        tc = targeted_cor.get(iid, 0)
        rows_i.append((iid, tot, bc / tot, blc / tot, tc / tot))
    rows_i.sort(key=lambda r: (r[2], -r[1], r[0]))
    lines.append("Per-instruction breakdown (sorted by baseline accuracy)")
    lines.append("-" * 78)
    lines.append(f"  {'instruction':<52} {'n':>4} {'base':>8} {'blind':>8} "
                 f"{'target':>8} {'d_blind':>8} {'d_targ':>8}")
    for iid, tot, ba, bla, ta in rows_i:
        lines.append(f"  {iid:<52} {tot:>4} {pct(ba):>8} {pct(bla):>8} {pct(ta):>8} "
                     f"{(bla-ba)*100:>+7.1f}pp {(ta-ba)*100:>+7.1f}pp")
    lines.append("")
    lines.append("  Focus instructions")
    lines.append("-" * 78)
    focus_rows = [r for r in rows_i if r[0] in FOCUS]
    if not focus_rows:
        lines.append("  none")
    for iid, tot, ba, bla, ta in focus_rows:
        lines.append(f"  {iid:<52} n={tot:<4} base={pct(ba):>7} "
                     f"blind={pct(bla):>7} ({(bla-ba)*100:+.1f}pp) "
                     f"targeted={pct(ta):>7} ({(ta-ba)*100:+.1f}pp) "
                     f"targ-blind={(ta-bla)*100:+.1f}pp")
    lines.append("")

    lines.append("Per-family breakdown")
    lines.append("-" * 78)
    fam_tot, fam_b, fam_bl, fam_t = {}, {}, {}, {}
    for iid, tot, ba, bla, ta in rows_i:
        fam = iid.split(":")[0]
        fam_tot[fam] = fam_tot.get(fam, 0) + tot
        fam_b[fam] = fam_b.get(fam, 0) + round(ba * tot)
        fam_bl[fam] = fam_bl.get(fam, 0) + round(bla * tot)
        fam_t[fam] = fam_t.get(fam, 0) + round(ta * tot)
    for fam in sorted(fam_tot, key=lambda f: fam_b[f] / fam_tot[f]):
        tot = fam_tot[fam]
        lines.append(f"  {fam:<24} n={tot:<4} baseline={pct(fam_b[fam]/tot):>8} "
                     f"blind={pct(fam_bl[fam]/tot):>8} targeted={pct(fam_t[fam]/tot):>8}")
    lines.append("")

    lines.append("Truncation")
    lines.append("-" * 78)
    if base256_max is not None:
        lines.append(f"  256 baseline: {base256_max}/{n} ({pct(base256_max/n)})")
    lines.append(f"  512 baseline: {base_max}/{n} ({pct(base_max/n)})")
    lines.append(f"  512 blind retry: {blind_max}/{len(blind_rows)} "
                 f"({pct(blind_max/len(blind_rows))})")
    lines.append(f"  512 targeted retry: {targeted_max}/{len(targeted_rows)} "
                 f"({pct(targeted_max/len(targeted_rows))})")
    lines.append("")

    lines.append("Latency (ms, per prompt)")
    lines.append("-" * 78)
    lines.append(stat_line("baseline gen", b_gen_st))
    lines.append(stat_line("baseline total", b_tot_st))
    lines.append(stat_line("blind gen", blind_gen_st))
    lines.append(stat_line("blind total", blind_tot_st))
    lines.append(stat_line("targeted gen", targ_gen_st))
    lines.append(stat_line("targeted total", targ_tot_st))
    lines.append(stat_line("blind pipeline", blind_pipe_st))
    lines.append(stat_line("targeted pipeline", targeted_pipe_st))
    lines.append(f"  invocation wall (generated this run): baseline "
                 f"{fnum(base_wall) + 's' if base_wall is not None else 'reused'}, "
                 f"blind {fnum(blind_wall) + 's' if blind_wall is not None else 'reused'}, "
                 f"targeted {fnum(targeted_wall) + 's' if targeted_wall is not None else 'reused'}")
    if log_wall_seconds(baseline_log) is not None:
        lines.append(f"  harness log wall (observed generation): baseline "
                     f"{fnum(log_wall_seconds(baseline_log))}s, "
                     f"blind {fnum(log_wall_seconds(blind_log))}s, "
                     f"targeted {fnum(log_wall_seconds(targeted_log))}s")
    lines.append("")
    lines.append("  Cost of the retry (observed, not +34 ms)")
    lines.append(f"    blind retry mean gen/total per triggered prompt   : "
                 f"{fnum(blind_gen_st['mean'])} / {fnum(blind_tot_st['mean'])} ms")
    lines.append(f"    targeted retry mean gen/total per triggered prompt: "
                 f"{fnum(targ_gen_st['mean'])} / {fnum(targ_tot_st['mean'])} ms")
    lines.append(f"    blind extra over all {n} prompts                   : "
                 f"{extra_blind/1000.0:.1f}s ({amort_blind:.1f} ms/prompt)")
    lines.append(f"    targeted extra over all {n} prompts                : "
                 f"{extra_targeted/1000.0:.1f}s ({amort_targeted:.1f} ms/prompt)")
    lines.append("    A retry is a full extra generation. The amortized extra is the")
    lines.append("    actual measured pipeline cost, not a fixed +34 ms.")
    lines.append("")

    lines.append("Examples")
    lines.append("-" * 78)
    lines.append(f"  targeted fail -> pass: {len(targ_pass_idx)} "
                 f"(targeted-only, blind also fails: {len(target_only_pass)})")
    if ex_pass is not None:
        i = ex_pass
        lines.append(f"    key={keys[i]} baseline_follow={base_follow[i]} "
                     f"blind_follow={blind_follow[i]} targeted_follow={targeted_follow[i]}")
    lines.append(f"  targeted pass -> fail (instruction-level): {targ_reg} slots in "
                 f"{len(target_reg_idx)} prompts (targeted-only: {target_only_reg_slots} slots "
                 f"in {len(target_only_reg_idx)} prompts)")
    if ex_reg is not None:
        i, iid = ex_reg
        lines.append(f"    key={keys[i]} regressed={iid} "
                     f"baseline_follow={base_follow[i]} blind_follow={blind_follow[i]} "
                     f"targeted_follow={targeted_follow[i]}")
    lines.append(f"  baseline truncations at 512: {len(trunc_idx)}")
    if trunc_choice is not None:
        i = trunc_choice
        lines.append(f"    key={keys[i]} n_tokens={raw[keys[i]].get('n_tokens')} "
                     f"finish={raw[keys[i]].get('finish')}")
    lines.append("")
    lines.append(f"  Full examples: {os.path.relpath(examples_txt, ROOT)}")
    lines.append("")

    lines.append("Commands")
    lines.append("-" * 78)
    lines.append("  build: scripts\\build_tool.bat build\\cuda-graphs "
                 "harness\\engine\\ifeval.cpp harness\\engine\\ifeval.exe")
    lines.append("  baseline: " + " ".join(base_cmd))
    lines.append("  blind:    " + " ".join(retry_cmd(
        args.exe, args.model, blind_input, blind_raw, args.max_tokens,
        args.seed, args.retry_seed_add)))
    lines.append("  targeted: " + " ".join(retry_cmd(
        args.exe, args.model, targeted_input, targeted_raw, args.max_tokens,
        args.seed, args.retry_seed_add)))
    lines.append("  report:   python scripts\\run_ifeval_retry_compare.py "
                 "--reuse-baseline --reuse-blind --reuse-targeted")
    lines.append("")

    lines.append("Hashes")
    lines.append("-" * 78)
    for label, path in [
        ("baseline_raw", baseline_raw),
        ("blind_input", blind_input),
        ("blind_raw", blind_raw),
        ("targeted_input", targeted_input),
        ("targeted_raw", targeted_raw),
        ("bench_csv", bench_csv),
        ("examples", examples_txt),
    ]:
        lines.append(f"  {label:<16} {sha256(path)}")
    lines.append("")

    lines.append("Notes / bugs")
    lines.append("-" * 78)
    lines.append("  - Retry prompts use official instruction descriptions built with")
    lines.append("    each prompt's instruction_id_list and kwargs. If build_description")
    lines.append("    raises or returns empty, the fallback is the instruction id + kwargs +")
    lines.append("    the original prompt text.")
    lines.append("  - The blind and targeted prompts differ only by the satisfied-list block")
    lines.append("    (and, when a prompt has no satisfied instruction, they are identical).")
    lines.append("  - Both retry variants use the same seed_add, so the comparison is not")
    lines.append("    confounded by different sampling seeds.")
    lines.append("  - The official checker uses random fallbacks for invalid kwargs (e.g.")
    lines.append("    keywords:letter_frequency with letter='#'). We seed Python's random")
    lines.append("    per prompt (seed=key) and DetectorFactory.seed=0, then call the")
    lines.append("    official strict function unchanged.")
    lines.append("  - `length_constraints:nth_paragraph_first_word` and some")
    lines.append("    `length_constraints:number_words` prompts may be unachievable at 512")
    lines.append("    tokens; those failures are not retry-fixable.")
    lines.append("")

    with open(summary_txt, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")

    print("\n".join(lines[:90]), flush=True)
    print(f"\n[summary] {summary_txt}", flush=True)
    print(f"[bench]   {bench_csv}", flush=True)
    print(f"[examples]{examples_txt}", flush=True)


if __name__ == "__main__":
    main()
