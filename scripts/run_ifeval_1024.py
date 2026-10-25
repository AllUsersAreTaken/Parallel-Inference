#!/usr/bin/env python3
"""IFEval capstone at --max-tokens 1024: baseline vs blind retry, 3 seeds.

Reuses harness/engine/ifeval.cpp for generation, the official strict checker in
vendor/ifeval, and the evaluation/statistics/retry-prompt helpers in
scripts/run_ifeval.py and scripts/run_ifeval_retry_compare.py. The targeted
(satisfied-list) retry is deliberately not included: it was null at 512.

Protocol (per seed S in 1234, 5678, 9012)
-----------------------------------------
Baseline: all 541 official prompts, --max-tokens 1024 --seed S.
Retry trigger: baseline prompt strict `follow_all` is false.
Blind retry: original prompt + failed instruction descriptions only
    (same builder as the 512 blind arm), --max-tokens 1024 --seed S
    --seed-add 10000.
Final: retry if produced, else baseline.

Generation resumes from partial raw CSVs: if the harness was cut off mid-run,
only the missing keys are generated and the rows are merged back in data order.
Evaluated CSVs are flushed after every prompt.

Outputs
-------
    results/ifeval_1024_seed{S}_baseline.csv
    results/ifeval_1024_seed{S}_blind.csv
    results/ifeval_1024_summary.txt
    results/ifeval_1024_bench.csv
    plus raw/log artifacts: results/ifeval_1024_seed{S}_{baseline,blind}_raw.csv/.log

Usage
-----
    python scripts/run_ifeval_1024.py --seeds 1234
    python scripts/run_ifeval_1024.py --seeds 5678,9012
    python scripts/run_ifeval_1024.py --report-only --seeds 1234,5678,9012
"""

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "vendor", "ifeval"))

import run_ifeval as ri  # noqa: E402
from run_ifeval_retry_compare import describe_all, make_retry_prompt  # noqa: E402
from instruction_following_eval import evaluation_lib  # noqa: E402
from langdetect import DetectorFactory  # noqa: E402

DetectorFactory.seed = 0

RAW_FIELDS = ["key", "response", "prefill_ms", "gen_ms", "total_ms",
              "n_tokens", "idx", "seed", "finish"]

FOCUS = [
    "length_constraints:number_words",
    "length_constraints:number_paragraphs",
    "length_constraints:nth_paragraph_first_word",
    "detectable_format:json_format",
    "language:response_language",
    "keywords:forbidden_words",
]


# ---------------- small helpers ----------------

def pct(x, digits=1):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{100.0 * x:.{digits}f}%"


def ppts(x, digits=1):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x * 100.0:+.{digits}f} pp"


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


def lat_line(name, st):
    if not st:
        return f"  {name:<22} n=0"
    return (f"  {name:<22} n={st['n']:<4} mean={st['mean']:8.1f}  "
            f"median={st['median']:8.1f}  p95={st['p95']:8.1f}  "
            f"max={st['max']:8.1f}  sum={st['sum'] / 1000.0:8.1f}s")


def git_head():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


# ---------------- raw csv I/O (generation) ----------------

def read_raw(path):
    rows = {}
    if not path or not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh)
        try:
            for r in reader:
                if r.get("key") is None:
                    continue
                if any(r.get(f) is None for f in RAW_FIELDS):
                    continue
                rows[str(r["key"])] = r
        except csv.Error:
            # A harness killed mid-write can leave the last record incomplete;
            # keep every complete row and regenerate the missing keys.
            pass
    return rows


def write_raw(path, rows, order):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=RAW_FIELDS, lineterminator="\n",
                           extrasaction="ignore")
        w.writeheader()
        for k in order:
            w.writerow(rows[k])


def write_tsv(path, header, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(header)
        for k, prompt, idx in rows:
            fh.write(f"{k}\t{ri.esc(prompt)}\t{idx}\n")


def run_exe(cmd, log_path, env):
    print("[run] " + " ".join(cmd), flush=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    t0 = time.perf_counter()
    with open(log_path, "w", encoding="utf-8", newline="\n") as log:
        p = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log,
                           stderr=subprocess.STDOUT, text=True, timeout=6 * 3600)
    wall = time.perf_counter() - t0
    if p.returncode != 0:
        raise SystemExit(f"command failed ({p.returncode}); see {log_path}")
    print(f"[run] done in {wall:.1f}s -> {log_path}", flush=True)
    return wall


def run_phase(exe, model, raw_path, prompts, idx_of, max_tokens, seed,
              seed_add, mode, data_tsv, log_path, env):
    """Generate a phase, resuming any rows already present in raw_path."""
    order = [k for k in prompts]
    have = {k: v for k, v in read_raw(raw_path).items() if k in prompts}
    missing = [k for k in order if k not in have]
    if have:
        print(f"[resume] {raw_path}: {len(have)}/{len(order)} rows present, "
              f"{len(missing)} missing", flush=True)
    if missing:
        partial_tsv = raw_path + ".partial_input.tsv"
        partial_raw = raw_path + ".partial_raw.csv"
        write_tsv(partial_tsv, "key\tprompt\tidx\n",
                  [(k, prompts[k], idx_of[k]) for k in missing])
        flag = "--retry-data" if mode == "retry" else "--data"
        cmd = [exe, "--model", model, flag, partial_tsv,
               "--out", partial_raw, "--limit", str(len(missing)),
               "--max-tokens", str(max_tokens), "--seed", str(seed),
               "--seed-add", str(seed_add)]
        if mode == "baseline" and not have and data_tsv:
            os.remove(partial_tsv)
            cmd = [exe, "--model", model, "--data", data_tsv,
                   "--out", raw_path, "--limit", str(len(order)),
                   "--max-tokens", str(max_tokens), "--seed", str(seed)]
            run_exe(cmd, log_path, env)
            return read_raw(raw_path)
        run_exe(cmd, log_path, env)
        have.update(read_raw(partial_raw))
        write_raw(raw_path, have, order)
        for p in (partial_tsv, partial_raw):
            try:
                os.remove(p)
            except OSError:
                pass
    return {k: have[k] for k in order if k in have}


# ---------------- evaluated csv I/O ----------------

BASE_EVAL_FIELDS = [
    "key", "idx", "prompt", "instruction_id_list", "response", "follow_all",
    "follow_instruction_list", "failed_instruction_ids", "finish", "n_tokens",
    "gen_ms", "total_ms", "seed", "truncated",
]

BLIND_EVAL_FIELDS = [
    "key", "idx", "retry_prompt", "retry_response", "follow_all",
    "follow_instruction_list", "failed_instruction_ids", "finish", "n_tokens",
    "gen_ms", "total_ms", "seed", "truncated", "final_source",
    "final_follow_all",
]


def write_eval_csv(path, fields, rows, flush_every=25):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow(r)
            if i % flush_every == 0:
                fh.flush()


def read_eval_csv(path):
    rows = {}
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
        for r in csv.DictReader(fh):
            rows[str(r["key"])] = r
    return rows


def read_eval_order(path, order):
    rows = read_eval_csv(path)
    missing = [k for k in order if k not in rows]
    if missing:
        raise SystemExit(f"{path}: missing {len(missing)} keys, e.g. {missing[:3]}")
    return [rows[k] for k in order]


# ---------------- statistics ----------------

def instr_totals(inputs, follow):
    tot, cor = {}, {}
    for inp, fl in zip(inputs, follow):
        for iid, ok in zip(inp.instruction_id_list, fl):
            tot[iid] = tot.get(iid, 0) + 1
            cor[iid] = cor.get(iid, 0) + (1 if ok else 0)
    return tot, cor


def flips(base_follow, final_follow):
    rec = reg = 0
    for bf, ff in zip(base_follow, final_follow):
        for b, f in zip(bf, ff):
            if not b and f:
                rec += 1
            elif b and not f:
                reg += 1
    return rec, reg


def compare(base_v, final_v, iters, seed):
    obs, lo, hi = ri.boot_delta(base_v, final_v, iters=iters, seed=seed)
    n01, n10, p = ri.mcnemar_exact(base_v, final_v)
    return {"delta": obs, "lo": lo, "hi": hi, "n01": n01, "n10": n10, "p": p,
            "base_acc": sum(base_v) / len(base_v),
            "final_acc": sum(final_v) / len(final_v)}


def seed_stats(seed, inputs, base_rows, blind_rows, triggers, bootstrap,
               boot_seed, max_tokens, walls):
    keys = [str(inp.key) for inp in inputs]
    n = len(inputs)

    base_pass = [int(r["follow_all"]) for r in base_rows]
    base_follow = [json.loads(r["follow_instruction_list"]) for r in base_rows]

    blind_by_key = {str(r["key"]): r for r in blind_rows}
    final_pass, final_follow = [], []
    retry_pass, retry_follow = {}, {}
    for idx, inp in enumerate(inputs):
        k = keys[idx]
        r = blind_by_key.get(k)
        if r is not None:
            rp = int(r["follow_all"])
            retry_pass[k] = rp
            retry_follow[k] = json.loads(r["follow_instruction_list"])
            final_pass.append(rp)
            final_follow.append(list(retry_follow[k]))
        else:
            final_pass.append(base_pass[idx])
            final_follow.append(list(base_follow[idx]))

    n_trig = len(triggers)
    n_instr = sum(len(x) for x in base_follow)
    b_tot, b_cor = instr_totals(inputs, base_follow)
    _, f_cor = instr_totals(inputs, final_follow)
    b_instr = sum(b_cor.values()) / sum(b_tot.values())
    f_instr = sum(f_cor.values()) / sum(b_tot.values())

    cmp = compare(base_pass, final_pass, bootstrap, boot_seed)

    base_trunc = sum(1 for r in base_rows if r["finish"] == "max")
    base_trunc_fail = sum(1 for r in base_rows
                          if r["finish"] == "max" and int(r["follow_all"]) == 0)
    retry_trunc = sum(1 for r in blind_rows if r["finish"] == "max")
    retry_ok = sum(retry_pass.values())

    b_gen = [float(r["gen_ms"]) for r in base_rows]
    b_tot_ms = [float(r["total_ms"]) for r in base_rows]
    r_gen = [float(r["gen_ms"]) for r in blind_rows]
    r_tot_ms = [float(r["total_ms"]) for r in blind_rows]
    pipeline = list(b_tot_ms)
    for idx, inp in enumerate(inputs):
        r = blind_by_key.get(keys[idx])
        if r is not None:
            pipeline[idx] += float(r["total_ms"])
    pipe_st = lat_stats(pipeline)
    extra = sum(r_tot_ms)
    amort = extra / n

    # per-instruction table
    per_instr = []
    for iid in sorted(b_tot):
        tot = b_tot[iid]
        bc = b_cor.get(iid, 0)
        fc = f_cor.get(iid, 0)
        per_instr.append({
            "instruction": iid, "n": tot, "base": bc / tot,
            "final": fc / tot, "delta": (fc - bc) / tot,
        })
    rec, reg = flips(base_follow, final_follow)

    return {
        "seed": seed,
        "n": n,
        "n_instr": n_instr,
        "keys": keys,
        "inputs": inputs,
        "base_rows": base_rows,
        "base_pass": base_pass,
        "base_follow": base_follow,
        "blind_rows": blind_rows,
        "blind_by_key": blind_by_key,
        "final_pass": final_pass,
        "final_follow": final_follow,
        "retry_pass": retry_pass,
        "retry_follow": retry_follow,
        "cmp": cmp,
        "n_trig": n_trig,
        "n_retry": len(blind_rows),
        "retry_rate": n_trig / n,
        "retry_ok": retry_ok,
        "retry_success": retry_ok / n_trig if n_trig else float("nan"),
        "base_pass_n": sum(base_pass),
        "final_pass_n": sum(final_pass),
        "b_instr": b_instr,
        "f_instr": f_instr,
        "b_cor_n": sum(b_cor.values()),
        "f_cor_n": sum(f_cor.values()),
        "base_trunc": base_trunc,
        "base_trunc_fail": base_trunc_fail,
        "base_trunc_fail": base_trunc_fail,
        "retry_trunc": retry_trunc,
        "per_instr": per_instr,
        "instr_rec": rec,
        "instr_reg": reg,
        "b_gen": lat_stats(b_gen),
        "b_tot": lat_stats(b_tot_ms),
        "r_gen": lat_stats(r_gen),
        "r_tot": lat_stats(r_tot_ms),
        "pipe": pipe_st,
        "extra_s": extra / 1000.0,
        "amort_ms": amort,
        "walls": walls,
    }


# ---------------- examples ----------------

def clip(s, limit=2000):
    s = (s or "").replace("\r\n", "\n")
    if len(s) <= limit:
        return s
    return s[:limit] + f"\n...[truncated {len(s) - limit} chars]"


def example_block(st, idx, title):
    k = st["keys"][idx]
    inp = st["inputs"][idx]
    br = st["base_rows"][idx]
    rr = st["blind_by_key"].get(k)
    lines = [title, f"seed={st['seed']} key={k} idx={idx}",
             f"instructions={inp.instruction_id_list}"]
    lines.append(f"prompt:\n{clip(inp.prompt)}")
    lines.append(f"baseline_follow_all={st['base_pass'][idx]} "
                 f"follow={st['base_follow'][idx]} finish={br.get('finish')} "
                 f"n_tokens={br.get('n_tokens')}")
    lines.append(f"baseline_response:\n{clip(br['response'])}")
    if rr is not None:
        lines.append(f"retry_prompt:\n{clip(rr['retry_prompt'])}")
        lines.append(f"retry_follow_all={rr['follow_all']} "
                     f"follow={rr['follow_instruction_list']} "
                     f"finish={rr.get('finish')} n_tokens={rr.get('n_tokens')}")
        lines.append(f"retry_response:\n{clip(rr['retry_response'])}")
    else:
        lines.append("retry: none")
    return "\n".join(lines)


# ---------------- summary ----------------

def per_seed_block(st, max_tokens, bootstrap, boot_seed):
    n = st["n"]
    c = st["cmp"]
    L = []
    L.append(f"Seed {st['seed']}")
    L.append("-" * 78)
    L.append(f"  prompts                          : {n}")
    L.append(f"  baseline prompt strict           : {pct(st['base_pass_n'] / n)} "
             f"({st['base_pass_n']}/{n})  Wilson95 "
             f"[{pct(ri.wilson(st['base_pass_n'], n)[0])}, "
             f"{pct(ri.wilson(st['base_pass_n'], n)[1])}]")
    L.append(f"  final prompt strict              : {pct(st['final_pass_n'] / n)} "
             f"({st['final_pass_n']}/{n})  Wilson95 "
             f"[{pct(ri.wilson(st['final_pass_n'], n)[0])}, "
             f"{pct(ri.wilson(st['final_pass_n'], n)[1])}]")
    L.append(f"  prompt gain                      : {ppts(c['delta'])}  "
             f"bootstrap 95% CI [{c['lo'] * 100:+.1f}, {c['hi'] * 100:+.1f}] pp "
             f"({bootstrap} resamples, seed {boot_seed})")
    L.append(f"  McNemar exact                    : n01={c['n01']} (fail->pass), "
             f"n10={c['n10']} (pass->fail), two-sided p={c['p']:.3g}")
    L.append(f"  baseline instruction strict      : {pct(st['b_instr'])} "
             f"({st['b_cor_n']}/{st['n_instr']})")
    L.append(f"  final instruction strict         : {pct(st['f_instr'])} "
             f"({st['f_cor_n']}/{st['n_instr']})  "
             f"delta {ppts(st['f_instr'] - st['b_instr'])}")
    L.append(f"  instruction flips                : fail->pass={st['instr_rec']}, "
             f"pass->fail={st['instr_reg']}")
    L.append(f"  truncation at {max_tokens} (baseline)   : {st['base_trunc']}/{n} "
             f"({pct(st['base_trunc'] / n)})")
    L.append(f"  truncated baseline responses     : {st['base_trunc_fail']} failed, "
             f"{st['base_trunc'] - st['base_trunc_fail']} passed the checker")
    L.append(f"  truncation at {max_tokens} (blind retry): {st['retry_trunc']}/"
             f"{st['n_retry']} "
             f"({pct(st['retry_trunc'] / st['n_retry']) if st['n_retry'] else 'n/a'})")
    L.append(f"  retry rate (baseline failures)   : {st['n_trig']}/{n} "
             f"({pct(st['retry_rate'])})")
    L.append(f"  retry rows produced              : {st['n_retry']}/{st['n_trig']}")
    L.append(f"  retry success                    : {st['retry_ok']}/{st['n_trig']} "
             f"({pct(st['retry_success'])})")
    L.append("")
    L.append("  Per-instruction strict accuracy (sorted by baseline)")
    L.append(f"  {'instruction':<52} {'n':>4} {'base':>8} {'final':>8} {'delta':>8}")
    for r in sorted(st["per_instr"], key=lambda x: (x["base"], -x["n"], x["instruction"])):
        L.append(f"  {r['instruction']:<52} {r['n']:>4} {pct(r['base']):>8} "
                 f"{pct(r['final']):>8} {r['delta'] * 100:>+7.1f}pp")
    L.append("")
    L.append("  Focus instructions")
    for iid in FOCUS:
        r = next((x for x in st["per_instr"] if x["instruction"] == iid), None)
        if r is None:
            L.append(f"  {iid:<52} not present")
        else:
            L.append(f"  {iid:<52} n={r['n']:<4} base={pct(r['base']):>7} "
                     f"final={pct(r['final']):>7} ({r['delta'] * 100:+.1f}pp)")
    L.append("")
    deltas = sorted(st["per_instr"], key=lambda x: (-x["delta"], -x["n"], x["instruction"]))
    top = [r for r in deltas if r["delta"] > 0][:5]
    bottom = sorted([r for r in deltas if r["delta"] < 0],
                    key=lambda x: (x["delta"], -x["n"], x["instruction"]))[:5]
    L.append("  Top deltas (final - baseline)")
    if not top:
        L.append("    none positive")
    for r in top:
        L.append(f"    {r['instruction']:<52} {pct(r['base']):>7} -> "
                 f"{pct(r['final']):>7}  {r['delta'] * 100:+.1f}pp  (n={r['n']})")
    L.append("  Bottom deltas")
    if not bottom:
        L.append("    none negative")
    for r in bottom:
        L.append(f"    {r['instruction']:<52} {pct(r['base']):>7} -> "
                 f"{pct(r['final']):>7}  {r['delta'] * 100:+.1f}pp  (n={r['n']})")
    L.append("")
    L.append("  Latency (ms per prompt, measured)")
    L.append(lat_line("baseline gen", st["b_gen"]))
    L.append(lat_line("baseline total", st["b_tot"]))
    L.append(lat_line("blind retry gen", st["r_gen"]))
    L.append(lat_line("blind retry total", st["r_tot"]))
    L.append(lat_line("pipeline total", st["pipe"]))
    L.append(f"  retry extra over all {n} prompts : {st['extra_s']:.1f}s "
             f"({st['amort_ms']:.1f} ms/prompt) -- full extra generation, "
             f"not a fixed +34 ms")
    w = st["walls"]
    L.append(f"  wall clock (generation)          : baseline "
             f"{fnum(w.get('baseline'))}s, blind {fnum(w.get('blind'))}s")
    L.append("")
    return L


def pooled_stats(sts, bootstrap, boot_seed):
    base_v = [x for st in sts for x in st["base_pass"]]
    final_v = [x for st in sts for x in st["final_pass"]]
    cmp = compare(base_v, final_v, bootstrap, boot_seed)
    n = len(base_v)
    n_instr = sum(st["n_instr"] for st in sts)
    b_cor = sum(st["b_cor_n"] for st in sts)
    f_cor = sum(st["f_cor_n"] for st in sts)
    n_trig = sum(st["n_trig"] for st in sts)
    n_retry = sum(st["n_retry"] for st in sts)
    retry_ok = sum(sum(1 for v in st["retry_pass"].values() if v) for st in sts)
    base_trunc = sum(st["base_trunc"] for st in sts)
    base_trunc_fail = sum(st["base_trunc_fail"] for st in sts)
    retry_trunc = sum(st["retry_trunc"] for st in sts)
    per = {}
    for st in sts:
        for r in st["per_instr"]:
            e = per.setdefault(r["instruction"], {"n": 0, "b": 0.0, "f": 0.0})
            e["n"] += r["n"]
            e["b"] += r["base"] * r["n"]
            e["f"] += r["final"] * r["n"]
    per_instr = [{"instruction": i, "n": e["n"], "base": e["b"] / e["n"],
                  "final": e["f"] / e["n"], "delta": (e["f"] - e["b"]) / e["n"]}
                 for i, e in per.items()]
    lat = {
        "b_gen": [float(r["gen_ms"]) for st in sts for r in st["base_rows"]],
        "b_tot": [float(r["total_ms"]) for st in sts for r in st["base_rows"]],
        "r_gen": [float(r["gen_ms"]) for st in sts for r in st["blind_rows"]],
        "r_tot": [float(r["total_ms"]) for st in sts for r in st["blind_rows"]],
    }
    pipe = []
    extra = 0.0
    for st in sts:
        for r in st["base_rows"]:
            pipe.append(float(r["total_ms"]))
        for idx, r in enumerate(st["base_rows"]):
            k = st["keys"][idx]
            rr = st["blind_by_key"].get(k)
            if rr is not None:
                pipe[idx + len(pipe) - len(st["base_rows"])] += float(rr["total_ms"])
        for r in st["blind_rows"]:
            extra += float(r["total_ms"])
    return {
        "seeds": [st["seed"] for st in sts],
        "n": n,
        "n_base_slots": sum(st["n"] for st in sts),
        "base_pass_n": sum(base_v),
        "final_pass_n": sum(final_v),
        "cmp": cmp,
        "n_instr": n_instr,
        "b_instr": b_cor / n_instr,
        "f_instr": f_cor / n_instr,
        "b_cor_n": b_cor,
        "f_cor_n": f_cor,
        "n_trig": n_trig,
        "n_retry": n_retry,
        "retry_ok": retry_ok,
        "retry_rate": n_trig / n,
        "retry_success": retry_ok / n_retry if n_retry else float("nan"),
        "base_trunc": base_trunc,
        "base_trunc_fail": base_trunc_fail,
        "base_trunc_fail": base_trunc_fail,
        "retry_trunc": retry_trunc,
        "per_instr": per_instr,
        "b_gen": lat_stats(lat["b_gen"]),
        "b_tot": lat_stats(lat["b_tot"]),
        "r_gen": lat_stats(lat["r_gen"]),
        "r_tot": lat_stats(lat["r_tot"]),
        "pipe": lat_stats(pipe),
        "extra_s": extra / 1000.0,
        "amort_ms": extra / n,
    }


def pooled_block(p):
    n = p["n"]
    c = p["cmp"]
    L = []
    L.append(f"Pooled over seeds {', '.join(str(s) for s in p['seeds'])} "
             f"(DESCRIPTIVE: the same 541 prompts repeat across seeds, so units "
             f"are not independent)")
    L.append("-" * 78)
    L.append(f"  prompt-seed units                : {n}")
    L.append(f"  baseline prompt strict           : {pct(p['base_pass_n'] / n)} "
             f"({p['base_pass_n']}/{n})")
    L.append(f"  final prompt strict              : {pct(p['final_pass_n'] / n)} "
             f"({p['final_pass_n']}/{n})")
    L.append(f"  prompt gain                      : {ppts(c['delta'])}  "
             f"bootstrap 95% CI [{c['lo'] * 100:+.1f}, {c['hi'] * 100:+.1f}] pp")
    L.append(f"  McNemar exact                    : n01={c['n01']}, n10={c['n10']}, "
             f"two-sided p={c['p']:.3g} (assumes independent units; see note)")
    L.append(f"  baseline instruction strict      : {pct(p['b_instr'])} "
             f"({p['b_cor_n']}/{p['n_instr']})")
    L.append(f"  final instruction strict         : {pct(p['f_instr'])} "
             f"({p['f_cor_n']}/{p['n_instr']})  "
             f"delta {ppts(p['f_instr'] - p['b_instr'])}")
    L.append(f"  truncation at 1024 (baseline)    : {p['base_trunc']}/{p['n_base_slots']} "
             f"({pct(p['base_trunc'] / p['n_base_slots'])})")
    L.append(f"  truncated baseline responses     : {p['base_trunc_fail']} failed, "
             f"{p['base_trunc'] - p['base_trunc_fail']} passed the checker")
    L.append(f"  truncation at 1024 (blind retry) : {p['retry_trunc']}/{p['n_retry']} "
             f"({pct(p['retry_trunc'] / p['n_retry']) if p['n_retry'] else 'n/a'})")
    L.append(f"  retry rate                       : {p['n_trig']}/{p['n_base_slots']} "
             f"({pct(p['retry_rate'])})")
    L.append(f"  retry success                    : {p['retry_ok']}/{p['n_retry']} "
             f"({pct(p['retry_success'])})")
    L.append("")
    L.append("  Focus instructions (pooled)")
    L.append(f"  {'instruction':<52} {'n':>5} {'base':>8} {'final':>8} {'delta':>8}")
    for iid in FOCUS:
        r = next((x for x in p["per_instr"] if x["instruction"] == iid), None)
        if r is None:
            L.append(f"  {iid:<52} not present")
        else:
            L.append(f"  {iid:<52} {r['n']:>5} {pct(r['base']):>8} "
                     f"{pct(r['final']):>8} {r['delta'] * 100:>+7.1f}pp")
    L.append("")
    L.append("  Latency (ms per prompt-seed unit, measured)")
    L.append(lat_line("baseline gen", p["b_gen"]))
    L.append(lat_line("baseline total", p["b_tot"]))
    L.append(lat_line("blind retry gen", p["r_gen"]))
    L.append(lat_line("blind retry total", p["r_tot"]))
    L.append(lat_line("pipeline total", p["pipe"]))
    L.append(f"  retry extra over {p['n_base_slots']} baseline prompts: "
             f"{p['extra_s']:.1f}s ({p['amort_ms']:.1f} ms/prompt)")
    L.append("")
    return L


def ref_512_block(args, inputs, max_tokens):
    tag = os.path.join(args.out_dir, "ifeval_512")
    raw_b = tag + "_baseline_raw.csv"
    raw_r = tag + "_blind_raw.csv"
    if not (os.path.exists(raw_b) and os.path.exists(raw_r)):
        return ["Reference 512 (seed 1234): artifacts not found; skipped", ""]
    keys = [str(inp.key) for inp in inputs]
    try:
        have = read_raw(raw_r)
        b = ri.load_raw(raw_b, keys)
        r = ri.load_raw(raw_r, [k for k in keys if k in have])
    except SystemExit as exc:
        return [f"Reference 512 (seed 1234): cannot load ({exc}); skipped", ""]
    base_pass, final_pass = [], []
    b_trunc = 0
    for inp in inputs:
        k = str(inp.key)
        o = ri.check_strict(inp, b[k]["response"])
        bp = 1 if o.follow_all_instructions else 0
        base_pass.append(bp)
        b_trunc += 1 if b[k].get("finish") == "max" else 0
        if not bp and k in r:
            o2 = ri.check_strict(inp, r[k]["response"])
            final_pass.append(1 if o2.follow_all_instructions else 0)
        else:
            final_pass.append(bp)
    gain = sum(final_pass) / len(final_pass) - sum(base_pass) / len(base_pass)
    L = [f"Reference: existing 512 artifacts, seed 1234 (re-checked here)",
         "-" * 78,
         f"  baseline prompt strict : {pct(sum(base_pass) / len(base_pass))} "
         f"({sum(base_pass)}/{len(base_pass)})",
         f"  blind final            : {pct(sum(final_pass) / len(final_pass))} "
         f"({sum(final_pass)}/{len(final_pass)})",
         f"  blind gain             : {ppts(gain)}",
         f"  truncation at 512      : {b_trunc}/{len(base_pass)} "
         f"({pct(b_trunc / len(base_pass))})",
         ""]
    return L


def write_summary(args, sts, pending, inputs, hashes):
    max_tokens = args.max_tokens
    lines = []
    lines.append("RecycleLM IFEval capstone: 1024-token baseline + blind retry")
    lines.append("=" * 78)
    lines.append("Verifier: official IFEval strict checker "
                 "(google-research instruction_following_eval)")
    lines.append("vendor/ifeval upstream commit "
                 "3ef52111c1fabc301880818f5a547570e926a76b")
    lines.append("")
    lines.append(f"model: {os.path.relpath(args.model, ROOT)}")
    lines.append(f"  sha256: {hashes['model']}")
    lines.append(f"data:  {os.path.relpath(args.data, ROOT)}")
    lines.append(f"  sha256: {hashes['data']}")
    lines.append(f"meta:  {os.path.relpath(args.meta, ROOT)}")
    lines.append(f"  sha256: {hashes['meta']}")
    lines.append(f"official input_data.jsonl sha256: {hashes['official']}")
    lines.append("harness source sha256: "
                 f"{hashes['harness'] if 'harness' in hashes else ri.sha256(os.path.join(ROOT, 'harness', 'engine', 'ifeval.cpp'))}")
    lines.append(f"exe sha256: {hashes['exe'] if 'exe' in hashes else ri.sha256(args.exe)}")
    lines.append(f"git HEAD: {git_head()}")
    lines.append(f"n prompts per seed: {len(inputs)}")
    n_repl = sum(r["response"].count("\ufffd") for st in sts for r in st["base_rows"])
    n_repl += sum(r["retry_response"].count("\ufffd")
                  for st in sts for r in st["blind_rows"])
    lines.append(f"utf-8 replacement chars in evaluated responses: {n_repl} "
                 f"(incomplete trailing sequences are dropped by the harness)")
    lines.append(f"sampling: temp 0.6, top_k 20, top_p 0.95, max_tokens {max_tokens}, "
                 f"seed S+idx; blind retry seed S+idx+{args.retry_seed_add}")
    lines.append(f"seeds completed: {', '.join(str(s['seed']) for s in sts) if sts else 'none'}")
    if pending:
        lines.append(f"seeds PENDING (not run): {', '.join(str(s) for s in pending)}")
    lines.append("")
    lines.append("Protocol")
    lines.append("-" * 78)
    lines.append("  one baseline response per prompt at the original prompt.")
    lines.append("  retry trigger: baseline prompt strict `follow_all` is false.")
    lines.append("  blind retry: original prompt + failed instruction descriptions only;")
    lines.append("    no satisfied list, no target list. Same builder as the 512 blind arm.")
    lines.append("  final = retry response if produced, else baseline. No targeted arm.")
    lines.append("")

    for st in sts:
        lines += per_seed_block(st, max_tokens, args.bootstrap, args.bootstrap_seed)

    if len(sts) > 1:
        lines += pooled_block(pooled_stats(sts, args.bootstrap, args.bootstrap_seed))
    elif sts:
        lines.append("Pooled")
        lines.append("-" * 78)
        lines.append("  pending: only one seed completed so far; pooled numbers need the")
        lines.append("  remaining seeds and are descriptive (prompts repeat across seeds).")
        lines.append("")

    if 1234 in [s["seed"] for s in sts]:
        lines += ref_512_block(args, inputs, max_tokens)

    # truncation verdict, only from observed numbers
    lines.append("Truncation check")
    lines.append("-" * 78)
    slots = sum(st["n"] for st in sts)
    trunc = sum(st["base_trunc"] for st in sts)
    trunc_fail = sum(st["base_trunc_fail"] for st in sts)
    fails = sum(st["n"] - st["base_pass_n"] for st in sts)
    if slots:
        lines.append(f"  baseline truncation at {max_tokens}: {trunc}/{slots} "
                     f"({pct(trunc / slots)}) across completed seeds")
        lines.append(f"  of the {trunc} truncated baseline responses, {trunc_fail} failed "
                     f"the strict check and {trunc - trunc_fail} passed")
        lines.append(f"  truncated responses account for {trunc_fail}/{fails} baseline "
                     f"prompt failures ({pct(trunc_fail / fails) if fails else 'n/a'})")
        if trunc == 0:
            lines.append("  1024 removes the cap as a source of failure: no baseline")
            lines.append("  response reached max_tokens in these runs.")
        elif trunc_fail == 0:
            lines.append(f"  the cap still binds ({trunc} responses at max_tokens) but no")
            lines.append("  truncated response failed the checker in these runs.")
        else:
            lines.append(f"  {trunc_fail} truncated responses still fail at "
                         f"{max_tokens}; the cap therefore still")
            lines.append("  binds for those prompts, but it is no longer the dominant")
            lines.append("  failure mode.")
    else:
        lines.append("  no completed seeds")
    lines.append("")

    # examples
    lines.append("Examples")
    lines.append("-" * 78)
    ex_pass = ex_fail = None
    for st in sts:
        if ex_pass is None:
            for idx, inp in enumerate(st["inputs"]):
                k = st["keys"][idx]
                r = st["blind_by_key"].get(k)
                if r is not None and not st["base_pass"][idx] and int(r["follow_all"]):
                    ex_pass = (st, idx)
                    break
        if ex_fail is None:
            for idx, inp in enumerate(st["inputs"]):
                k = st["keys"][idx]
                r = st["blind_by_key"].get(k)
                if r is not None and not st["base_pass"][idx] and not int(r["follow_all"]):
                    ex_fail = (st, idx)
                    break
    if ex_pass:
        lines.append(example_block(ex_pass[0], ex_pass[1], "FAIL -> PASS") + "\n")
    else:
        lines.append("FAIL -> PASS: none in completed seeds\n")
    if ex_fail:
        lines.append(example_block(ex_fail[0], ex_fail[1], "FAIL -> FAIL") + "\n")
    else:
        lines.append("FAIL -> FAIL: none in completed seeds\n")

    lines.append("Commands")
    lines.append("-" * 78)
    lines.append("  build: scripts\\build_tool.bat build\\cuda-graphs "
                 "harness\\engine\\ifeval.cpp harness\\engine\\ifeval.exe")
    for st in sts:
        s = st["seed"]
        lines.append(f"  seed {s} baseline: {args.exe} --model {args.model} "
                     f"--data {args.data} --out "
                     f"results\\ifeval_1024_seed{s}_baseline_raw.csv --limit 541 "
                     f"--max-tokens {max_tokens} --seed {s}")
        lines.append(f"  seed {s} blind   : {args.exe} --model {args.model} "
                     f"--retry-data results\\ifeval_1024_seed{s}_blind_input.tsv "
                     f"--out results\\ifeval_1024_seed{s}_blind_raw.csv "
                     f"--limit {st['n_trig']} --max-tokens {max_tokens} "
                     f"--seed {s} --seed-add {args.retry_seed_add}")
    lines.append("  report: python scripts\\run_ifeval_1024.py --report-only")
    lines.append("")
    lines.append("Artifacts")
    lines.append("-" * 78)
    for st in sts:
        s = st["seed"]
        for label, name in [
            ("baseline_csv", f"ifeval_1024_seed{s}_baseline.csv"),
            ("blind_csv", f"ifeval_1024_seed{s}_blind.csv"),
            ("baseline_raw", f"ifeval_1024_seed{s}_baseline_raw.csv"),
            ("blind_raw", f"ifeval_1024_seed{s}_blind_raw.csv"),
            ("blind_input", f"ifeval_1024_seed{s}_blind_input.tsv"),
            ("baseline_log", f"ifeval_1024_seed{s}_baseline.log"),
            ("blind_log", f"ifeval_1024_seed{s}_blind.log"),
        ]:
            p = os.path.join(args.out_dir, name)
            lines.append(f"  seed {s} {label:<12} {ri.sha256(p)}  {name}")
    lines.append(f"  bench_csv           {ri.sha256(os.path.join(args.out_dir, 'ifeval_1024_bench.csv'))}")
    lines.append("")
    lines.append("Notes")
    lines.append("-" * 78)
    lines.append("  - Reference 512 numbers above are recomputed from the existing")
    lines.append("    results/ifeval_512_{baseline,blind}_raw.csv (seed 1234) with the")
    lines.append("    same checker and helpers; they are not copied from a text file.")
    lines.append("  - Retry prompts use official instruction descriptions built from each")
    lines.append("    prompt's instruction_id_list and kwargs (same builder as 512).")
    lines.append("  - Retries are generated only for baseline failures, so prompt-level")
    lines.append("    pass->fail against the baseline is impossible by construction.")
    lines.append("  - Prompt-level bootstrap resamples prompt units within a seed.")
    lines.append("    Pooled resampling treats the 3*541 prompt-seed units as independent,")
    lines.append("    which they are not (the prompts repeat); pooled is descriptive.")
    lines.append("  - The official checker has random kwarg fallbacks; RNGs are seeded per")
    lines.append("    prompt (random.seed(key), DetectorFactory.seed=0) before the")
    lines.append("    official strict function is called unchanged.")
    lines.append("")

    path = os.path.join(args.out_dir, "ifeval_1024_summary.txt")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    return path, lines


def write_bench(args, sts):
    path = os.path.join(args.out_dir, "ifeval_1024_bench.csv")
    fields = [
        "seed", "n_prompts", "n_instr", "baseline_prompt_strict", "baseline_prompt_pct",
        "final_prompt_strict", "final_prompt_pct", "gain_pp", "ci_lo_pp", "ci_hi_pp",
        "mcnemar_n01", "mcnemar_n10", "mcnemar_p", "baseline_instr_strict",
        "baseline_instr_pct", "final_instr_strict", "final_instr_pct", "instr_delta_pp",
        "baseline_trunc", "baseline_trunc_pct", "baseline_trunc_fail", "retry_rows", "retry_trunc",
        "retry_trunc_pct", "retry_rate", "retry_success", "baseline_gen_mean_ms",
        "baseline_total_mean_ms", "retry_gen_mean_ms", "retry_total_mean_ms",
        "pipeline_total_mean_ms", "extra_s", "amortized_extra_ms",
        "baseline_wall_s", "retry_wall_s",
    ]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(fields)
        for st in sts:
            c = st["cmp"]
            n = st["n"]
            w.writerow([
                st["seed"], n, st["n_instr"], st["base_pass_n"],
                st["base_pass_n"] / n, st["final_pass_n"], st["final_pass_n"] / n,
                c["delta"] * 100, c["lo"] * 100, c["hi"] * 100,
                c["n01"], c["n10"], c["p"], st["b_cor_n"], st["b_instr"],
                st["f_cor_n"], st["f_instr"], (st["f_instr"] - st["b_instr"]) * 100,
                st["base_trunc"], st["base_trunc"] / n, st["base_trunc_fail"], st["n_retry"],
                st["retry_trunc"],
                (st["retry_trunc"] / st["n_retry"]) if st["n_retry"] else "",
                st["retry_rate"], st["retry_success"],
                st["b_gen"]["mean"], st["b_tot"]["mean"], st["r_gen"]["mean"],
                st["r_tot"]["mean"], st["pipe"]["mean"], st["extra_s"],
                st["amort_ms"], st["walls"].get("baseline", ""),
                st["walls"].get("blind", ""),
            ])
        if sts:
            p = pooled_stats(sts, args.bootstrap, args.bootstrap_seed)
            c = p["cmp"]
            w.writerow([
                "pooled", p["n"], p["n_instr"], p["base_pass_n"],
                p["base_pass_n"] / p["n"], p["final_pass_n"],
                p["final_pass_n"] / p["n"], c["delta"] * 100, c["lo"] * 100,
                c["hi"] * 100, c["n01"], c["n10"], c["p"], p["b_cor_n"],
                p["b_instr"], p["f_cor_n"], p["f_instr"],
                (p["f_instr"] - p["b_instr"]) * 100, p["base_trunc"],
                p["base_trunc"] / p["n_base_slots"], p["base_trunc_fail"], p["n_retry"], p["retry_trunc"],
                (p["retry_trunc"] / p["n_retry"]) if p["n_retry"] else "",
                p["retry_rate"], p["retry_success"], p["b_gen"]["mean"],
                p["b_tot"]["mean"], p["r_gen"]["mean"], p["r_tot"]["mean"],
                p["pipe"]["mean"], p["extra_s"], p["amort_ms"], "", "",
            ])
    return path


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default=os.path.join(ROOT, "harness", "engine", "ifeval.exe"))
    ap.add_argument("--model", default=os.path.join(
        ROOT, "models", "Llama-3.2-3B-Instruct-Q4_K_M.gguf"))
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "ifeval_541.tsv"))
    ap.add_argument("--meta", default=os.path.join(ROOT, "data", "ifeval_541_meta.json"))
    ap.add_argument("--official-data", default=os.path.join(
        ROOT, "vendor", "ifeval", "instruction_following_eval", "data",
        "input_data.jsonl"))
    ap.add_argument("--build-dir", default=os.path.join(ROOT, "build", "cuda-graphs"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "results"))
    ap.add_argument("--seeds", default="1234,5678,9012")
    ap.add_argument("--planned-seeds", default="1234,5678,9012")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--retry-seed-add", type=int, default=10000)
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--bootstrap-seed", type=int, default=1234)
    ap.add_argument("--reuse", action="store_true",
                    help="reuse evaluated CSVs if present")
    ap.add_argument("--report-only", action="store_true",
                    help="do not generate; rebuild summary from evaluated CSVs")
    ap.add_argument("--no-report", action="store_true",
                    help="skip summary/bench writing")
    ap.add_argument("--smoke-limit", type=int, default=0,
                    help="DEBUG: limit prompts (0 = all 541); output is not official")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    planned = [int(s) for s in args.planned_seeds.split(",") if s.strip()]
    os.makedirs(args.out_dir, exist_ok=True)

    inputs_all = evaluation_lib.read_prompt_list(args.official_data)
    if len(inputs_all) != 541:
        raise SystemExit(f"official data has {len(inputs_all)} prompts, expected 541")
    inputs = inputs_all
    if args.smoke_limit:
        inputs = inputs[: args.smoke_limit]
        print(f"[SMOKE] --smoke-limit {args.smoke_limit}: output is not official",
              flush=True)
    keys = [str(inp.key) for inp in inputs]
    idx_of = {k: i for i, k in enumerate(keys)}
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

    hashes = {
        "model": ri.sha256(args.model),
        "data": ri.sha256(args.data),
        "meta": ri.sha256(args.meta),
        "official": ri.sha256(args.official_data),
        "harness": ri.sha256(os.path.join(ROOT, "harness", "engine", "ifeval.cpp")),
        "exe": ri.sha256(args.exe),
    }

    def pending_seeds():
        out = []
        for s in planned:
            if s in [x["seed"] for x in sts]:
                continue
            if os.path.exists(os.path.join(
                    args.out_dir, f"ifeval_1024_seed{s}_baseline.csv")) and \
                    os.path.exists(os.path.join(
                        args.out_dir, f"ifeval_1024_seed{s}_blind.csv")):
                continue
            out.append(s)
        return out

    def load_seed(seed):
        base_csv = os.path.join(args.out_dir, f"ifeval_1024_seed{seed}_baseline.csv")
        blind_csv = os.path.join(args.out_dir, f"ifeval_1024_seed{seed}_blind.csv")
        base_rows = read_eval_order(base_csv, keys)
        blind_by_key = read_eval_csv(blind_csv)
        triggers = [int(r["idx"]) for r in base_rows if int(r["follow_all"]) == 0]
        blind_rows = sorted(blind_by_key.values(), key=lambda r: int(r["idx"]))
        walls = {
            "baseline": ri.log_wall_seconds(os.path.join(
                args.out_dir, f"ifeval_1024_seed{seed}_baseline.log")),
            "blind": ri.log_wall_seconds(os.path.join(
                args.out_dir, f"ifeval_1024_seed{seed}_blind.log")),
        }
        return seed_stats(seed, inputs, base_rows, blind_rows, triggers,
                          args.bootstrap, args.bootstrap_seed, args.max_tokens,
                          walls)

    sts = []
    for seed in seeds:
        print(f"\n[seed {seed}] start", flush=True)
        base_csv = os.path.join(args.out_dir, f"ifeval_1024_seed{seed}_baseline.csv")
        blind_csv = os.path.join(args.out_dir, f"ifeval_1024_seed{seed}_blind.csv")
        base_raw = os.path.join(args.out_dir, f"ifeval_1024_seed{seed}_baseline_raw.csv")
        blind_input = os.path.join(args.out_dir, f"ifeval_1024_seed{seed}_blind_input.tsv")
        blind_raw = os.path.join(args.out_dir, f"ifeval_1024_seed{seed}_blind_raw.csv")
        base_log = os.path.join(args.out_dir, f"ifeval_1024_seed{seed}_baseline.log")
        blind_log = os.path.join(args.out_dir, f"ifeval_1024_seed{seed}_blind.log")
        walls = {}

        reuse_eval = (args.reuse or args.report_only) and \
            os.path.exists(base_csv) and os.path.exists(blind_csv)

        if reuse_eval:
            print(f"[reuse] {base_csv}\n[reuse] {blind_csv}", flush=True)
            st = load_seed(seed)
            sts.append(st)
            for line in per_seed_block(st, args.max_tokens, args.bootstrap,
                                       args.bootstrap_seed):
                print(line, flush=True)
            continue

        if args.report_only:
            raise SystemExit(f"missing evaluated CSVs for seed {seed}; "
                             f"cannot --report-only")

        # ---- baseline generation (resumes partial raw) ----
        prompts = {str(inp.key): inp.prompt for inp in inputs}
        base_raw_rows = run_phase(
            args.exe, args.model, base_raw, prompts, idx_of, args.max_tokens,
            seed, 0, "baseline", args.data, base_log, env)
        missing = [k for k in keys if k not in base_raw_rows]
        if missing:
            raise SystemExit(f"seed {seed}: baseline incomplete ({len(missing)} missing)")
        walls["baseline"] = ri.log_wall_seconds(base_log)
        base_raw_rows = {k: base_raw_rows[k] for k in keys}

        base_out = [ri.check_strict(inp, base_raw_rows[k]["response"])
                    for inp, k in zip(inputs, keys)]
        base_pass = [1 if o.follow_all_instructions else 0 for o in base_out]
        base_follow = [list(o.follow_instruction_list) for o in base_out]
        base_rows = []
        for idx, (inp, k) in enumerate(zip(inputs, keys)):
            r = base_raw_rows[k]
            base_rows.append({
                "key": k, "idx": idx, "prompt": inp.prompt,
                "instruction_id_list": json.dumps(inp.instruction_id_list),
                "response": r["response"], "follow_all": base_pass[idx],
                "follow_instruction_list": json.dumps(base_follow[idx]),
                "failed_instruction_ids": json.dumps(
                    [iid for iid, ok in zip(inp.instruction_id_list, base_follow[idx])
                     if not ok]),
                "finish": r.get("finish", ""), "n_tokens": r.get("n_tokens", ""),
                "gen_ms": r.get("gen_ms", ""), "total_ms": r.get("total_ms", ""),
                "seed": r.get("seed", ""),
                "truncated": 1 if r.get("finish") == "max" else 0,
            })
        write_eval_csv(base_csv, BASE_EVAL_FIELDS, base_rows)
        base_rows = read_eval_order(base_csv, keys)
        triggers = [i for i, ok in enumerate(base_pass) if not ok]
        print(f"[eval] seed {seed} baseline strict {sum(base_pass)}/{n}; "
              f"triggered {len(triggers)}", flush=True)

        # ---- blind retry ----
        retry_prompts = {}
        for idx in triggers:
            inp = inputs[idx]
            descs = describe_all(inp)
            prompt, _sat, _fail = make_retry_prompt(
                inp, descs, base_follow[idx], include_satisfied=False)
            retry_prompts[str(inp.key)] = prompt
        write_tsv(blind_input, "key\tretry_prompt\tidx\n",
                  [(str(inp.key), retry_prompts[str(inp.key)], idx)
                   for idx, inp in enumerate(inputs) if str(inp.key) in retry_prompts])
        print(f"[retry-input] seed {seed}: {len(retry_prompts)} prompts", flush=True)

        blind_raw_rows = run_phase(
            args.exe, args.model, blind_raw, retry_prompts, idx_of,
            args.max_tokens, seed, args.retry_seed_add, "retry", None,
            blind_log, env)
        missing = [k for k in retry_prompts if k not in blind_raw_rows]
        if missing:
            raise SystemExit(f"seed {seed}: blind retry incomplete "
                             f"({len(missing)} missing)")
        walls["blind"] = ri.log_wall_seconds(blind_log)

        blind_rows = []
        for idx in triggers:
            inp = inputs[idx]
            k = str(inp.key)
            r = blind_raw_rows[k]
            o = ri.check_strict(inp, r["response"])
            fl = list(o.follow_instruction_list)
            blind_rows.append({
                "key": k, "idx": idx,
                "retry_prompt": retry_prompts[k],
                "retry_response": r["response"],
                "follow_all": 1 if o.follow_all_instructions else 0,
                "follow_instruction_list": json.dumps(fl),
                "failed_instruction_ids": json.dumps(
                    [iid for iid, ok in zip(inp.instruction_id_list, fl) if not ok]),
                "finish": r.get("finish", ""), "n_tokens": r.get("n_tokens", ""),
                "gen_ms": r.get("gen_ms", ""), "total_ms": r.get("total_ms", ""),
                "seed": r.get("seed", ""),
                "truncated": 1 if r.get("finish") == "max" else 0,
                "final_source": "retry",
                "final_follow_all": 1 if o.follow_all_instructions else 0,
            })
        write_eval_csv(blind_csv, BLIND_EVAL_FIELDS, blind_rows)
        blind_rows = sorted(read_eval_csv(blind_csv).values(),
                            key=lambda r: int(r["idx"]))
        print(f"[eval] seed {seed} blind strict "
              f"{sum(int(r['follow_all']) for r in blind_rows)}/{len(blind_rows)}",
              flush=True)

        st = seed_stats(seed, inputs, base_rows, blind_rows, triggers,
                        args.bootstrap, args.bootstrap_seed, args.max_tokens, walls)
        sts.append(st)
        for line in per_seed_block(st, args.max_tokens, args.bootstrap,
                                   args.bootstrap_seed):
            print(line, flush=True)

        if not args.no_report:
            path, lines = write_summary(args, sts, pending_seeds(), inputs, hashes)
            write_bench(args, sts)
            print(f"[summary] {path}", flush=True)

    for s in planned:
        if s in [x["seed"] for x in sts]:
            continue
        b = os.path.join(args.out_dir, f"ifeval_1024_seed{s}_baseline.csv")
        c = os.path.join(args.out_dir, f"ifeval_1024_seed{s}_blind.csv")
        if os.path.exists(b) and os.path.exists(c):
            print(f"[reuse] seed {s} from evaluated CSVs", flush=True)
            sts.append(load_seed(s))
    sts.sort(key=lambda x: x["seed"])

    if sts and not args.no_report:
        path, lines = write_summary(args, sts, pending_seeds(), inputs, hashes)
        bench = write_bench(args, sts)
        print("\n".join(lines), flush=True)
        print(f"[summary] {path}", flush=True)
        print(f"[bench]   {bench}", flush=True)
    elif not sts:
        print("[warn] no seeds processed", flush=True)


if __name__ == "__main__":
    main()
