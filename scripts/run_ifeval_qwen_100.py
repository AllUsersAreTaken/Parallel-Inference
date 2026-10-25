#!/usr/bin/env python3
"""Qwen2.5-3B IFEval transfer check: 100 stratified prompts, baseline + blind retry.

Protocol
--------
Data: data/ifeval_qwen_100.tsv (4 prompts per primary instruction id, built by
scripts/prep_ifeval_100.py). Prompts/instruction ids/kwargs and the strict
checker are the official ones in vendor/ifeval.
Baseline: --max-tokens 512, seed 1234 + official prompt index.
Retry trigger: baseline prompt strict `follow_all` is false (same scoring as
the final report). Blind retry: original prompt + failed instruction
descriptions only, no satisfied list, seed 1234 + idx + 10000.
Final: retry response if produced, else baseline.

Response text is normalised CRLF/CR -> LF before the official checker runs.
The harness writes its CSV in Windows text mode, so every '\\n' in a response is
stored as CRLF; the official nth_paragraph_first_word check splits on '\\n\\n'
and would otherwise see one paragraph per response. The same normalisation is
applied when re-scoring the existing Llama-3.2-3B 512-token CSVs, so the two
models are scored on identical text. The un-normalised pass flag is also
written to the bench CSV for transparency.

Llama reference: results/ifeval_512_baseline_raw.csv and
results/ifeval_512_blind_raw.csv, restricted to the same 100 keys. No new Llama
generation.

Outputs
-------
    results/ifeval_qwen_100_baseline_raw.csv
    results/ifeval_qwen_100_blind_input.tsv
    results/ifeval_qwen_100_blind_raw.csv
    results/ifeval_qwen_100_summary.txt
    results/ifeval_qwen_100_bench.csv

Usage
-----
    python scripts/run_ifeval_qwen_100.py
    python scripts/run_ifeval_qwen_100.py --reuse-baseline --reuse-blind
"""

import argparse
import collections
import csv
import hashlib
import json
import math
import os
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


def norm(text):
    return (text or "").replace("\r\n", "\n").replace("\r", "\n")


def check(inp, response, normalize):
    text = norm(response) if normalize else (response or "")
    return ri.check_strict(inp, text)


def read_raw(path):
    rows = {}
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("key") is not None:
                rows[str(r["key"])] = r
    return rows


def unescape(text):
    out = []
    i = 0
    while i < len(text):
        if text[i] == "\\" and i + 1 < len(text):
            c = text[i + 1]
            if c == "n":
                out.append("\n")
                i += 2
                continue
            if c == "t":
                out.append("\t")
                i += 2
                continue
            if c == "r":
                out.append("\r")
                i += 2
                continue
            if c == "\\":
                out.append("\\")
                i += 2
                continue
        out.append(text[i])
        i += 1
    return "".join(out)


def read_tsv(path):
    items = []
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for r in reader:
            if r.get("key") is None:
                continue
            items.append({
                "key": str(r["key"]),
                "prompt": unescape(r["prompt"]),
                "idx": int(r["idx"]),
            })
    return items


def write_tsv(path, header, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(header)
        for k, prompt, idx in rows:
            fh.write(f"{k}\t{ri.esc(prompt)}\t{idx}\n")


def run_exe(cmd, log_path, env):
    print("[run] " + " ".join(cmd), flush=True)
    t0 = time.perf_counter()
    with open(log_path, "w", encoding="utf-8", newline="\n") as log:
        p = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log,
                           stderr=subprocess.STDOUT, text=True,
                           timeout=2 * 3600)
    wall = time.perf_counter() - t0
    if p.returncode != 0:
        raise SystemExit(f"command failed ({p.returncode}); see {log_path}")
    print(f"[run] done in {wall:.1f}s -> {log_path}", flush=True)
    return wall


def eval_rows(inputs, raw_rows, normalize):
    follow, passed, failed_ids = [], [], []
    for inp in inputs:
        r = raw_rows[str(inp.key)]
        o = check(inp, r["response"], normalize)
        fl = list(o.follow_instruction_list)
        follow.append(fl)
        passed.append(1 if o.follow_all_instructions else 0)
        failed_ids.append([iid for iid, ok in zip(inp.instruction_id_list, fl)
                           if not ok])
    return passed, follow, failed_ids


def instr_counts(inputs, follow):
    tot = cor = 0
    by_id = collections.Counter()
    by_id_cor = collections.Counter()
    for inp, fl in zip(inputs, follow):
        for iid, ok in zip(inp.instruction_id_list, fl):
            tot += 1
            cor += 1 if ok else 0
            by_id[iid] += 1
            by_id_cor[iid] += 1 if ok else 0
    return tot, cor, by_id, by_id_cor


def summarize(inputs, base_pass, base_follow, final_pass, final_follow,
              triggers, retry_by_key):
    n = len(inputs)
    tot, cor, _, _ = instr_counts(inputs, base_follow)
    _, fcor, _, _ = instr_counts(inputs, final_follow)
    n_trig = len(triggers)
    retry_ok = sum(1 for k in triggers if retry_by_key.get(k, 0))
    return {
        "n": n,
        "base_pass": sum(base_pass),
        "final_pass": sum(final_pass),
        "n_trig": n_trig,
        "n_retry": len(retry_by_key),
        "retry_ok": retry_ok,
        "instr_tot": tot,
        "instr_cor": cor,
        "instr_final_cor": fcor,
    }


def type_table(inputs, types, base_pass, base_follow, final_pass,
               final_follow, triggers):
    groups = collections.defaultdict(list)
    for i, inp in enumerate(inputs):
        groups[types[i]].append(i)
    out = []
    for t in sorted(groups):
        idxs = groups[t]
        ti = [inputs[i] for i in idxs]
        bf = [base_follow[i] for i in idxs]
        ff = [final_follow[i] for i in idxs]
        tot, cor, _, _ = instr_counts(ti, bf)
        _, fcor, _, _ = instr_counts(ti, ff)
        ks = {str(inputs[i].key) for i in idxs}
        trig = [k for k in triggers if k in ks]
        retry_ok = sum(1 for i in idxs if int(base_pass[i]) == 0
                       and int(final_pass[i]) == 1)
        prim_tot = len(idxs)
        prim_cor = sum(1 for i in idxs
                       if base_follow[i][0])
        prim_final = sum(1 for i in idxs
                         if final_follow[i][0])
        out.append({
            "type": t,
            "n": len(idxs),
            "base_pass": sum(int(base_pass[i]) for i in idxs),
            "final_pass": sum(int(final_pass[i]) for i in idxs),
            "n_trig": len(trig),
            "retry_ok": retry_ok,
            "instr_tot": tot,
            "instr_cor": cor,
            "instr_final_cor": fcor,
            "prim_tot": prim_tot,
            "prim_cor": prim_cor,
            "prim_final": prim_final,
        })
    return out


def parse_log_wall(path):
    return ri.log_wall_seconds(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default=os.path.join(
        ROOT, "harness", "engine", "ifeval.exe"))
    ap.add_argument("--model", default=os.path.join(
        ROOT, "models", "Qwen2.5-3B-Instruct-Q4_K_M.gguf"))
    ap.add_argument("--data", default=os.path.join(
        ROOT, "data", "ifeval_qwen_100.tsv"))
    ap.add_argument("--official-data", default=os.path.join(
        ROOT, "vendor", "ifeval", "instruction_following_eval", "data",
        "input_data.jsonl"))
    ap.add_argument("--build-dir", default=os.path.join(
        ROOT, "build", "cuda-graphs"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "results"))
    ap.add_argument("--tag", default="ifeval_qwen_100")
    ap.add_argument("--llama-baseline", default=os.path.join(
        ROOT, "results", "ifeval_512_baseline_raw.csv"))
    ap.add_argument("--llama-blind", default=os.path.join(
        ROOT, "results", "ifeval_512_blind_raw.csv"))
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--retry-seed-add", type=int, default=10000)
    ap.add_argument("--reuse-baseline", action="store_true")
    ap.add_argument("--reuse-blind", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    base_raw = os.path.join(args.out_dir, f"{args.tag}_baseline_raw.csv")
    blind_input = os.path.join(args.out_dir, f"{args.tag}_blind_input.tsv")
    blind_raw = os.path.join(args.out_dir, f"{args.tag}_blind_raw.csv")
    base_log = os.path.join(args.out_dir, f"{args.tag}_baseline.log")
    blind_log = os.path.join(args.out_dir, f"{args.tag}_blind.log")
    summary_txt = os.path.join(args.out_dir, f"{args.tag}_summary.txt")
    bench_csv = os.path.join(args.out_dir, f"{args.tag}_bench.csv")

    inputs_all = evaluation_lib.read_prompt_list(args.official_data)
    if len(inputs_all) != 541:
        raise SystemExit(f"official data has {len(inputs_all)} prompts, "
                         f"expected 541")
    official = {str(inp.key): inp for inp in inputs_all}

    subset = read_tsv(args.data)
    if len(subset) != 100:
        raise SystemExit(f"{args.data}: {len(subset)} rows, expected 100")
    inputs = []
    for item in subset:
        inp = official.get(item["key"])
        if inp is None:
            raise SystemExit(f"key {item['key']} not in official data")
        if inp.prompt != item["prompt"]:
            raise SystemExit(f"key {item['key']}: prompt differs from official")
        inputs.append(inp)
    keys = [str(inp.key) for inp in inputs]
    if len(set(keys)) != len(keys):
        raise SystemExit("duplicate keys in subset")
    pos = {str(inp.key): i for i, inp in enumerate(inputs_all)}
    official_idx = {item["key"]: item["idx"] for item in subset}
    for item in subset:
        if item["idx"] != pos[item["key"]]:
            raise SystemExit(f"key {item['key']}: idx does not match official order")
    types = [inp.instruction_id_list[0] for inp in inputs]
    n = len(inputs)

    env = dict(os.environ)
    env["PATH"] = os.path.join(args.build_dir, "bin") + os.pathsep + \
        env.get("PATH", "")

    # ---------------- Qwen: baseline ----------------
    if args.reuse_baseline and os.path.exists(base_raw):
        print(f"[reuse] {base_raw}", flush=True)
        base_wall = None
    else:
        cmd = [args.exe, "--model", args.model, "--data", args.data,
               "--out", base_raw, "--limit", str(n),
               "--max-tokens", str(args.max_tokens), "--seed", str(args.seed)]
        base_wall = run_exe(cmd, base_log, env)
    base_raw_rows = read_raw(base_raw)
    missing = [k for k in keys if k not in base_raw_rows]
    if missing:
        raise SystemExit(f"{base_raw}: missing keys {missing[:3]}")

    base_pass, base_follow, base_failed = eval_rows(inputs, base_raw_rows, True)
    base_pass_raw, base_follow_raw, _ = eval_rows(inputs, base_raw_rows, False)
    triggers = [keys[i] for i in range(n) if not base_pass[i]]
    print(f"[eval] Qwen baseline strict {sum(base_pass)}/{n}; "
          f"triggered {len(triggers)}", flush=True)

    # ---------------- Qwen: blind retry ----------------
    retry_prompts = {}
    for i in range(n):
        if base_pass[i]:
            continue
        inp = inputs[i]
        descs = describe_all(inp)
        prompt, _sat, _fail = make_retry_prompt(
            inp, descs, base_follow[i], include_satisfied=False)
        retry_prompts[str(inp.key)] = prompt
    write_tsv(blind_input, "key\tretry_prompt\tidx\n",
              [(k, p, official_idx[k]) for k, p in retry_prompts.items()])
    print(f"[retry-input] {len(retry_prompts)} prompts", flush=True)

    if args.reuse_blind and os.path.exists(blind_raw):
        print(f"[reuse] {blind_raw}", flush=True)
        blind_wall = None
    else:
        cmd = [args.exe, "--model", args.model, "--retry-data", blind_input,
               "--out", blind_raw, "--limit", str(len(retry_prompts)),
               "--max-tokens", str(args.max_tokens), "--seed", str(args.seed),
               "--seed-add", str(args.retry_seed_add)]
        blind_wall = run_exe(cmd, blind_log, env)
    blind_raw_rows = read_raw(blind_raw)
    missing = [k for k in retry_prompts if k not in blind_raw_rows]
    if missing:
        raise SystemExit(f"{blind_raw}: missing retry keys {missing[:3]}")

    retry_pass, retry_follow = {}, {}
    for k in retry_prompts:
        i = keys.index(k)
        o = check(inputs[i], blind_raw_rows[k]["response"], True)
        retry_pass[k] = 1 if o.follow_all_instructions else 0
        retry_follow[k] = list(o.follow_instruction_list)

    final_pass, final_follow = [], []
    for i, k in enumerate(keys):
        if k in retry_pass:
            final_pass.append(retry_pass[k])
            final_follow.append(list(retry_follow[k]))
        else:
            final_pass.append(base_pass[i])
            final_follow.append(list(base_follow[i]))

    qwen = summarize(inputs, base_pass, base_follow, final_pass, final_follow,
                     triggers, retry_pass)
    qwen_types = type_table(inputs, types, base_pass, base_follow, final_pass,
                            final_follow, triggers)

    # ---------------- Llama: same keys, existing 512 raw CSVs ----------------
    if not (os.path.exists(args.llama_baseline) and os.path.exists(args.llama_blind)):
        raise SystemExit("Llama 512 raw CSVs not found; cannot build the subset")
    llama_base_rows = read_raw(args.llama_baseline)
    llama_blind_rows = read_raw(args.llama_blind)
    missing = [k for k in keys if k not in llama_base_rows]
    if missing:
        raise SystemExit(f"{args.llama_baseline}: missing keys {missing[:3]}")

    l_base_pass, l_base_follow, _ = eval_rows(inputs, llama_base_rows, True)
    l_base_pass_raw, _, _ = eval_rows(inputs, llama_base_rows, False)
    l_triggers = [keys[i] for i in range(n) if not l_base_pass[i]]
    l_retry_pass, l_retry_follow = {}, {}
    for k in l_triggers:
        if k not in llama_blind_rows:
            raise SystemExit(f"{args.llama_blind}: no retry row for key {k}")
        i = keys.index(k)
        o = check(inputs[i], llama_blind_rows[k]["response"], True)
        l_retry_pass[k] = 1 if o.follow_all_instructions else 0
        l_retry_follow[k] = list(o.follow_instruction_list)
    l_final_pass, l_final_follow = [], []
    for i, k in enumerate(keys):
        if k in l_retry_pass:
            l_final_pass.append(l_retry_pass[k])
            l_final_follow.append(list(l_retry_follow[k]))
        else:
            l_final_pass.append(l_base_pass[i])
            l_final_follow.append(list(l_base_follow[i]))

    llama = summarize(inputs, l_base_pass, l_base_follow, l_final_pass,
                      l_final_follow, l_triggers, l_retry_pass)
    llama_types = type_table(inputs, types, l_base_pass, l_base_follow,
                             l_final_pass, l_final_follow, l_triggers)

    # ---------------- bench CSV ----------------
    fields = [
        "key", "idx", "primary_type", "instruction_id_list", "prompt",
        "qwen_baseline_follow_all", "qwen_baseline_follow_raw",
        "qwen_baseline_follow_instruction_list", "qwen_baseline_failed_ids",
        "qwen_baseline_response",
        "qwen_finish", "qwen_n_tokens", "qwen_gen_ms", "qwen_total_ms",
        "qwen_triggered", "qwen_retry_prompt", "qwen_retry_response",
        "qwen_retry_follow_all", "qwen_retry_failed_ids",
        "qwen_final_follow_all", "qwen_final_source",
        "llama_baseline_follow_all", "llama_baseline_follow_raw",
        "llama_baseline_follow_instruction_list", "llama_baseline_failed_ids",
        "llama_baseline_response",
        "llama_finish", "llama_n_tokens",
        "llama_triggered", "llama_retry_response", "llama_retry_follow_all",
        "llama_retry_failed_ids",
        "llama_final_follow_all", "llama_final_source",
    ]
    with open(bench_csv, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for i, inp in enumerate(inputs):
            k = str(inp.key)
            br = base_raw_rows[k]
            lb = llama_base_rows[k]
            qr = blind_raw_rows.get(k)
            lr = llama_blind_rows.get(k) if k in l_retry_pass else None
            w.writerow({
                "key": k, "idx": official_idx[k], "primary_type": types[i],
                "instruction_id_list": json.dumps(inp.instruction_id_list),
                "prompt": inp.prompt,
                "qwen_baseline_follow_all": base_pass[i],
                "qwen_baseline_follow_raw": base_pass_raw[i],
                "qwen_baseline_follow_instruction_list": json.dumps(base_follow[i]),
                "qwen_baseline_failed_ids": json.dumps(base_failed[i]),
                "qwen_baseline_response": br["response"],
                "qwen_finish": br.get("finish", ""),
                "qwen_n_tokens": br.get("n_tokens", ""),
                "qwen_gen_ms": br.get("gen_ms", ""),
                "qwen_total_ms": br.get("total_ms", ""),
                "qwen_triggered": 1 if k in retry_pass else 0,
                "qwen_retry_prompt": retry_prompts.get(k, ""),
                "qwen_retry_response": qr["response"] if qr else "",
                "qwen_retry_follow_all": retry_pass.get(k, ""),
                "qwen_retry_failed_ids": json.dumps(
                    [iid for iid, ok in zip(inp.instruction_id_list,
                                            retry_follow.get(k, [])) if not ok])
                    if k in retry_follow else "",
                "qwen_final_follow_all": final_pass[i],
                "qwen_final_source": "retry" if k in retry_pass else "baseline",
                "llama_baseline_follow_all": l_base_pass[i],
                "llama_baseline_follow_raw": l_base_pass_raw[i],
                "llama_baseline_follow_instruction_list":
                    json.dumps(l_base_follow[i]),
                "llama_baseline_failed_ids":
                    json.dumps([iid for iid, ok in zip(
                        inp.instruction_id_list, l_base_follow[i]) if not ok]),
                "llama_baseline_response": lb["response"],
                "llama_finish": lb.get("finish", ""),
                "llama_n_tokens": lb.get("n_tokens", ""),
                "llama_triggered": 1 if k in l_retry_pass else 0,
                "llama_retry_response": lr["response"] if lr else "",
                "llama_retry_follow_all": l_retry_pass.get(k, ""),
                "llama_retry_failed_ids": json.dumps(
                    [iid for iid, ok in zip(inp.instruction_id_list,
                                            l_retry_follow.get(k, [])) if not ok])
                    if k in l_retry_follow else "",
                "llama_final_follow_all": l_final_pass[i],
                "llama_final_source": "retry" if k in l_retry_pass
                else "baseline",
            })

    # ---------------- summary ----------------
    def rate(k, d):
        return k / d if d else float("nan")

    def metrics(m):
        return (
            m["base_pass"], rate(m["base_pass"], m["n"]),
            m["final_pass"], rate(m["final_pass"], m["n"]),
            m["n_trig"], rate(m["n_trig"], m["n"]),
            m["retry_ok"], rate(m["retry_ok"], m["n_trig"]),
            m["instr_cor"], rate(m["instr_cor"], m["instr_tot"]),
            m["instr_final_cor"], rate(m["instr_final_cor"], m["instr_tot"]),
        )

    n_raw_diff_q = sum(1 for i in range(n) if base_pass[i] != base_pass_raw[i])
    n_raw_diff_l = sum(1 for i in range(n) if l_base_pass[i] != l_base_pass_raw[i])
    n_retry_trunc = sum(1 for r in blind_raw_rows.values()
                        if r.get("finish") == "max")
    n_base_trunc = sum(1 for r in base_raw_rows.values()
                       if r.get("finish") == "max")
    n_base_trunc_fail = sum(
        1 for i, k in enumerate(keys)
        if base_raw_rows[k].get("finish") == "max" and not base_pass[i])
    n_retry_trunc_fail = sum(
        1 for k in retry_prompts
        if blind_raw_rows[k].get("finish") == "max" and not retry_pass[k])
    n_llama_trunc = sum(1 for k in keys
                        if llama_base_rows[k].get("finish") == "max")
    n_llama_trunc_fail = sum(
        1 for i, k in enumerate(keys)
        if llama_base_rows[k].get("finish") == "max" and not l_base_pass[i])

    def slot_flips(bf, ff):
        rec = reg = 0
        for a, b in zip(bf, ff):
            for x, y in zip(a, b):
                if not x and y:
                    rec += 1
                elif x and not y:
                    reg += 1
        return rec, reg

    q_rec, q_reg = slot_flips(base_follow, final_follow)
    l_rec, l_reg = slot_flips(l_base_follow, l_final_follow)
    q_gen = [float(base_raw_rows[k]["gen_ms"]) for k in keys]
    q_tot = [float(base_raw_rows[k]["total_ms"]) for k in keys]
    q_retry_gen = [float(blind_raw_rows[k]["gen_ms"]) for k in retry_prompts]
    q_retry_tot = [float(blind_raw_rows[k]["total_ms"]) for k in retry_prompts]
    l_gen = [float(llama_base_rows[k]["gen_ms"]) for k in keys]
    l_tot = [float(llama_base_rows[k]["total_ms"]) for k in keys]

    L = []
    L.append("IFEval transfer check: Qwen2.5-3B-Instruct Q4_K_M vs "
             "Llama-3.2-3B-Instruct Q4_K_M")
    L.append("=" * 78)
    L.append("100 stratified prompts (4 per primary instruction id), "
             "max_tokens 512, seed 1234.")
    L.append("Verifier: official IFEval strict checker "
             "(google-research instruction_following_eval).")
    L.append("")
    L.append("Provenance")
    L.append("-" * 78)
    L.append(f"  qwen model : {os.path.relpath(args.model, ROOT)}")
    L.append(f"    sha256   : {sha256(args.model)}")
    L.append(f"  subset     : {os.path.relpath(args.data, ROOT)}")
    L.append(f"    sha256   : {sha256(args.data)}")
    L.append(f"  official   : {os.path.relpath(args.official_data, ROOT)}")
    L.append(f"    sha256   : {sha256(args.official_data)}")
    L.append(f"  harness    : harness/engine/ifeval.cpp")
    L.append(f"    sha256   : {sha256(os.path.join(ROOT, 'harness', 'engine', 'ifeval.cpp'))}")
    L.append(f"  exe        : {os.path.relpath(args.exe, ROOT)}")
    L.append(f"    sha256   : {sha256(args.exe)}")
    L.append(f"  llama ref  : {os.path.relpath(args.llama_baseline, ROOT)}")
    L.append(f"    sha256   : {sha256(args.llama_baseline)}")
    L.append(f"  llama blind: {os.path.relpath(args.llama_blind, ROOT)}")
    L.append(f"    sha256   : {sha256(args.llama_blind)}")
    L.append("")
    L.append("Protocol")
    L.append("-" * 78)
    L.append("  selection: 4 prompts per primary instruction id (25 ids), sampled")
    L.append("    with random.Random(1234) in scripts/prep_ifeval_100.py; rows in")
    L.append("    official order; prompt idx = official prompt index.")
    L.append("  qwen baseline: one response per prompt, temp 0.6, top_k 20,")
    L.append("    top_p 0.95, max_tokens 512, seed 1234 + idx.")
    L.append(f"  retry trigger: baseline prompt strict is false "
             f"({qwen['n_trig']}/{n}).")
    L.append("  blind retry: original prompt + failed instruction descriptions only")
    L.append("    (no satisfied list), seed 1234 + idx + 10000.")
    L.append("  final = retry response if produced, else baseline.")
    L.append("  llama: existing 512-token raw responses for the same 100 keys,")
    L.append("    re-scored here; no new llama generation. Llama retry rows are")
    L.append("    used only for keys that fail under the scoring below.")
    L.append("  descriptive only: no bootstrap CIs, no McNemar.")
    L.append("  the 100-prompt set is stratified (4 per type), not a random sample;")
    L.append("    absolute levels are not directly the full-541 levels.")
    L.append("  newline handling: responses are normalised CRLF/CR -> LF before")
    L.append("    the checker (see Evaluator notes); both models treated identically.")
    L.append("")
    L.append("Overall (100 prompts)")
    L.append("-" * 78)
    L.append(f"  {'model':<12} {'base prompt':>12} {'final prompt':>13} "
             f"{'retry rate':>11} {'retry succ':>13} "
             f"{'instr base':>14} {'instr final':>14}")
    for name, m in (("Qwen2.5-3B", qwen), ("Llama-3.2-3B", llama)):
        b, br, f, fr, t, tr, ro, rs, ic, ir, ifc, ifr = metrics(m)
        L.append(f"  {name:<12} {pct(br):>7} ({b:>2}) {pct(fr):>8} ({f:>2}) "
                 f"{pct(tr):>6} ({t:>2}) {pct(rs):>7} ({ro:>2}/{t:<2}) "
                 f"{pct(ir):>7} ({ic:>3}/{m['instr_tot']:<3}) "
                 f"{pct(ifr):>7} ({ifc:>3}/{m['instr_tot']:<3})")
    L.append("")
    L.append("Per primary instruction type (Qwen2.5-3B)")
    L.append("-" * 78)
    L.append(f"  {'primary instruction':<48} {'n':>2} {'base':>7} {'final':>7} "
             f"{'retry':>6} {'succ':>6} {'instr b':>8} {'instr f':>8}")
    for r in qwen_types:
        L.append(f"  {r['type']:<48} {r['n']:>2} "
                 f"{pct(rate(r['base_pass'], r['n'])):>7} "
                 f"{pct(rate(r['final_pass'], r['n'])):>7} "
                 f"{pct(rate(r['n_trig'], r['n'])):>6} "
                 f"{pct(rate(r['retry_ok'], r['n_trig'])):>6} "
                 f"{pct(rate(r['instr_cor'], r['instr_tot'])):>8} "
                 f"{pct(rate(r['instr_final_cor'], r['instr_tot'])):>8}")
    L.append("")
    L.append("Per primary instruction type (Llama-3.2-3B, same 100 prompts)")
    L.append("-" * 78)
    L.append(f"  {'primary instruction':<48} {'n':>2} {'base':>7} {'final':>7} "
             f"{'retry':>6} {'succ':>6} {'instr b':>8} {'instr f':>8}")
    for r in llama_types:
        L.append(f"  {r['type']:<48} {r['n']:>2} "
                 f"{pct(rate(r['base_pass'], r['n'])):>7} "
                 f"{pct(rate(r['final_pass'], r['n'])):>7} "
                 f"{pct(rate(r['n_trig'], r['n'])):>6} "
                 f"{pct(rate(r['retry_ok'], r['n_trig'])):>6} "
                 f"{pct(rate(r['instr_cor'], r['instr_tot'])):>8} "
                 f"{pct(rate(r['instr_final_cor'], r['instr_tot'])):>8}")
    L.append("")
    L.append("Side by side (Qwen vs Llama, same 100 keys)")
    L.append("-" * 78)
    L.append(f"  {'primary instruction':<44} {'n':>2} "
             f"{'base Q':>7} {'base L':>7} {'final Q':>8} {'final L':>8} "
             f"{'succ Q':>7} {'succ L':>7} {'instr Q':>8} {'instr L':>8}")
    l_by_type = {r["type"]: r for r in llama_types}
    for r in qwen_types:
        s = l_by_type[r["type"]]
        L.append(f"  {r['type']:<44} {r['n']:>2} "
                 f"{pct(rate(r['base_pass'], r['n'])):>7} "
                 f"{pct(rate(s['base_pass'], s['n'])):>7} "
                 f"{pct(rate(r['final_pass'], r['n'])):>8} "
                 f"{pct(rate(s['final_pass'], s['n'])):>8} "
                 f"{pct(rate(r['retry_ok'], r['n_trig'])):>7} "
                 f"{pct(rate(s['retry_ok'], s['n_trig'])):>7} "
                 f"{pct(rate(r['instr_cor'], r['instr_tot'])):>8} "
                 f"{pct(rate(s['instr_cor'], s['instr_tot'])):>8}")
    L.append("")
    L.append("Primary-instruction-only strict (the instruction that defines the type)")
    L.append("-" * 78)
    L.append(f"  {'primary instruction':<44} {'n':>2} {'base Q':>7} {'base L':>7} "
             f"{'final Q':>8} {'final L':>8}")
    for r in qwen_types:
        s = l_by_type[r["type"]]
        L.append(f"  {r['type']:<44} {r['n']:>2} "
                 f"{pct(rate(r['prim_cor'], r['prim_tot'])):>7} "
                 f"{pct(rate(s['prim_cor'], s['prim_tot'])):>7} "
                 f"{pct(rate(r['prim_final'], r['prim_tot'])):>8} "
                 f"{pct(rate(s['prim_final'], s['prim_tot'])):>8}")
    L.append("")
    L.append("Retry accounting (prompt level)")
    L.append("-" * 78)
    for name, m, rec in (("Qwen2.5-3B", qwen, q_rec),
                         ("Llama-3.2-3B", llama, l_rec)):
        L.append(f"  {name:<12} baseline failures {m['n_trig']}/{m['n']} "
                 f"({pct(rate(m['n_trig'], m['n']))}); retry rows "
                 f"{m['n_retry']}/{m['n_trig']}; retry successes "
                 f"{m['retry_ok']}/{m['n_trig']} "
                 f"({pct(rate(m['retry_ok'], m['n_trig']))})")
    L.append(f"  qwen instruction slots: fail->pass={q_rec}, pass->fail={q_reg}")
    L.append(f"  llama instruction slots: fail->pass={l_rec}, pass->fail={l_reg}")
    L.append("  A retry can break a slot the baseline already satisfied; final uses")
    L.append("  the retry response for every triggered prompt, as in the 512 arm.")
    L.append("")
    L.append("Truncation and latency")
    L.append("-" * 78)
    L.append(f"  qwen baseline at max_tokens: {n_base_trunc}/{n} "
             f"({pct(rate(n_base_trunc, n))}); of those "
             f"{n_base_trunc_fail} failed the checker")
    L.append(f"  qwen blind retry at max_tokens: {n_retry_trunc}/{len(blind_raw_rows)} "
             f"({pct(rate(n_retry_trunc, len(blind_raw_rows)))}); of those "
             f"{n_retry_trunc_fail} failed the checker")
    L.append(f"  llama baseline at max_tokens: {n_llama_trunc}/{n} "
             f"({pct(rate(n_llama_trunc, n))}); of those "
             f"{n_llama_trunc_fail} failed the checker (existing 512 run)")
    L.append(f"  qwen baseline gen  : mean {sum(q_gen)/len(q_gen):.1f} ms, "
             f"total {sum(q_tot)/len(q_tot):.1f} ms")
    L.append(f"  qwen retry gen     : mean {sum(q_retry_gen)/len(q_retry_gen):.1f} ms, "
             f"total {sum(q_retry_tot)/len(q_retry_tot):.1f} ms")
    L.append(f"  qwen retry extra over {n} prompts: "
             f"{sum(q_retry_tot)/1000.0:.1f}s "
             f"({sum(q_retry_tot)/n:.1f} ms/prompt amortized)")
    L.append(f"  qwen wall          : baseline "
             f"{(base_wall if base_wall is not None else parse_log_wall(base_log)) or float('nan'):.1f}s, "
             f"blind "
             f"{(blind_wall if blind_wall is not None else parse_log_wall(blind_log)) or float('nan'):.1f}s")
    L.append(f"  llama baseline gen : mean {sum(l_gen)/len(l_gen):.1f} ms, "
             f"total {sum(l_tot)/len(l_tot):.1f} ms (existing 512 run)")
    L.append("")
    L.append("Evaluator notes")
    L.append("-" * 78)
    L.append("  - Chat template: the harness applies the GGUF chat template with a")
    L.append("    system turn ('You are a helpful assistant. Follow all instructions")
    L.append("    exactly.') and one user turn, add_assistant=true. A 3-prompt smoke")
    L.append("    (keys 1005, 1019, 1040) produced plain answer text with no role")
    smoke_raw = os.path.join(args.out_dir, f"{args.tag}_smoke_raw.csv")
    if os.path.exists(smoke_raw):
        srows = read_raw(smoke_raw)
        detail = ", ".join(
            f"{k}: {srows[k].get('n_tokens')} tok/{srows[k].get('finish')}"
            for k in ("1005", "1019", "1040") if k in srows)
        L.append(f"    markers and no template leakage ({detail}).")
    else:
        L.append("    markers and no template leakage.")
    L.append("  - Newline artifact: harness/engine/ifeval.cpp writes the raw CSV with")
    L.append("    std::fopen(..., \"w\"), so every '\\n' is stored as CRLF on Windows.")
    L.append("    The official nth_paragraph_first_word checker splits on '\\n\\n' and")
    L.append("    would see one paragraph in every response. This run normalises")
    L.append("    CRLF/CR -> LF for both models before checking.")
    L.append(f"    Baseline prompts whose pass flag changes under normalisation: "
             f"qwen {n_raw_diff_q}/{n}, llama {n_raw_diff_l}/{n}.")
    L.append(f"    Baseline prompt strict as stored vs normalised: qwen "
             f"{sum(base_pass_raw)}/{n} vs {sum(base_pass)}/{n}; llama "
             f"{sum(l_base_pass_raw)}/{n} vs {sum(l_base_pass)}/{n}.")
    L.append("    The bench CSV keeps both the normalised and as-stored flags.")
    L.append("  - Llama baseline failures were triggered from the original un-normalised")
    L.append("    full-set run; here retry responses are used only for keys that still")
    L.append("    fail after normalisation, so the Llama final score uses the same")
    L.append("    trigger rule as Qwen.")
    L.append("  - Retries were generated only for baseline failures, so prompt-level")
    L.append("    pass->fail against baseline is impossible by construction.")
    L.append("")
    L.append("Commands")
    L.append("-" * 78)
    L.append("  prep   : python scripts\\prep_ifeval_100.py")
    L.append("  smoke  : harness\\engine\\ifeval.exe --model "
             "models\\Qwen2.5-3B-Instruct-Q4_K_M.gguf --data data\\ifeval_qwen_100.tsv "
             "--out results\\ifeval_qwen_100_smoke_raw.csv --limit 3 --max-tokens 512 --seed 1234")
    L.append("  build  : scripts\\build_tool.bat build\\cuda-graphs "
             "harness\\engine\\ifeval.cpp harness\\engine\\ifeval.exe")
    L.append(f"  baseline: {args.exe} --model {args.model} --data {args.data} "
             f"--out {base_raw} --limit {n} --max-tokens {args.max_tokens} "
             f"--seed {args.seed}")
    L.append(f"  blind   : {args.exe} --model {args.model} --retry-data {blind_input} "
             f"--out {blind_raw} --limit {len(retry_prompts)} "
             f"--max-tokens {args.max_tokens} --seed {args.seed} "
             f"--seed-add {args.retry_seed_add}")
    L.append("  report  : python scripts\\run_ifeval_qwen_100.py "
             "--reuse-baseline --reuse-blind")
    L.append("")
    L.append("Artifacts")
    L.append("-" * 78)
    for label, path in [
        ("baseline_raw", base_raw),
        ("blind_input", blind_input),
        ("blind_raw", blind_raw),
        ("smoke_raw", os.path.join(args.out_dir, f"{args.tag}_smoke_raw.csv")),
        ("smoke_log", os.path.join(args.out_dir, f"{args.tag}_smoke.log")),
        ("bench_csv", bench_csv),
    ]:
        L.append(f"  {label:<14} {sha256(path)}  {os.path.basename(path)}")
    L.append("")

    with open(summary_txt, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L), flush=True)
    print(f"[summary] {summary_txt}", flush=True)
    print(f"[bench]   {bench_csv}", flush=True)


if __name__ == "__main__":
    main()
