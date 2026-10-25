#!/usr/bin/env python3
"""Context-Aware Decoding (CAD) conflict-probe audit.

Five conflict probes (data/_cad_probes.json): a false context asserts a target
fact, the bare question elicits the model's prior. Arms:
  P  prior-only (query stream only)
  A  RAG baseline (context+query stream only)
  B  CAD alpha=0.5, beta=0   (no APC)
  C  CAD alpha=0.5, beta=0.1
  D  CAD alpha=1.0, beta=0.1

Two CLI invocations total (one per temperature), each --arm all:
  greedy    temp 0.0,  seed 1234
  sampled   temp 0.7,  seeds 1234..1238
so each arm/probe gets 1 greedy + 5 sampled runs.

Scoring per run: target alias -> context_ok, prior alias -> prior_ok, and
coherent = non-empty + finish==eos + <=40 tokens + no word 4-gram repeated
more than 3 times. A probe whose prior is not reproduced by arm P
(prior_ok in none of its P runs) is flagged invalid and excluded from gates.

Usage:  python harness/run_cad_probes.py [--smoke]
Writes: results/cad_probe.json (full audit only)
"""
import argparse
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLI = os.path.join(ROOT, "app", "cad_cli.exe")
BIN = os.path.join(ROOT, "build", "cuda-graphs", "bin")
MODEL = os.path.join(ROOT, "models", "Qwen2.5-3B-Instruct-Q4_K_M.gguf")
PROBES = os.path.join(ROOT, "data", "_cad_probes.json")
GREEDY_OUT = os.path.join(ROOT, "data", "_cad_greedy.json")
SAMPLED_OUT = os.path.join(ROOT, "data", "_cad_sampled.json")
SMOKE_OUT = os.path.join(ROOT, "data", "_cad_smoke.json")
RESULTS = os.path.join(ROOT, "results", "cad_probe.json")

ARMS = ["P", "A", "B", "C", "D"]
MAX_TOKENS = 40
SAMPLE_SEEDS = [1234, 1235, 1236, 1237, 1238]
SAMPLE_TEMP = 0.7


def norm_words(text):
    return " " + " ".join(re.findall(r"[a-z0-9]+", text.lower())) + " "


def has_alias(text, aliases):
    hay = norm_words(text)
    for alias in aliases:
        needle = norm_words(alias).strip()
        if needle and f" {needle} " in hay:
            return True
    return False


def max_4gram(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    counts = {}
    for i in range(len(words) - 3):
        gram = tuple(words[i:i + 4])
        counts[gram] = counts.get(gram, 0) + 1
    return max(counts.values(), default=0)


def coherent(run):
    text = run["text"]
    return (bool(text.strip()) and run["finish"] == "eos"
            and run["n_tokens"] <= MAX_TOKENS and max_4gram(text) <= 3)


def invoke(temp, seeds, out_path, probe=None):
    cmd = [CLI, "--model", MODEL, "--probes", PROBES, "--arm", "all",
           "--max-tokens", str(MAX_TOKENS), "--temp", str(temp), "--out", out_path]
    if probe:
        cmd += ["--probe", probe]
    if len(seeds) == 1:
        cmd += ["--seed", str(seeds[0])]
    else:
        cmd += ["--seeds", ",".join(str(s) for s in seeds)]
    env = dict(os.environ)
    env["PATH"] = BIN + os.pathsep + env.get("PATH", "")
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True,
                       timeout=1800)
    if r.returncode != 0:
        raise SystemExit(f"cad_cli failed ({r.returncode}):\n{r.stderr[-1500:]}")
    with open(out_path, encoding="utf-8") as fh:
        return json.load(fh)["runs"]


def load_probes():
    with open(PROBES, encoding="utf-8") as fh:
        probes = json.load(fh)
    if isinstance(probes, dict):
        probes = probes["probes"]
    return {p["id"]: p for p in probes}


def score(runs, probes):
    scored = []
    for r in runs:
        p = probes[r["probe"]]
        text = r["text"]
        scored.append({
            "arm": r["arm"],
            "probe": r["probe"],
            "seed": r["seed"],
            "temp": r["temp"],
            "alpha": r["alpha"],
            "beta": r["beta"],
            "n_tokens": r["n_tokens"],
            "finish": r["finish"],
            "prefill_ms": r["prefill_ms"],
            "decode_ms": r["decode_ms"],
            "wall_ms": r["prefill_ms"] + r["decode_ms"],
            "context_ok": has_alias(text, p["target_aliases"]),
            "prior_ok": has_alias(text, p["prior_aliases"]),
            "coherent": coherent(r),
            "max_4gram": max_4gram(text),
            "text": text,
        })
    return scored


def rate(rows, key):
    return (sum(1 for r in rows if r[key]) / len(rows)) if rows else 0.0


def mean(rows, key):
    return (sum(r[key] for r in rows) / len(rows)) if rows else 0.0


def aggregate(scored, arms, probe_ids=None):
    rows = [r for r in scored if probe_ids is None or r["probe"] in probe_ids]
    out = {}
    for arm in arms:
        a = [r for r in rows if r["arm"] == arm]
        out[arm] = {
            "runs": len(a),
            "context_rate": rate(a, "context_ok"),
            "prior_rate": rate(a, "prior_ok"),
            "coherence_rate": rate(a, "coherent"),
            "mean_prefill_ms": mean(a, "prefill_ms"),
            "mean_decode_ms": mean(a, "decode_ms"),
            "mean_wall_ms": mean(a, "wall_ms"),
        }
    return out


def per_probe(scored, probes):
    notes = []
    for pid in probes:
        rec = {"probe": pid, "prior": probes[pid]["prior"],
               "target": probes[pid]["target"], "arms": {}}
        for arm in ARMS:
            a = [r for r in scored if r["probe"] == pid and r["arm"] == arm]
            rec["arms"][arm] = {
                "runs": len(a),
                "context": sum(1 for r in a if r["context_ok"]),
                "prior": sum(1 for r in a if r["prior_ok"]),
                "coherent": sum(1 for r in a if r["coherent"]),
            }
        notes.append(rec)
    return notes


def print_table(agg, label):
    print(f"\n=== {label} ===")
    print(f"{'arm':<4}{'runs':>5}{'ctx%':>7}{'prior%':>8}{'coh%':>7}"
          f"{'prefill':>10}{'decode':>10}{'wall':>10}")
    for arm in ARMS:
        a = agg[arm]
        print(f"{arm:<4}{a['runs']:>5}{100 * a['context_rate']:>7.1f}"
              f"{100 * a['prior_rate']:>8.1f}{100 * a['coherence_rate']:>7.1f}"
              f"{a['mean_prefill_ms']:>10.1f}{a['mean_decode_ms']:>10.1f}"
              f"{a['mean_wall_ms']:>10.1f}")


def print_per_probe(notes, invalid):
    print("\n=== per probe (hits / runs) ===")
    print(f"{'probe':<6}{'P prior':>9}{'A ctx':>8}{'A coh':>7}{'B ctx':>8}"
          f"{'B coh':>7}{'C ctx':>8}{'C coh':>7}{'D ctx':>8}{'D coh':>7}  flag")
    for n in notes:
        a = n["arms"]
        flag = "INVALID-PRIOR" if n["probe"] in invalid else ""
        print(f"{n['probe']:<6}{a['P']['prior']:>4}/{a['P']['runs']:<4}"
              f"{a['A']['context']:>4}/{a['A']['runs']:<3}"
              f"{a['A']['coherent']:>4}/{a['A']['runs']:<2}"
              f"{a['B']['context']:>4}/{a['B']['runs']:<3}"
              f"{a['B']['coherent']:>4}/{a['B']['runs']:<2}"
              f"{a['C']['context']:>4}/{a['C']['runs']:<3}"
              f"{a['C']['coherent']:>4}/{a['C']['runs']:<2}"
              f"{a['D']['context']:>4}/{a['D']['runs']:<3}"
              f"{a['D']['coherent']:>4}/{a['D']['runs']:<2}  {flag}")


def gates(scored, probes, agg_all):
    invalid = sorted({pid for pid in probes
                      if not any(r["probe"] == pid and r["arm"] == "P" and r["prior_ok"]
                                 for r in scored)})
    valid = [pid for pid in probes if pid not in invalid]
    agg_valid = aggregate(scored, ARMS, set(valid)) if valid else agg_all

    by_arm = {arm: [r for r in scored if r["arm"] == arm and r["probe"] in valid]
              for arm in ARMS}
    c_all_coh = all(r["coherent"] for r in by_arm["C"]) and bool(by_arm["C"])
    b_bad = sorted({r["probe"] for r in by_arm["B"] if not r["coherent"]})

    g1 = agg_valid["C"]["context_rate"] > agg_valid["A"]["context_rate"]
    g2 = c_all_coh
    g3 = (agg_valid["C"]["context_rate"] >= agg_valid["A"]["context_rate"]
          and len(b_bad) >= 1)
    ratio = (agg_valid["C"]["mean_wall_ms"] / agg_valid["A"]["mean_wall_ms"]
             if agg_valid["A"]["mean_wall_ms"] > 0 else float("inf"))
    g4 = ratio <= 1.10

    return {
        "valid_probes": valid,
        "invalid_probes": invalid,
        "g1": {"pass": g1, "c_context": agg_valid["C"]["context_rate"],
               "a_context": agg_valid["A"]["context_rate"]},
        "g2": {"pass": g2, "c_incoherent": sum(1 for r in by_arm["C"] if not r["coherent"]),
               "c_runs": len(by_arm["C"])},
        "g3": {"pass": g3, "c_context": agg_valid["C"]["context_rate"],
               "a_context": agg_valid["A"]["context_rate"],
               "b_collapse_probes": b_bad},
        "g4": {"pass": g4, "c_wall_ms": agg_valid["C"]["mean_wall_ms"],
               "a_wall_ms": agg_valid["A"]["mean_wall_ms"], "ratio": ratio},
    }, agg_valid


def smoke():
    runs = invoke(0.0, [1234], SMOKE_OUT, probe="P1")
    by_arm = {r["arm"]: r for r in runs if r["probe"] == "P1"}
    print("=== smoke: P1, greedy (temp 0), arms A / P / C ===")
    for arm in ("A", "P", "C"):
        r = by_arm[arm]
        print(f"\n--- arm {arm} (n={r['n_tokens']}, finish={r['finish']}, "
              f"prefill={r['prefill_ms']:.1f}ms, decode={r['decode_ms']:.1f}ms) ---")
        print(r["text"])
    print(f"\nwrote {SMOKE_OUT}")


def full():
    probes = load_probes()
    greedy = invoke(0.0, [1234], GREEDY_OUT)
    sampled = invoke(SAMPLE_TEMP, SAMPLE_SEEDS, SAMPLED_OUT)
    scored = score(greedy + sampled, probes)

    agg_all = aggregate(scored, ARMS)
    notes = per_probe(scored, probes)
    g, agg_valid = gates(scored, probes, agg_all)

    print_table(agg_all, "all probes")
    if g["invalid_probes"]:
        print_table(agg_valid, "valid probes only (gates)")
    print_per_probe(notes, set(g["invalid_probes"]))

    print("\n=== prior check (arm P) ===")
    for n in notes:
        a = n["arms"]["P"]
        print(f"  {n['probe']}: prior={n['prior']!r} reproduced "
              f"{a['prior']}/{a['runs']}"
              + ("  -> INVALID for this model" if n["probe"] in g["invalid_probes"] else ""))

    print("\n=== gates ===")
    g1, g2, g3, g4 = g["g1"], g["g2"], g["g3"], g["g4"]
    print(f"  G1 C context > A context: "
          f"{100 * g1['c_context']:.1f}% vs {100 * g1['a_context']:.1f}%  "
          f"-> {'PASS' if g1['pass'] else 'FAIL'}")
    print(f"  G2 C coherent on all runs: "
          f"{g2['c_runs'] - g2['c_incoherent']}/{g2['c_runs']}  "
          f"-> {'PASS' if g2['pass'] else 'FAIL'}")
    print(f"  G3 C context >= A AND B collapses on >=1 probe: "
          f"{100 * g3['c_context']:.1f}% vs {100 * g3['a_context']:.1f}%, "
          f"B incoherent probes {g3['b_collapse_probes']}  "
          f"-> {'PASS' if g3['pass'] else 'FAIL'}")
    print(f"  G4 wall(C)/wall(A) = {g4['ratio']:.3f} (<= 1.10)  "
          f"-> {'PASS' if g4['pass'] else 'FAIL'}")

    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "w", encoding="utf-8") as fh:
        json.dump({
            "ok": True,
            "model": os.path.relpath(MODEL, ROOT),
            "probes_file": os.path.relpath(PROBES, ROOT),
            "max_tokens": MAX_TOKENS,
            "greedy_temp": 0.0,
            "greedy_seed": 1234,
            "sample_temp": SAMPLE_TEMP,
            "sample_seeds": SAMPLE_SEEDS,
            "invocations": [GREEDY_OUT, SAMPLED_OUT],
            "probes": list(probes),
            "runs": scored,
            "aggregate_all": agg_all,
            "aggregate_valid": agg_valid,
            "per_probe": notes,
            "gates": g,
        }, fh, indent=1)
    print(f"\nwrote {RESULTS}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="P1 greedy, all arms; print A/P/C texts and exit")
    args = ap.parse_args()
    if not os.path.exists(CLI):
        raise SystemExit(f"missing {CLI}; build it with scripts\\build_tool.bat")
    if args.smoke:
        smoke()
    else:
        full()


if __name__ == "__main__":
    main()
