"""In-flight memory extraction test (RecycleLM).

k=2 on parallel::Runtime: Slot 0 = conversational reply, Slot 1 = Mem0-style JSON
fact extraction, both decoding concurrently in the same batch.

- 5 solo reps (serial mode, Slot 0 prompt only) vs 5 batch k=2 reps, seeds 1234-1238.
- Measures: Slot 1 JSON parse rate + field accuracy (TP/FP vs hand label),
  Slot 0 latency ratio (batch/solo) for ttft and completion.

Writes results/ingame_memory_test.json and results/ingame_memory_test.txt.
Raw per-rep CLI JSON kept in results/ingame_memory_raw/.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "app" / "parallel_cli.exe"
DLL_DIR = ROOT / "build" / "cuda-graphs" / "bin"
MODEL = "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
SOLO_TSV = "data/ingame_mem_solo.tsv"
BATCH_TSV = "data/ingame_mem_batch.tsv"
RAW_DIR = ROOT / "results" / "ingame_memory_raw"
OUT_JSON = ROOT / "results" / "ingame_memory_test.json"
OUT_TXT = ROOT / "results" / "ingame_memory_test.txt"

SEEDS = [1234, 1235, 1236, 1237, 1238]
MAX_TOKENS = 160
TEMP = 0.2
TOP_P = 0.9
SYSTEM = "You are a helpful assistant."

# Hand label. Each concept: list of token groups; every group must match
# (str = substring, tuple = any of the substrings), case-insensitive.
FACTS_LABEL = {
    "buster_knee_issue": ["buster", "knee"],
    "stick_to_flat_river_trail": ["flat", ("trail", "river")],
    "user_bought_tesla": ["tesla"],
    "needs_charging_stations": ["charg", ("station", "commute")],
}
INVALIDATED_LABEL = {
    "user_drives_honda": ["honda"],
    "needs_gas_stations": ["gas", ("station", "pump", "fuel")],
}
ALL_LABEL = {**FACTS_LABEL, **INVALIDATED_LABEL}


def match_concepts(text):
    """Return ALL label names this string matches, case-insensitive."""
    t = text.lower()
    hits = []
    for name, groups in ALL_LABEL.items():
        ok = True
        for g in groups:
            opts = (g,) if isinstance(g, str) else tuple(g)
            if not any(o in t for o in opts):
                ok = False
                break
        if ok:
            hits.append(name)
    return hits


def run_cli(mode, tsv, seed, out_path):
    env = dict(os.environ)
    env["PATH"] = str(DLL_DIR) + os.pathsep + env.get("PATH", "")
    cmd = [
        str(CLI), "--model", MODEL, "--data", tsv, "--out", str(out_path),
        "--mode", mode, "--max-tokens", str(MAX_TOKENS),
        "--temperature", str(TEMP), "--top-p", str(TOP_P),
        "--seed", str(seed), "--system", SYSTEM,
    ]
    r = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True,
                       text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"CLI failed ({mode} seed={seed}): {r.stderr.strip()}")
    return json.loads(r.stdout)


def parse_slot1_json(text):
    """Return (obj_or_None, method). Strips markdown fences; brace-scan fallback."""
    t = text.strip()
    method = "direct"
    if t.startswith("```"):
        lines = t.splitlines()
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        t = "\n".join(lines).strip()
        method = "fence_stripped"
    try:
        return json.loads(t), method
    except ValueError:
        pass
    i, depth = t.find("{"), 0
    if i >= 0:
        for j in range(i, len(t)):
            if t[j] == "{":
                depth += 1
            elif t[j] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(t[i:j + 1]), method + "+brace_scan"
                    except ValueError:
                        break
    return None, method + "+failed"


def score_slot1(obj):
    """TP/FP for facts and invalidated lists vs the hand label.

    A concept counts as TP if any string in the concept's expected list covers
    it. A string matching no concept is an FP. A concept covered only in the
    opposite list is a placement miss (wrong_list), not an FP.
    """
    out = {"facts_tp": 0, "facts_fp": 0, "inv_tp": 0, "inv_fp": 0,
           "facts_matched": [], "inv_matched": [], "facts_unmatched": [],
           "inv_unmatched": [], "wrong_list": [], "coverage": {}}
    for key, exp, tp, fp, matched, unmatched in (
        ("facts", FACTS_LABEL, "facts_tp", "facts_fp", "facts_matched", "facts_unmatched"),
        ("invalidated", INVALIDATED_LABEL, "inv_tp", "inv_fp", "inv_matched", "inv_unmatched"),
    ):
        seen = set()
        raw = obj.get(key, []) if isinstance(obj.get(key), list) else []
        for s in raw:
            hits = match_concepts(s)
            if not hits:
                out[fp] += 1
                out[unmatched].append(s)
                continue
            out[matched].append(s)
            for h in hits:
                out["coverage"].setdefault(h, []).append(key)
                if h in exp:
                    seen.add(h)
                else:
                    out["wrong_list"].append({"concept": h, "in_list": key, "text": s})
        out[tp] = len(seen)
    return out


def get_stream(doc, marker):
    arr = doc.get("streams") if doc.get("mode") == "batch" else doc.get("serial")
    for s in arr:
        if marker in s["prompt"]:
            return s
    raise KeyError(f"stream with marker {marker!r} not found")


def summarize_branch(b):
    return {k: b[k] for k in ("text", "generated", "ttft_ms", "completion_ms",
                              "tok_s", "finish")}


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def build_rep(seed, solo, batch):
    s0_solo = get_stream(solo, "Reply to the user")
    s0_batch = get_stream(batch, "Reply to the user")
    s1_batch = get_stream(batch, "Extract memory updates")

    obj, method = parse_slot1_json(s1_batch["text"])
    sc = score_slot1(obj) if isinstance(obj, dict) else None
    return {
        "seed": seed,
        "solo_slot0": summarize_branch(s0_solo),
        "batch_slot0": summarize_branch(s0_batch),
        "batch_slot1": summarize_branch(s1_batch),
        "slot1_parse_ok": obj is not None,
        "slot1_parse_method": method,
        "slot1_json": obj,
        "slot1_score": sc,
        "slot0_text_identical": s0_solo["text"] == s0_batch["text"],
        "ratios": {
            "completion": s0_batch["completion_ms"] / s0_solo["completion_ms"],
            "ttft": s0_batch["ttft_ms"] / s0_solo["ttft_ms"],
            "tok_s": (s0_batch["tok_s"] / s0_solo["tok_s"]
                      if s0_solo["tok_s"] > 0 else None),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true", help="print plan only")
    ap.add_argument("--rescore", action="store_true",
                    help="re-analyze saved raw CLI JSON without re-running the GPU")
    args = ap.parse_args()
    seeds = SEEDS[: args.reps]
    if args.dry_run:
        print("plan:", seeds, "solo", SOLO_TSV, "batch", BATCH_TSV)
        return 0

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    reps = []
    for seed in seeds:
        if args.rescore:
            solo = json.loads((RAW_DIR / f"rep{seed}_solo.json").read_text(encoding="utf-8"))
            batch = json.loads((RAW_DIR / f"rep{seed}_batch.json").read_text(encoding="utf-8"))
        else:
            print(f"[run] seed={seed} solo (serial)", flush=True)
            solo = run_cli("serial", SOLO_TSV, seed, RAW_DIR / f"rep{seed}_solo.json")
            print(f"[run] seed={seed} batch k=2", flush=True)
            batch = run_cli("batch", BATCH_TSV, seed, RAW_DIR / f"rep{seed}_batch.json")
        rep = build_rep(seed, solo, batch)
        reps.append(rep)
        r = rep["ratios"]
        print(f"[rep {seed}] parse={rep['slot1_parse_ok']} "
              f"comp_ratio={r['completion']:.3f} ttft_ratio={r['ttft']:.3f} "
              f"s1_tok={rep['batch_slot1']['generated']} "
              f"s1_finish={rep['batch_slot1']['finish']}", flush=True)

    parsed = [r for r in reps if r["slot1_parse_ok"]]
    parse_rate = len(parsed) / len(reps)
    comp_ratios = [r["ratios"]["completion"] for r in reps]
    ttft_ratios = [r["ratios"]["ttft"] for r in reps]
    facts_tp = sum(r["slot1_score"]["facts_tp"] for r in reps if r["slot1_score"])
    facts_fp = sum(r["slot1_score"]["facts_fp"] for r in reps if r["slot1_score"])
    inv_tp = sum(r["slot1_score"]["inv_tp"] for r in reps if r["slot1_score"])
    inv_fp = sum(r["slot1_score"]["inv_fp"] for r in reps if r["slot1_score"])
    wrong_list = sum(len(r["slot1_score"]["wrong_list"]) for r in reps if r["slot1_score"])
    n_facts = len(FACTS_LABEL) * len(reps)
    n_inv = len(INVALIDATED_LABEL) * len(reps)
    concept_hits = {name: 0 for name in ALL_LABEL}
    for r in reps:
        if not r["slot1_score"]:
            continue
        for name, lists in r["slot1_score"]["coverage"].items():
            exp_list = "facts" if name in FACTS_LABEL else "invalidated"
            if exp_list in lists:
                concept_hits[name] += 1

    med_comp, mean_comp = med(comp_ratios), statistics.mean(comp_ratios)
    med_ttft, mean_ttft = med(ttft_ratios), statistics.mean(ttft_ratios)

    if parse_rate < 0.80 or med_comp > 1.25:
        verdict = "dead-end"
    elif parse_rate >= 0.95 and med_comp <= 1.10:
        verdict = "meets"
    else:
        verdict = "borderline"
    why = {
        "meets": "parse rate >=95% and Slot 0 median completion ratio <=1.10x solo",
        "borderline": "between success and dead-end thresholds",
        "dead-end": "parse rate <80% or Slot 0 median ratio >1.25x",
    }[verdict]

    summary = {
        "experiment": "in-flight memory extraction (k=2, Slot 0 chat + Slot 1 Mem0 JSON)",
        "model": MODEL, "system": SYSTEM, "seeds": seeds,
        "max_tokens": MAX_TOKENS, "temperature": TEMP, "top_p": TOP_P,
        "n_reps": len(reps),
        "parse_rate": parse_rate,
        "field_accuracy": {
            "facts_tp": facts_tp, "facts_fp": facts_fp, "facts_possible": n_facts,
            "invalidated_tp": inv_tp, "invalidated_fp": inv_fp,
            "invalidated_possible": n_inv,
            "wrong_list_mentions": wrong_list,
            "concept_hits_in_expected_list": concept_hits,
        },
        "latency": {
            "completion_ratio_median": med_comp, "completion_ratio_mean": mean_comp,
            "ttft_ratio_median": med_ttft, "ttft_ratio_mean": mean_ttft,
            "solo_completion_ms_median": med([r["solo_slot0"]["completion_ms"] for r in reps]),
            "batch_slot0_completion_ms_median": med([r["batch_slot0"]["completion_ms"] for r in reps]),
            "solo_ttft_ms_median": med([r["solo_slot0"]["ttft_ms"] for r in reps]),
            "batch_slot0_ttft_ms_median": med([r["batch_slot0"]["ttft_ms"] for r in reps]),
            "per_rep_completion_ratios": comp_ratios,
            "per_rep_ttft_ratios": ttft_ratios,
        },
        "slot1_tokens": [r["batch_slot1"]["generated"] for r in reps],
        "slot1_finish": [r["batch_slot1"]["finish"] for r in reps],
        "slot0_text_identical_all_reps": all(r["slot0_text_identical"] for r in reps),
        "verdict": verdict, "verdict_reason": why,
        "reps": reps,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lat = summary["latency"]
    L = []
    L.append("In-flight memory extraction: k=2 parallel::Runtime")
    L.append(f"model={MODEL}  max_tokens={MAX_TOKENS} temp={TEMP} top_p={TOP_P}")
    L.append(f"reps={len(reps)} seeds={seeds}  solo=serial(Slot0)  batch=k=2(Slot0+Slot1)")
    L.append("")
    L.append(f"{'seed':>6} | {'s0_solo_ms':>10} {'s0_batch_ms':>11} {'comp_r':>7} "
             f"{'ttft_r':>7} {'s0_same':>7} | {'s1_tok':>6} {'s1_fin':>11} "
             f"{'parse':>5} {'factTP/FP':>9} {'invTP/FP':>8}")
    for r in reps:
        sc = r["slot1_score"] or {}
        fcell = f"{sc.get('facts_tp', 0)}/{sc.get('facts_fp', 0)}"
        icell = f"{sc.get('inv_tp', 0)}/{sc.get('inv_fp', 0)}"
        L.append(f"{r['seed']:>6} | {r['solo_slot0']['completion_ms']:>10.1f} "
                 f"{r['batch_slot0']['completion_ms']:>11.1f} "
                 f"{r['ratios']['completion']:>7.3f} {r['ratios']['ttft']:>7.3f} "
                 f"{str(r['slot0_text_identical']):>7} | {r['batch_slot1']['generated']:>6} "
                 f"{r['batch_slot1']['finish']:>11} {str(r['slot1_parse_ok']):>5} "
                 f"{fcell:>9} {icell:>8}")
    L.append("")
    L.append(f"Slot 1 parse rate: {parse_rate:.0%} "
             f"({len(parsed)}/{len(reps)})  methods="
             f"{[r['slot1_parse_method'] for r in reps]}")
    L.append(f"Slot 1 facts TP/FP: {facts_tp}/{facts_fp} of {n_facts} possible; "
             f"invalidated TP/FP: {inv_tp}/{inv_fp} of {n_inv} possible; "
             f"wrong-list mentions: {wrong_list}")
    L.append("Concept covered in its expected list (reps): "
             + ", ".join(f"{k}={v}/{len(reps)}" for k, v in concept_hits.items()))
    L.append(f"Slot 0 completion ratio batch/solo: median={med_comp:.3f} mean={mean_comp:.3f} "
             f"(solo med {lat['solo_completion_ms_median']:.1f} ms, "
             f"batch med {lat['batch_slot0_completion_ms_median']:.1f} ms)")
    L.append(f"Slot 0 ttft ratio batch/solo:       median={med_ttft:.3f} mean={mean_ttft:.3f} "
             f"(solo med {lat['solo_ttft_ms_median']:.2f} ms, "
             f"batch med {lat['batch_slot0_ttft_ms_median']:.2f} ms)")
    L.append(f"Slot 1 tokens per rep: {summary['slot1_tokens']} finish={summary['slot1_finish']}")
    L.append(f"Slot 0 text identical solo vs batch (same seed): "
             f"{summary['slot0_text_identical_all_reps']}")
    L.append(f"VERDICT: {verdict} - {why}")
    text = "\n".join(L) + "\n"
    OUT_TXT.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
