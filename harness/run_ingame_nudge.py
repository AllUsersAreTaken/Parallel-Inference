"""Phase 1 - invalidation-nudge probe (RecycleLM, follow-up to angle 27).

Protocol identical to harness/run_ingame_memory.py: k=2 on parallel::Runtime,
Slot 0 = conversational reply, Slot 1 = Mem0-style JSON extraction, 5 solo
(serial, Slot 0 only) vs 5 batch (k=2) reps, seeds 1234-1238, max_tokens=160,
temp=0.2, top_p=0.9, system="You are a helpful assistant.".

Change vs angle 27: Slot 1's user text carries a nudge rule + 1-shot anchor
(data/ingame_nudge_batch.tsv) so superseded prior state lands in "invalidated"
and current state in "facts". Slot 0's prompt is byte-identical to angle 27
(reuses data/ingame_mem_solo.tsv; verified at run time via prompt equality).

Scoring/parse/run helpers are imported from run_ingame_memory (single source
of truth for the TP/FP label logic).

Acceptance (checked in this run):
  - user_drives_honda recovered in "invalidated" in >=4/5 reps
  - Slot 1 JSON parses 5/5
  - Slot 0 completion ratio (batch/solo) median <= 1.10x

Writes results/ingame_memory_nudge.json and results/ingame_memory_nudge.txt.
Raw per-rep CLI JSON in results/ingame_memory_nudge_raw/.
Run: python harness\\run_ingame_nudge.py [--reps N] [--rescore] [--dry-run]
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_ingame_memory import (  # noqa: E402
    FACTS_LABEL, INVALIDATED_LABEL, MAX_TOKENS, MODEL, ROOT, SEEDS, SOLO_TSV,
    SYSTEM, TEMP, TOP_P, get_stream, med, parse_slot1_json, run_cli, score_slot1,
    summarize_branch,
)

BATCH_TSV = "data/ingame_nudge_batch.tsv"
RAW_DIR = ROOT / "results" / "ingame_memory_nudge_raw"
OUT_JSON = ROOT / "results" / "ingame_memory_nudge.json"
OUT_TXT = ROOT / "results" / "ingame_memory_nudge.txt"
S1_MARKER = "Extract memory updates from the conversation"
N_FACTS = len(FACTS_LABEL)
N_INV = len(INVALIDATED_LABEL)


def build_rep(seed, solo, batch):
    s0_solo = get_stream(solo, "Reply to the user")
    s0_batch = get_stream(batch, "Reply to the user")
    s1_batch = get_stream(batch, S1_MARKER)

    obj, method = parse_slot1_json(s1_batch["text"])
    sc = score_slot1(obj) if isinstance(obj, dict) else None
    cov = sc["coverage"] if sc else {}
    return {
        "seed": seed,
        "solo_slot0": summarize_branch(s0_solo),
        "batch_slot0": summarize_branch(s0_batch),
        "batch_slot1": summarize_branch(s1_batch),
        "s0_prompt_identical": s0_solo["prompt"] == s0_batch["prompt"],
        "slot1_parse_ok": obj is not None,
        "slot1_parse_method": method,
        "slot1_json": obj,
        "slot1_score": sc,
        "honda_in_invalidated": "invalidated" in cov.get("user_drives_honda", []),
        "tesla_in_facts": "facts" in cov.get("user_bought_tesla", []),
        "ratios": {
            "completion": s0_batch["completion_ms"] / s0_solo["completion_ms"],
            "ttft": s0_batch["ttft_ms"] / s0_solo["ttft_ms"],
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true")
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
            print(f"[run] seed={seed} solo (serial, Slot 0)", flush=True)
            solo = run_cli("serial", SOLO_TSV, seed, RAW_DIR / f"rep{seed}_solo.json")
            print(f"[run] seed={seed} batch k=2 (nudge)", flush=True)
            batch = run_cli("batch", BATCH_TSV, seed, RAW_DIR / f"rep{seed}_batch.json")
        rep = build_rep(seed, solo, batch)
        reps.append(rep)
        r = rep["ratios"]
        print(f"[rep {seed}] parse={rep['slot1_parse_ok']} "
              f"honda_inv={rep['honda_in_invalidated']} tesla_fact={rep['tesla_in_facts']} "
              f"comp_ratio={r['completion']:.3f} ttft_ratio={r['ttft']:.3f} "
              f"s1_tok={rep['batch_slot1']['generated']} "
              f"s1_finish={rep['batch_slot1']['finish']}", flush=True)

    n = len(reps)
    parsed = [r for r in reps if r["slot1_parse_ok"]]
    parse_rate = len(parsed) / n
    inv_hits = sum(r["honda_in_invalidated"] for r in reps)
    inv_hit_rate = inv_hits / n
    facts_hits = sum(r["tesla_in_facts"] for r in reps)
    comp_ratios = [r["ratios"]["completion"] for r in reps]
    ttft_ratios = [r["ratios"]["ttft"] for r in reps]
    med_comp, mean_comp = med(comp_ratios), statistics.mean(comp_ratios)
    med_ttft, mean_ttft = med(ttft_ratios), statistics.mean(ttft_ratios)

    facts_tp = sum(r["slot1_score"]["facts_tp"] for r in reps if r["slot1_score"])
    facts_fp = sum(r["slot1_score"]["facts_fp"] for r in reps if r["slot1_score"])
    inv_tp = sum(r["slot1_score"]["inv_tp"] for r in reps if r["slot1_score"])
    inv_fp = sum(r["slot1_score"]["inv_fp"] for r in reps if r["slot1_score"])
    wrong_list = sum(len(r["slot1_score"]["wrong_list"]) for r in reps if r["slot1_score"])
    concept_hits = {name: 0 for name in {**FACTS_LABEL, **INVALIDATED_LABEL}}
    for r in reps:
        if not r["slot1_score"]:
            continue
        for name, lists in r["slot1_score"]["coverage"].items():
            exp_list = "facts" if name in FACTS_LABEL else "invalidated"
            if exp_list in lists:
                concept_hits[name] += 1

    criteria = {
        "invalidated_honda_ge_4_of_5": inv_hits >= 4 if n == 5 else inv_hit_rate >= 0.80,
        "json_parse_5_of_5": len(parsed) == n,
        "slot0_completion_median_le_1_10": med_comp <= 1.10,
    }
    verdict = "meets" if all(criteria.values()) else "miss"

    lat = {
        "completion_ratio_median": med_comp, "completion_ratio_mean": mean_comp,
        "ttft_ratio_median": med_ttft, "ttft_ratio_mean": mean_ttft,
        "solo_completion_ms_median": med([r["solo_slot0"]["completion_ms"] for r in reps]),
        "batch_slot0_completion_ms_median": med([r["batch_slot0"]["completion_ms"] for r in reps]),
        "solo_ttft_ms_median": med([r["solo_slot0"]["ttft_ms"] for r in reps]),
        "batch_slot0_ttft_ms_median": med([r["batch_slot0"]["ttft_ms"] for r in reps]),
        "ratio_of_completion_medians": (
            med([r["batch_slot0"]["completion_ms"] for r in reps])
            / med([r["solo_slot0"]["completion_ms"] for r in reps])),
        "per_rep_completion_ratios": comp_ratios,
        "per_rep_ttft_ratios": ttft_ratios,
    }

    summary = {
        "experiment": "Phase 1 invalidation-nudge probe (k=2, Slot 0 chat + Slot 1 nudged Mem0 JSON)",
        "parent": "results/ingame_memory_test.json (angle 27)",
        "model": MODEL, "system": SYSTEM, "seeds": seeds,
        "max_tokens": MAX_TOKENS, "temperature": TEMP, "top_p": TOP_P,
        "n_reps": n,
        "data_files": {"solo": SOLO_TSV, "batch": BATCH_TSV,
                       "note": "s0 prompt byte-identical to angle 27 (data/ingame_mem_solo.tsv)"},
        "nudge_location": "inside Slot 1 user text (chat template wraps system+user)",
        "parse_rate": parse_rate,
        "acceptance": criteria,
        "invalidated_honda_hits": inv_hits,
        "invalidated_honda_hit_rate": inv_hit_rate,
        "facts_tesla_hits": facts_hits,
        "field_accuracy": {
            "facts_tp": facts_tp, "facts_fp": facts_fp, "facts_possible": N_FACTS * n,
            "invalidated_tp": inv_tp, "invalidated_fp": inv_fp,
            "invalidated_possible": N_INV * n,
            "wrong_list_mentions": wrong_list,
            "concept_hits_in_expected_list": concept_hits,
        },
        "latency": lat,
        "slot1_tokens": [r["batch_slot1"]["generated"] for r in reps],
        "slot1_finish": [r["batch_slot1"]["finish"] for r in reps],
        "s0_prompt_identical_all_reps": all(r["s0_prompt_identical"] for r in reps),
        "verdict": verdict,
        "reps": reps,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    L = []
    L.append("Phase 1 - invalidation-nudge probe: k=2 parallel::Runtime")
    L.append(f"model={MODEL}  max_tokens={MAX_TOKENS} temp={TEMP} top_p={TOP_P} system={SYSTEM!r}")
    L.append(f"reps={n} seeds={seeds}  solo=serial({SOLO_TSV})  batch=k=2({BATCH_TSV})")
    L.append("Slot 1 user text = angle-27 instruction + nudge rule + 1-shot anchor (invalidated placement)")
    L.append("")
    L.append(f"{'seed':>6} | {'s0_solo_ms':>10} {'s0_batch_ms':>11} {'comp_r':>7} "
             f"{'ttft_r':>7} | {'s1_tok':>6} {'s1_fin':>11} {'parse':>5} "
             f"{'factsRec':>8} {'invRec':>6} {'hondaInv':>8} {'teslaFct':>8}")
    for r in reps:
        sc = r["slot1_score"] or {}
        L.append(f"{r['seed']:>6} | {r['solo_slot0']['completion_ms']:>10.1f} "
                 f"{r['batch_slot0']['completion_ms']:>11.1f} "
                 f"{r['ratios']['completion']:>7.3f} {r['ratios']['ttft']:>7.3f} | "
                 f"{r['batch_slot1']['generated']:>6} {r['batch_slot1']['finish']:>11} "
                 f"{str(r['slot1_parse_ok']):>5} "
                 f"{sc.get('facts_tp', 0)}/{N_FACTS:>6} {sc.get('inv_tp', 0)}/{N_INV:>4} "
                 f"{str(r['honda_in_invalidated']):>8} {str(r['tesla_in_facts']):>8}")
    L.append(f"{'median':>6} | {lat['solo_completion_ms_median']:>10.1f} "
             f"{lat['batch_slot0_completion_ms_median']:>11.1f} {med_comp:>7.3f} "
             f"{med_ttft:>7.3f} | {'':>6} {'':>11} {'':>5} {'':>8} {'':>6} "
             f"{inv_hits}/{n}{'':>3} {facts_hits}/{n}")
    L.append("")
    L.append(f"Slot 1 parse rate: {parse_rate:.0%} ({len(parsed)}/{n})  "
             f"methods={[r['slot1_parse_method'] for r in reps]}")
    L.append(f"invalidated Honda hits: {inv_hits}/{n}  (accept >=4/5)   "
             f"facts Tesla hits: {facts_hits}/{n}")
    L.append(f"Slot 1 facts TP/FP: {facts_tp}/{facts_fp} of {N_FACTS * n} possible; "
             f"invalidated TP/FP: {inv_tp}/{inv_fp} of {N_INV * n} possible; "
             f"wrong-list mentions: {wrong_list}")
    L.append("Concept covered in its expected list (reps): "
             + ", ".join(f"{k}={v}/{n}" for k, v in concept_hits.items()))
    L.append(f"Slot 0 completion ratio batch/solo: median={med_comp:.3f} mean={mean_comp:.3f} "
             f"(solo med {lat['solo_completion_ms_median']:.1f} ms, "
             f"batch med {lat['batch_slot0_completion_ms_median']:.1f} ms, "
             f"ratio of medians {lat['ratio_of_completion_medians']:.3f})")
    L.append(f"Slot 0 ttft ratio batch/solo:       median={med_ttft:.3f} mean={mean_ttft:.3f} "
             f"(solo med {lat['solo_ttft_ms_median']:.2f} ms, "
             f"batch med {lat['batch_slot0_ttft_ms_median']:.2f} ms)")
    L.append(f"Slot 1 tokens per rep: {summary['slot1_tokens']} finish={summary['slot1_finish']}")
    L.append(f"Slot 0 prompt identical solo vs batch (same seed): "
             f"{summary['s0_prompt_identical_all_reps']}")
    L.append("ACCEPTANCE: " + ", ".join(f"{k}={'PASS' if v else 'FAIL'}"
                                        for k, v in criteria.items()))
    L.append(f"VERDICT: {verdict}")
    text = "\n".join(L) + "\n"
    OUT_TXT.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
