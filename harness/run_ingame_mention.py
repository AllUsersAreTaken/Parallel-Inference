"""Angle 30 - must-mention probe (RecycleLM, follow-up to angle 28).

Angle 28 failed two ways: every seed silently dropped Honda from "invalidated"
(0/5), and its few-shot anchor induced markdown fences + // comments (direct
json.loads 0/5; fence-stripped parse only 4/5). This probe bans few-shot
examples and fences entirely.

Slot 1 user text (data/ingame_mention_batch.tsv): raw-JSON-only rule, schema
{"previous": [...], "current": [...]}, ending with a literal `{` after
"Output:". Per app/parallel_cli.cpp, prompts pass through unescape() then
rt.format_chat({system, user}) which calls llama_chat_apply_template with
add_assistant=false -- so the trailing `{` lands INSIDE the user turn (legal,
unmangled), not as a true assistant prefill. Noted, not a CLI change.

Protocol identical to angles 27/28: k=2, seeds 1234-1238, temp 0.2,
max_tokens=160, system="You are a helpful assistant.", fresh solo arm
(serial, Slot 0 only; data/ingame_mem_solo.tsv byte-identical to prior
angles, verified at run time).

Scoring is STRICT: direct json.loads on the raw Slot 1 text -- no fence
stripping, no brace scan, no comment stripping. Also records: Honda in
"previous"; Tesla in "current"; wrong-list placement; old-schema field
fallback (facts/invalidated keys); "Honda anywhere in output" (drop vs
misplace).

Gates: Honda-in-previous >=4/5; direct parse 5/5; Slot 0 completion ratio
(batch/solo) median <= 1.10x.

Writes results/ingame_mention_probe.json and results/ingame_mention_probe.txt.
Raw per-rep CLI JSON in results/ingame_mention_probe_raw/.
Run: python harness\\run_ingame_mention.py [--reps N] [--rescore] [--dry-run]
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_ingame_memory import (  # noqa: E402
    MAX_TOKENS, MODEL, ROOT, SEEDS, SOLO_TSV, SYSTEM, TEMP, TOP_P, get_stream, med,
    run_cli, summarize_branch,
)

BATCH_TSV = "data/ingame_mention_batch.tsv"
RAW_DIR = ROOT / "results" / "ingame_mention_probe_raw"
OUT_JSON = ROOT / "results" / "ingame_mention_probe.json"
OUT_TXT = ROOT / "results" / "ingame_mention_probe.txt"
S1_MARKER = "Extract state transitions"


def strict_parse(text):
    """STRICT direct json.loads on raw output. No fences/braces/comments help."""
    try:
        return json.loads(text)
    except ValueError:
        return None


def list_has(obj, key, needle):
    raw = obj.get(key) if isinstance(obj, dict) else None
    if not isinstance(raw, list):
        return False
    return any(needle in str(s).lower() for s in raw)


def build_rep(seed, solo, batch):
    s0_solo = get_stream(solo, "Reply to the user")
    s0_batch = get_stream(batch, "Reply to the user")
    s1_batch = get_stream(batch, S1_MARKER)

    raw = s1_batch["text"]
    obj = strict_parse(raw)
    old_fields = ([k for k in ("facts", "invalidated")
                   if isinstance(obj, dict) and k in obj]
                  if obj is not None else [])
    rep = {
        "seed": seed,
        "solo_slot0": summarize_branch(s0_solo),
        "batch_slot0": summarize_branch(s0_batch),
        "batch_slot1": summarize_branch(s1_batch),
        "s0_prompt_identical": s0_solo["prompt"] == s0_batch["prompt"],
        "slot1_direct_parse_ok": obj is not None,
        "slot1_json": obj if obj is not None else None,
        "honda_anywhere": "honda" in raw.lower(),
        "tesla_anywhere": "tesla" in raw.lower(),
        "fence_anywhere": "```" in raw,
        "comment_anywhere": "//" in raw or "/*" in raw,
        "old_field_keys": old_fields,
        "ratios": {
            "completion": s0_batch["completion_ms"] / s0_solo["completion_ms"],
            "ttft": s0_batch["ttft_ms"] / s0_solo["ttft_ms"],
        },
    }
    if obj is not None:
        rep["honda_in_previous"] = list_has(obj, "previous", "honda")
        rep["tesla_in_current"] = list_has(obj, "current", "tesla")
        rep["honda_in_current"] = list_has(obj, "current", "honda")
        rep["tesla_in_previous"] = list_has(obj, "previous", "tesla")
        rep["wrong_list"] = rep["honda_in_current"] or rep["tesla_in_previous"]
    else:
        rep["honda_in_previous"] = False
        rep["tesla_in_current"] = False
        rep["honda_in_current"] = None
        rep["tesla_in_previous"] = None
        rep["wrong_list"] = None
    return rep


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
            print(f"[run] seed={seed} batch k=2 (mention probe)", flush=True)
            batch = run_cli("batch", BATCH_TSV, seed, RAW_DIR / f"rep{seed}_batch.json")
        rep = build_rep(seed, solo, batch)
        reps.append(rep)
        r = rep["ratios"]
        print(f"[rep {seed}] parse={rep['slot1_direct_parse_ok']} "
              f"honda_prev={rep['honda_in_previous']} tesla_cur={rep['tesla_in_current']} "
              f"honda_any={rep['honda_anywhere']} wrong_list={rep['wrong_list']} "
              f"old_fields={rep['old_field_keys']} comp_ratio={r['completion']:.3f} "
              f"s1_tok={rep['batch_slot1']['generated']} "
              f"s1_finish={rep['batch_slot1']['finish']}", flush=True)

    n = len(reps)
    parsed = [r for r in reps if r["slot1_direct_parse_ok"]]
    parse_rate = len(parsed) / n
    honda_prev = sum(r["honda_in_previous"] for r in reps)
    tesla_cur = sum(r["tesla_in_current"] for r in reps)
    honda_any = sum(r["honda_anywhere"] for r in reps)
    wrong_list = sum(bool(r["wrong_list"]) for r in reps if r["wrong_list"] is not None)
    old_field_seeds = sum(1 for r in reps if r["old_field_keys"])
    comp_ratios = [r["ratios"]["completion"] for r in reps]
    ttft_ratios = [r["ratios"]["ttft"] for r in reps]
    med_comp, mean_comp = med(comp_ratios), statistics.mean(comp_ratios)
    med_ttft, mean_ttft = med(ttft_ratios), statistics.mean(ttft_ratios)

    criteria = {
        "honda_in_previous_ge_4_of_5": honda_prev >= 4 if n == 5 else honda_prev / n >= 0.80,
        "direct_json_parse_5_of_5": len(parsed) == n,
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
        "experiment": "Angle 30 must-mention probe (k=2, Slot 0 chat + Slot 1 raw-JSON previous/current)",
        "parent": "results/ingame_memory_nudge.json (angle 28)",
        "model": MODEL, "system": SYSTEM, "seeds": seeds,
        "max_tokens": MAX_TOKENS, "temperature": TEMP, "top_p": TOP_P,
        "n_reps": n,
        "data_files": {"solo": SOLO_TSV, "batch": BATCH_TSV,
                       "note": "s0 prompt byte-identical to angle 27 (data/ingame_mem_solo.tsv); "
                               "s1 ends with literal `{` inside the user turn "
                               "(format_chat add_assistant=false, not assistant prefill)"},
        "scoring": "STRICT direct json.loads on raw Slot 1 text; no fence strip, "
                   "no brace scan, no comment strip",
        "parse_rate": parse_rate,
        "acceptance": criteria,
        "honda_in_previous_hits": honda_prev,
        "tesla_in_current_hits": tesla_cur,
        "honda_anywhere_hits": honda_any,
        "wrong_list_seeds": wrong_list,
        "old_field_fallback_seeds": old_field_seeds,
        "latency": lat,
        "slot1_tokens": [r["batch_slot1"]["generated"] for r in reps],
        "slot1_finish": [r["batch_slot1"]["finish"] for r in reps],
        "s0_prompt_identical_all_reps": all(r["s0_prompt_identical"] for r in reps),
        "verdict": verdict,
        "reps": reps,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    L = []
    L.append("Angle 30 - must-mention probe: k=2 parallel::Runtime")
    L.append(f"model={MODEL}  max_tokens={MAX_TOKENS} temp={TEMP} top_p={TOP_P} system={SYSTEM!r}")
    L.append(f"reps={n} seeds={seeds}  solo=serial({SOLO_TSV})  batch=k=2({BATCH_TSV})")
    L.append("Slot 1 = raw-JSON-only rule, schema previous/current, no few-shot, trailing `{` in user turn")
    L.append("Scoring: STRICT direct json.loads (no fence strip / brace scan / comment strip)")
    L.append("")
    L.append(f"{'seed':>6} | {'s0_solo_ms':>10} {'s0_batch_ms':>11} {'comp_r':>7} "
             f"{'ttft_r':>7} | {'s1_tok':>6} {'s1_fin':>11} {'parse':>5} "
             f"{'hondPrev':>8} {'tesCur':>6} {'hondAny':>7} {'wrongL':>6} {'oldFld':>6}")
    for r in reps:
        L.append(f"{r['seed']:>6} | {r['solo_slot0']['completion_ms']:>10.1f} "
                 f"{r['batch_slot0']['completion_ms']:>11.1f} "
                 f"{r['ratios']['completion']:>7.3f} {r['ratios']['ttft']:>7.3f} | "
                 f"{r['batch_slot1']['generated']:>6} {r['batch_slot1']['finish']:>11} "
                 f"{str(r['slot1_direct_parse_ok']):>5} "
                 f"{str(r['honda_in_previous']):>8} {str(r['tesla_in_current']):>6} "
                 f"{str(r['honda_anywhere']):>7} {str(r['wrong_list']):>6} "
                 f"{','.join(r['old_field_keys']) or '-':>6}")
    L.append(f"{'median':>6} | {lat['solo_completion_ms_median']:>10.1f} "
             f"{lat['batch_slot0_completion_ms_median']:>11.1f} {med_comp:>7.3f} "
             f"{med_ttft:>7.3f} | {'':>6} {'':>11} {'':>5} "
             f"{honda_prev}/{n}{'':>4} {tesla_cur}/{n}  {honda_any}/{n}")
    L.append("")
    L.append(f"Slot 1 STRICT direct parse rate: {parse_rate:.0%} ({len(parsed)}/{n})")
    L.append(f"Honda in previous: {honda_prev}/{n}  (gate >=4/5)   "
             f"Tesla in current: {tesla_cur}/{n}   wrong-list seeds: {wrong_list}   "
             f"old-schema field fallback seeds: {old_field_seeds}")
    L.append(f"Honda anywhere in raw output: {honda_any}/{n} "
             f"(misplace vs drop discriminator)")
    L.append(f"fence in output: {sum(r['fence_anywhere'] for r in reps)}/{n}  "
             f"comment marker in output: {sum(r['comment_anywhere'] for r in reps)}/{n}")
    L.append(f"Slot 1 tokens per rep: {summary['slot1_tokens']} finish={summary['slot1_finish']}")
    L.append(f"Slot 0 completion ratio batch/solo: median={med_comp:.3f} mean={mean_comp:.3f} "
             f"(solo med {lat['solo_completion_ms_median']:.1f} ms, "
             f"batch med {lat['batch_slot0_completion_ms_median']:.1f} ms, "
             f"ratio of medians {lat['ratio_of_completion_medians']:.3f})")
    L.append(f"Slot 0 ttft ratio batch/solo:       median={med_ttft:.3f} mean={mean_ttft:.3f}")
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
