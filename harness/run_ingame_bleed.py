"""Angle 31 - attribute-bleed gate (RecycleLM, follow-up to angle 29).

Angle 29 (Letta-style block compaction) passed all original Phase 2 checks
(tesla present, honda absent, seattle/buster preserved, 34 tokens, Slot 0
median 1.055x) but every seed leaked the replaced item's year: output read
"Vehicle: 2019 Tesla". This gate re-runs the identical compaction arm with
one added sentence in the Slot 1 instruction:

  "Do not carry over adjectives, years, or specs from replaced items unless
   explicitly stated."

data/ingame_bleed_batch.tsv / data/ingame_bleed_solo.tsv differ from
data/ingame_compact_*.tsv ONLY by that inserted sentence (verified at
generation time). Slot 0 prompt is byte-identical to angle 27
(data/ingame_mem_solo.tsv).

Protocol identical to angle 29: seeds 1234-1238, temp 0.2, max_tokens=160;
per seed: serial solo0 (Slot 0 only), serial solo1 (Slot 1 compaction at
seed+1 to match batch row-1 sampling seed), batch k=2.

Per-seed checks: tesla present; honda/civic absent; seattle preserved;
buster preserved; profile tokens <75; "2019 Tesla" absent (the new gate);
year_survives records whether "2019" appears anywhere in the profile.

Gates: "2019 Tesla" absent 0/5; original Phase 2 per-seed checks all pass
(context); Slot 0 completion ratio median <= 1.10x.

Writes results/ingame_bleed_gate.json and results/ingame_bleed_gate.txt.
Raw per-rep CLI JSON in results/ingame_bleed_gate_raw/.
Run: python harness\\run_ingame_bleed.py [--reps N] [--rescore] [--dry-run]
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

BATCH_TSV = "data/ingame_bleed_batch.tsv"
SOLO1_TSV = "data/ingame_bleed_solo.tsv"
RAW_DIR = ROOT / "results" / "ingame_bleed_gate_raw"
OUT_JSON = ROOT / "results" / "ingame_bleed_gate.json"
OUT_TXT = ROOT / "results" / "ingame_bleed_gate.txt"
S1_MARKER = "[User Profile]"
PROFILE_TOKEN_MAX = 75


def bleed_checks(text):
    t = text.lower()
    year_lines = [ln.strip() for ln in text.splitlines() if "2019" in ln]
    return {
        "tesla_present": "tesla" in t,
        "honda_absent": "honda" not in t and "civic" not in t,
        "seattle_preserved": "seattle" in t,
        "buster_preserved": "buster" in t,
        "tesla_2019_absent": "2019 tesla" not in t,
        "year_survives": "2019" in t,
        "year_lines": year_lines,
    }


def build_rep(seed, solo0, solo1, batch):
    s0_solo = get_stream(solo0, "Reply to the user")
    s0_batch = get_stream(batch, "Reply to the user")
    s1_solo = get_stream(solo1, S1_MARKER)
    s1_batch = get_stream(batch, S1_MARKER)

    checks = bleed_checks(s1_batch["text"])
    tokens = s1_batch["generated"]
    phase2_keys = ("tesla_present", "honda_absent", "seattle_preserved", "buster_preserved")
    phase2_ok = all(checks[k] for k in phase2_keys)
    return {
        "seed": seed,
        "solo_slot0": summarize_branch(s0_solo),
        "batch_slot0": summarize_branch(s0_batch),
        "solo_slot1": summarize_branch(s1_solo),
        "batch_slot1": summarize_branch(s1_batch),
        "s0_prompt_identical": s0_solo["prompt"] == s0_batch["prompt"],
        "checks": checks,
        "phase2_checks_ok": phase2_ok,
        "profile_tokens": tokens,
        "profile_words": len(s1_batch["text"].split()),
        "profile_token_ok": tokens < PROFILE_TOKEN_MAX,
        "seed_ok": (phase2_ok and tokens < PROFILE_TOKEN_MAX
                    and checks["tesla_2019_absent"]),
        "ratios": {
            "completion": s0_batch["completion_ms"] / s0_solo["completion_ms"],
            "ttft": s0_batch["ttft_ms"] / s0_solo["ttft_ms"],
            "slot1_completion": (s1_batch["completion_ms"] / s1_solo["completion_ms"]
                                 if s1_solo["completion_ms"] > 0 else None),
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
        print("plan:", seeds, "solo0", SOLO_TSV, "solo1", SOLO1_TSV, "batch", BATCH_TSV)
        return 0

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    reps = []
    for seed in seeds:
        if args.rescore:
            solo0 = json.loads((RAW_DIR / f"rep{seed}_solo0.json").read_text(encoding="utf-8"))
            solo1 = json.loads((RAW_DIR / f"rep{seed}_solo1.json").read_text(encoding="utf-8"))
            batch = json.loads((RAW_DIR / f"rep{seed}_batch.json").read_text(encoding="utf-8"))
        else:
            print(f"[run] seed={seed} solo0 (serial, Slot 0)", flush=True)
            solo0 = run_cli("serial", SOLO_TSV, seed, RAW_DIR / f"rep{seed}_solo0.json")
            print(f"[run] seed={seed} solo1 (serial, Slot 1 bleed-compact, seed+1)", flush=True)
            solo1 = run_cli("serial", SOLO1_TSV, seed + 1, RAW_DIR / f"rep{seed}_solo1.json")
            print(f"[run] seed={seed} batch k=2 (bleed-compact)", flush=True)
            batch = run_cli("batch", BATCH_TSV, seed, RAW_DIR / f"rep{seed}_batch.json")
        rep = build_rep(seed, solo0, solo1, batch)
        reps.append(rep)
        r = rep["ratios"]
        c = rep["checks"]
        print(f"[rep {seed}] ok={rep['seed_ok']} tesla={c['tesla_present']} "
              f"honda_absent={c['honda_absent']} seattle={c['seattle_preserved']} "
              f"buster={c['buster_preserved']} 2019tesla_absent={c['tesla_2019_absent']} "
              f"year_survives={c['year_survives']} tok={rep['profile_tokens']} "
              f"comp_ratio={r['completion']:.3f} "
              f"finish={rep['batch_slot1']['finish']}", flush=True)

    n = len(reps)
    comp_ratios = [r["ratios"]["completion"] for r in reps]
    ttft_ratios = [r["ratios"]["ttft"] for r in reps]
    s1_ratios = [r["ratios"]["slot1_completion"] for r in reps]
    med_comp, mean_comp = med(comp_ratios), statistics.mean(comp_ratios)
    med_ttft, mean_ttft = med(ttft_ratios), statistics.mean(ttft_ratios)
    check_totals = {k: sum(r["checks"][k] for r in reps)
                    for k in ("tesla_present", "honda_absent", "seattle_preserved",
                              "buster_preserved", "tesla_2019_absent", "year_survives")}
    seeds_ok = sum(r["seed_ok"] for r in reps)
    phase2_ok = sum(r["phase2_checks_ok"] for r in reps)
    token_ok = sum(r["profile_token_ok"] for r in reps)

    criteria = {
        "tesla_2019_absent_0_of_5": check_totals["tesla_2019_absent"] == n,
        "original_phase2_checks_all_pass": phase2_ok == n,
        "slot0_completion_median_le_1_10": med_comp <= 1.10,
    }
    verdict = "meets" if all(criteria.values()) else "miss"

    lat = {
        "completion_ratio_median": med_comp, "completion_ratio_mean": mean_comp,
        "ttft_ratio_median": med_ttft, "ttft_ratio_mean": mean_ttft,
        "slot1_completion_ratio_median": med(s1_ratios),
        "solo0_completion_ms_median": med([r["solo_slot0"]["completion_ms"] for r in reps]),
        "batch_slot0_completion_ms_median": med([r["batch_slot0"]["completion_ms"] for r in reps]),
        "solo1_completion_ms_median": med([r["solo_slot1"]["completion_ms"] for r in reps]),
        "batch_slot1_completion_ms_median": med([r["batch_slot1"]["completion_ms"] for r in reps]),
        "solo0_ttft_ms_median": med([r["solo_slot0"]["ttft_ms"] for r in reps]),
        "batch_slot0_ttft_ms_median": med([r["batch_slot0"]["ttft_ms"] for r in reps]),
        "ratio_of_completion_medians": (
            med([r["batch_slot0"]["completion_ms"] for r in reps])
            / med([r["solo_slot0"]["completion_ms"] for r in reps])),
        "per_rep_completion_ratios": comp_ratios,
        "per_rep_ttft_ratios": ttft_ratios,
        "per_rep_slot1_completion_ratios": s1_ratios,
    }

    summary = {
        "experiment": "Angle 31 attribute-bleed gate (k=2, Slot 0 chat + Slot 1 bleed-blocked compaction)",
        "parent": "results/ingame_letta_compact.json (angle 29)",
        "model": MODEL, "system": SYSTEM, "seeds": seeds,
        "max_tokens": MAX_TOKENS, "temperature": TEMP, "top_p": TOP_P,
        "n_reps": n,
        "data_files": {"solo_slot0": SOLO_TSV, "solo_slot1": SOLO1_TSV, "batch": BATCH_TSV,
                       "note": "s0 prompt byte-identical to angle 27; s1 = angle 29 compaction "
                               "prompt + one inserted anti-bleed sentence; solo1 at seed+1"},
        "profile_token_max": PROFILE_TOKEN_MAX,
        "acceptance": criteria,
        "seeds_passing_all_checks": seeds_ok,
        "phase2_checks_passing": phase2_ok,
        "check_totals": check_totals,
        "profile_tokens_ok_seeds": token_ok,
        "profile_tokens": [r["profile_tokens"] for r in reps],
        "profile_words": [r["profile_words"] for r in reps],
        "year_lines": [r["checks"]["year_lines"] for r in reps],
        "slot1_finish": [r["batch_slot1"]["finish"] for r in reps],
        "latency": lat,
        "s0_prompt_identical_all_reps": all(r["s0_prompt_identical"] for r in reps),
        "verdict": verdict,
        "reps": reps,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    L = []
    L.append("Angle 31 - attribute-bleed gate: k=2 parallel::Runtime")
    L.append(f"model={MODEL}  max_tokens={MAX_TOKENS} temp={TEMP} top_p={TOP_P} system={SYSTEM!r}")
    L.append(f"reps={n} seeds={seeds}  solo0=serial({SOLO_TSV})  "
             f"solo1=serial({SOLO1_TSV}, seed+1)  batch=k=2({BATCH_TSV})")
    L.append("Slot 1 = angle 29 compaction + anti-bleed sentence (no adjectives/years/specs from replaced items)")
    L.append("")
    L.append(f"{'seed':>6} | {'s0_solo_ms':>10} {'s0_batch_ms':>11} {'comp_r':>7} "
             f"{'ttft_r':>7} | {'tok':>4} {'fin':>11} {'tesla':>5} {'hondX':>5} "
             f"{'sea':>5} {'bst':>4} {'no19T':>5} {'yrAny':>5} {'ok':>5}")
    for r in reps:
        c = r["checks"]
        L.append(f"{r['seed']:>6} | {r['solo_slot0']['completion_ms']:>10.1f} "
                 f"{r['batch_slot0']['completion_ms']:>11.1f} "
                 f"{r['ratios']['completion']:>7.3f} {r['ratios']['ttft']:>7.3f} | "
                 f"{r['profile_tokens']:>4} {r['batch_slot1']['finish']:>11} "
                 f"{str(c['tesla_present']):>5} {str(c['honda_absent']):>5} "
                 f"{str(c['seattle_preserved']):>5} {str(c['buster_preserved']):>4} "
                 f"{str(c['tesla_2019_absent']):>5} {str(c['year_survives']):>5} "
                 f"{str(r['seed_ok']):>5}")
    L.append(f"{'median':>6} | {lat['solo0_completion_ms_median']:>10.1f} "
             f"{lat['batch_slot0_completion_ms_median']:>11.1f} {med_comp:>7.3f} "
             f"{med_ttft:>7.3f} | {med([r['profile_tokens'] for r in reps]):>4.0f} "
             f"{'':>11} {check_totals['tesla_present']}/{n:>4} "
             f"{check_totals['honda_absent']}/{n:>4} "
             f"{check_totals['seattle_preserved']}/{n:>4} "
             f"{check_totals['buster_preserved']}/{n:>3} "
             f"{check_totals['tesla_2019_absent']}/{n:>4} "
             f"{check_totals['year_survives']}/{n:>4} {seeds_ok}/{n:>4}")
    L.append("")
    L.append(f"Per-seed checks passing: tesla={check_totals['tesla_present']}/{n}  "
             f"honda/civic_absent={check_totals['honda_absent']}/{n}  "
             f"seattle={check_totals['seattle_preserved']}/{n}  "
             f"buster={check_totals['buster_preserved']}/{n}  "
             f"tokens<75={token_ok}/{n}  "
             f"2019-Tesla-absent={check_totals['tesla_2019_absent']}/{n}  "
             f"year-survives={check_totals['year_survives']}/{n}")
    L.append(f"Year lines (2019 anywhere in profile): {summary['year_lines']}")
    L.append(f"Profile tokens per rep: {summary['profile_tokens']}  "
             f"words: {summary['profile_words']}  finish={summary['slot1_finish']}")
    L.append(f"Slot 0 completion ratio batch/solo: median={med_comp:.3f} mean={mean_comp:.3f} "
             f"(solo med {lat['solo0_completion_ms_median']:.1f} ms, "
             f"batch med {lat['batch_slot0_completion_ms_median']:.1f} ms, "
             f"ratio of medians {lat['ratio_of_completion_medians']:.3f})")
    L.append(f"Slot 0 ttft ratio batch/solo:       median={med_ttft:.3f} mean={mean_ttft:.3f}")
    L.append(f"Slot 1 completion ratio batch/solo (standalone control): "
             f"median={lat['slot1_completion_ratio_median']:.3f} "
             f"per-rep={[round(x, 3) for x in s1_ratios]}")
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
