"""Phase 2 - Letta-style block compaction (RecycleLM).

k=2 on parallel::Runtime: Slot 0 = conversational reply (prompt byte-identical
to angle 27, reuses data/ingame_mem_solo.tsv), Slot 1 = compaction of a seeded
[User Profile] block against the same 3-turn dialogue (Honda -> Tesla turn is
the last user message). Slot 1's profile seed + compaction rule live in its
user text (data/ingame_compact_batch.tsv / data/ingame_compact_solo.tsv).

Per seed, same protocol as angle 27 (seeds 1234-1238, max_tokens=160,
temp=0.2, top_p=0.9):
  1. serial solo, Slot 0 prompt only        -> Slot 0 baseline
  2. serial solo, Slot 1 compaction prompt  -> standalone latency.
     Run at seed+1 so sampling matches the batch's row-1 branch (the CLI
     assigns --seed + row index per branch).
  3. batch k=2 (Slot 0 + Slot 1)            -> in-flight measurement

Acceptance (per seed): tesla present; honda/civic absent; Seattle preserved;
Buster preserved; profile tokens (CLI reported generated) < 75.
Overall: every seed passes AND Slot 0 completion ratio median <= 1.10x.

Writes results/ingame_letta_compact.json and results/ingame_letta_compact.txt.
Raw per-rep CLI JSON in results/ingame_letta_compact_raw/.
Run: python harness\\run_letta_compact.py [--reps N] [--rescore] [--dry-run]
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

BATCH_TSV = "data/ingame_compact_batch.tsv"
SOLO1_TSV = "data/ingame_compact_solo.tsv"
RAW_DIR = ROOT / "results" / "ingame_letta_compact_raw"
OUT_JSON = ROOT / "results" / "ingame_letta_compact.json"
OUT_TXT = ROOT / "results" / "ingame_letta_compact.txt"
S1_MARKER = "[User Profile]"
PROFILE_TOKEN_MAX = 75


def profile_checks(text):
    t = text.lower()
    return {
        "tesla_present": "tesla" in t,
        "honda_absent": "honda" not in t and "civic" not in t,
        "seattle_preserved": "seattle" in t,
        "buster_preserved": "buster" in t,
    }


def build_rep(seed, solo0, solo1, batch):
    s0_solo = get_stream(solo0, "Reply to the user")
    s0_batch = get_stream(batch, "Reply to the user")
    s1_solo = get_stream(solo1, S1_MARKER)
    s1_batch = get_stream(batch, S1_MARKER)

    checks = profile_checks(s1_batch["text"])
    tokens = s1_batch["generated"]
    return {
        "seed": seed,
        "solo_slot0": summarize_branch(s0_solo),
        "batch_slot0": summarize_branch(s0_batch),
        "solo_slot1": summarize_branch(s1_solo),
        "batch_slot1": summarize_branch(s1_batch),
        "s0_prompt_identical": s0_solo["prompt"] == s0_batch["prompt"],
        "checks": checks,
        "profile_tokens": tokens,
        "profile_words": len(s1_batch["text"].split()),
        "profile_token_ok": tokens < PROFILE_TOKEN_MAX,
        "seed_ok": all(checks.values()) and tokens < PROFILE_TOKEN_MAX,
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
        print("plan:", seeds, "solo", SOLO_TSV, "solo1", SOLO1_TSV, "batch", BATCH_TSV)
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
            print(f"[run] seed={seed} solo1 (serial, Slot 1 compaction, seed+1)", flush=True)
            solo1 = run_cli("serial", SOLO1_TSV, seed + 1, RAW_DIR / f"rep{seed}_solo1.json")
            print(f"[run] seed={seed} batch k=2 (compact)", flush=True)
            batch = run_cli("batch", BATCH_TSV, seed, RAW_DIR / f"rep{seed}_batch.json")
        rep = build_rep(seed, solo0, solo1, batch)
        reps.append(rep)
        r = rep["ratios"]
        c = rep["checks"]
        print(f"[rep {seed}] ok={rep['seed_ok']} tesla={c['tesla_present']} "
              f"honda_absent={c['honda_absent']} seattle={c['seattle_preserved']} "
              f"buster={c['buster_preserved']} tok={rep['profile_tokens']} "
              f"comp_ratio={r['completion']:.3f} s1_ratio={r['slot1_completion']:.3f} "
              f"finish={rep['batch_slot1']['finish']}", flush=True)

    n = len(reps)
    comp_ratios = [r["ratios"]["completion"] for r in reps]
    ttft_ratios = [r["ratios"]["ttft"] for r in reps]
    s1_ratios = [r["ratios"]["slot1_completion"] for r in reps]
    med_comp, mean_comp = med(comp_ratios), statistics.mean(comp_ratios)
    med_ttft, mean_ttft = med(ttft_ratios), statistics.mean(ttft_ratios)
    seeds_ok = sum(r["seed_ok"] for r in reps)
    check_totals = {k: sum(r["checks"][k] for r in reps)
                    for k in profile_checks("")}
    token_ok = sum(r["profile_token_ok"] for r in reps)

    criteria = {
        "per_seed_checks_all_pass": seeds_ok == n,
        "profile_tokens_lt_75_all_seeds": token_ok == n,
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
        "experiment": "Phase 2 Letta-style block compaction (k=2, Slot 0 chat + Slot 1 profile compaction)",
        "parent": "results/ingame_memory_test.json (angle 27)",
        "model": MODEL, "system": SYSTEM, "seeds": seeds,
        "max_tokens": MAX_TOKENS, "temperature": TEMP, "top_p": TOP_P,
        "n_reps": n,
        "data_files": {"solo_slot0": SOLO_TSV, "solo_slot1": SOLO1_TSV, "batch": BATCH_TSV,
                       "note": "s0 prompt byte-identical to angle 27 (data/ingame_mem_solo.tsv); "
                               "solo1 run at seed+1 to match batch row-1 sampling seed"},
        "profile_token_max": PROFILE_TOKEN_MAX,
        "acceptance": criteria,
        "seeds_passing_all_checks": seeds_ok,
        "check_totals": check_totals,
        "profile_tokens_ok_seeds": token_ok,
        "profile_tokens": [r["profile_tokens"] for r in reps],
        "profile_words": [r["profile_words"] for r in reps],
        "slot1_finish": [r["batch_slot1"]["finish"] for r in reps],
        "latency": lat,
        "s0_prompt_identical_all_reps": all(r["s0_prompt_identical"] for r in reps),
        "verdict": verdict,
        "reps": reps,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    L = []
    L.append("Phase 2 - Letta-style block compaction: k=2 parallel::Runtime")
    L.append(f"model={MODEL}  max_tokens={MAX_TOKENS} temp={TEMP} top_p={TOP_P} system={SYSTEM!r}")
    L.append(f"reps={n} seeds={seeds}  solo0=serial({SOLO_TSV})  "
             f"solo1=serial({SOLO1_TSV}, seed+1)  batch=k=2({BATCH_TSV})")
    L.append("Slot 1 input = same dialogue + seeded [User Profile] + compaction rule (in user text)")
    L.append("")
    L.append(f"{'seed':>6} | {'s0_solo_ms':>10} {'s0_batch_ms':>11} {'comp_r':>7} "
             f"{'ttft_r':>7} | {'s1_solo_ms':>10} {'s1_batch_ms':>11} {'s1_r':>6} "
             f"{'tok':>4} {'fin':>11} {'tesla':>5} {'hondX':>5} {'sea':>5} {'bst':>4} {'ok':>5}")
    for r in reps:
        c = r["checks"]
        L.append(f"{r['seed']:>6} | {r['solo_slot0']['completion_ms']:>10.1f} "
                 f"{r['batch_slot0']['completion_ms']:>11.1f} "
                 f"{r['ratios']['completion']:>7.3f} {r['ratios']['ttft']:>7.3f} | "
                 f"{r['solo_slot1']['completion_ms']:>10.1f} "
                 f"{r['batch_slot1']['completion_ms']:>11.1f} "
                 f"{r['ratios']['slot1_completion']:>6.3f} "
                 f"{r['profile_tokens']:>4} {r['batch_slot1']['finish']:>11} "
                 f"{str(c['tesla_present']):>5} {str(c['honda_absent']):>5} "
                 f"{str(c['seattle_preserved']):>5} {str(c['buster_preserved']):>4} "
                 f"{str(r['seed_ok']):>5}")
    L.append(f"{'median':>6} | {lat['solo0_completion_ms_median']:>10.1f} "
             f"{lat['batch_slot0_completion_ms_median']:>11.1f} {med_comp:>7.3f} "
             f"{med_ttft:>7.3f} | {lat['solo1_completion_ms_median']:>10.1f} "
             f"{lat['batch_slot1_completion_ms_median']:>11.1f} "
             f"{lat['slot1_completion_ratio_median']:>6.3f} "
             f"{med([r['profile_tokens'] for r in reps]):>4.0f} {'':>11} "
             f"{check_totals['tesla_present']}/{n:>5} {check_totals['honda_absent']}/{n:>5} "
             f"{check_totals['seattle_preserved']}/{n:>5} {check_totals['buster_preserved']}/{n:>4} "
             f"{seeds_ok}/{n:>5}")
    L.append("")
    L.append("Per-seed checks passing: tesla="
             f"{check_totals['tesla_present']}/{n}  honda/civic_absent="
             f"{check_totals['honda_absent']}/{n}  seattle="
             f"{check_totals['seattle_preserved']}/{n}  buster="
             f"{check_totals['buster_preserved']}/{n}  tokens<75={token_ok}/{n}")
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
