"""Angle 36 control arm - 32b attribution probe (no anti-bleed sentence), 5 seeds.

Angle 32b (run_ingame_2024_gate.py) paired the angle-31 anti-bleed sentence
with a dialogue whose final user message states the new vehicle's own year
("I traded the 2019 Honda for a 2024 Tesla last week, so I need charging
stations along my commute."): "2024" was present 0/5 - false suppression.
MISSING CONTROL: the angle-29 prompt (same compaction rule WITHOUT the
anti-bleed sentence) + the same 2024 dialogue. This run is that control
arm - it attributes the suppression to the sentence vs the dialogue/prompt.

Prompt derivation (verified at generation time, diff recorded in the JSON):
  source = data/ingame_compact_batch.tsv (angle-29: NO anti-bleed sentence)
  derived = source with ONLY the final user message swapped
            (OLD_FINAL -> NEW_FINAL, exactly 2 occurrences: s0 and s1 rows,
             byte-substring replacement, reverse-asserted)
  s0 row  = angle-29 s0 with only the final message swapped (identical
            dialogue in both arms, as angle 32b did);
  s1 row  = angle-29 s1 with only the final message swapped;
  solo arm = the derived s0 row only (serial, Slot 0) - byte-identical to
            the batch s0 row (angle-32b arm design, replicated exactly).

Verified relations (all asserted):
  - anti-bleed sentence ABSENT from the derived s1;
  - derived s1 == angle-32b's s1 with exactly the one anti-bleed sentence
    removed (the attribution diff);
  - derived s0 == angle-32b's s0 byte-identical (angle-31 and angle-29
    shared the same s0 dialogue, so the swap lands on identical bytes);
  - angle-29 s1 == angle-29 source + sentence reconstruction holds for the
    angle-31 file (cross-check recorded).

Protocol identical to angle 32b: seeds 1234-1238, temp 0.2, top_p 0.9,
max_tokens=160, batch k=2 + fresh solo (serial, Slot 0 only), same exe.

Gates/metrics identical to angle 32b (code-reused: gate_checks and
build_rep imported from run_ingame_2024_gate): "2024" present, "2019
Tesla" absent, honda/civic absent, seattle/buster preserved, profile
tokens <75, Slot 0 completion median <=1.10, Slot 0 text identity.

Decision logic (reported, not a gate):
  2024 present >=4/5 -> the anti-bleed sentence IS the cause of the
      suppression (control confirms);
  2024 present 0/5   -> suppression persists without the sentence
      (sentence exonerated; dialogue/prompt-driven);
  1-3/5              -> inconclusive middle (reported as such).

Writes results/ingame_2024_control.{json,txt}.
Raw per-rep CLI JSON in results/ingame_2024_control_raw/; derived prompts
in data/ingame_2024_control_{batch,solo}.tsv.
Run: python harness\\run_ingame_2024_control.py [--reps N] [--rescore] [--dry-run]
"""
import argparse
import difflib
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ingame_common import (  # noqa: E402
    MAX_TOKENS, MODEL, ROOT, SEEDS, PROFILE_TOKEN_MAX, SYSTEM,
    TEMP, TOP_P, cli_provenance, load_raw, med, run_cli, write_tsv,
)
import run_ingame_2024_gate as a32b  # noqa: E402

SRC = "data/ingame_compact_batch.tsv"                 # angle-29 (no sentence)
A32B_SRC = "data/ingame_2024_batch.tsv"                # angle-32b (sentence)
BATCH_TSV = "data/ingame_2024_control_batch.tsv"
SOLO_TSV = "data/ingame_2024_control_solo.tsv"
RAW_DIR = ROOT / "results" / "ingame_2024_control_raw"
OUT_JSON = ROOT / "results" / "ingame_2024_control.json"
OUT_TXT = ROOT / "results" / "ingame_2024_control.txt"

OLD_FINAL = a32b.OLD_FINAL
NEW_FINAL = a32b.NEW_FINAL
SENTENCE = ("Do not carry over adjectives, years, or specs from replaced "
            "items unless explicitly stated. ")


def load_rows(path):
    rows = []
    for line in (ROOT / path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rid, prompt = line.split("\t", 1)
        rows.append((rid, prompt.replace("\\n", "\n").replace("\\t", "\t")))
    return rows


def derive_prompts():
    """Angle-29 prompts with ONLY the final user message swapped."""
    src = (ROOT / SRC).read_text(encoding="utf-8")
    if src.count(OLD_FINAL) != 2:
        raise SystemExit(f"expected 2 occurrences of OLD_FINAL in {SRC}, "
                         f"found {src.count(OLD_FINAL)}")
    derived = src.replace(OLD_FINAL, NEW_FINAL)
    # reverse substitution must recover the source exactly
    if derived.replace(NEW_FINAL, OLD_FINAL) != src:
        raise SystemExit("reverse substitution mismatch - swap not clean")
    rows = []
    for line in derived.splitlines():
        if not line.strip():
            continue
        rid, prompt = line.split("\t", 1)
        rows.append((rid, prompt.replace("\\n", "\n").replace("\\t", "\t")))
    batch_rows = rows
    solo_rows = [r for r in rows if r[0] == "s0"]
    d = dict(rows)
    s0, s1 = d["s0"], d["s1"]

    checks = {
        "old_final_count_in_source_is_2": True,
        "reverse_substitution_recovers_source": True,
        "anti_bleed_sentence_absent_in_s1": SENTENCE.strip() not in s1,
        "profile_seed_present_in_s1": "- Vehicle: 2019 Honda Civic" in s1,
        "new_final_in_s0": NEW_FINAL in s0,
        "new_final_in_s1": NEW_FINAL in s1,
        "old_final_absent_both": (OLD_FINAL not in s0 and OLD_FINAL not in s1),
        "solo_is_s0_only": [r[0] for r in solo_rows] == ["s0"],
    }

    # diffs for the record (decoded prompts, unified diff of swapped region)
    diffs = {}
    for name, path in (("angle29", SRC), ("angle32b", A32B_SRC)):
        src_rows = dict(load_rows(path))
        for rid, new in (("s0", s0), ("s1", s1)):
            old = src_rows[rid]
            ud = list(difflib.unified_diff(
                old.splitlines(), new.splitlines(),
                fromfile=f"{name}:{rid}:src", tofile=f"control:{rid}",
                lineterm=""))
            diffs[f"control_{rid}_vs_{name}_{rid}"] = ud

    # attribution cross-check vs angle 32b
    b32 = dict(load_rows(A32B_SRC))
    checks["control_s1_equals_32b_s1_minus_sentence"] = (
        b32["s1"].replace(SENTENCE, "") == s1)
    checks["control_s0_equals_32b_s0"] = b32["s0"] == s0
    checks["angle29_s1_plus_sentence_equals_angle31_s1"] = (
        dict(load_rows("data/ingame_bleed_batch.tsv"))["s1"]
        == dict(load_rows(SRC))["s1"].replace(
            "obsolete lines. ", "obsolete lines. " + SENTENCE))

    if not all(checks.values()):
        raise SystemExit(f"derivation verification FAILED: {checks}")
    return batch_rows, solo_rows, checks, diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rescore", action="store_true",
                    help="re-analyze saved raw CLI JSON without re-running the GPU")
    args = ap.parse_args()
    seeds = SEEDS[: args.reps]

    batch_rows, solo_rows, checks, diffs = derive_prompts()
    if args.dry_run:
        print("plan:", seeds, "derive", SRC, "->", BATCH_TSV, SOLO_TSV,
              "(final message swapped only; no anti-bleed sentence)")
        print("verification:", json.dumps(checks, indent=1))
        return 0

    write_tsv(ROOT / BATCH_TSV, batch_rows)
    write_tsv(ROOT / SOLO_TSV, solo_rows)
    print(f"[derive] wrote {BATCH_TSV} ({len(batch_rows)} rows) and "
          f"{SOLO_TSV} ({len(solo_rows)} rows); angle-29 prompt + 2024 "
          f"dialogue; only the final user message differs vs {SRC}",
          flush=True)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    reps = []
    t_gpu0 = time.monotonic()
    for seed in seeds:
        if args.rescore:
            solo0 = load_raw(RAW_DIR / f"rep{seed}_solo.json")
            batch = load_raw(RAW_DIR / f"rep{seed}_batch.json")
        else:
            print(f"[run] seed={seed} solo (serial, Slot 0, new dialogue)", flush=True)
            solo0 = run_cli("serial", SOLO_TSV, seed, RAW_DIR / f"rep{seed}_solo.json")
            print(f"[run] seed={seed} batch k=2 (2024 control)", flush=True)
            batch = run_cli("batch", BATCH_TSV, seed, RAW_DIR / f"rep{seed}_batch.json")
        rep = a32b.build_rep(seed, solo0, batch)
        reps.append(rep)
        r = rep["ratios"]
        c = rep["checks"]
        print(f"[rep {seed}] ok={rep['seed_ok']} 2024={c['y2024_present']} "
              f"2024_lines={c['y2024_lines']} 2019tesla_absent={c['tesla_2019_absent']} "
              f"honda_absent={c['honda_absent']} seattle={c['seattle_preserved']} "
              f"buster={c['buster_preserved']} stray2019={c['stray_2019_anywhere']} "
              f"tok={rep['profile_tokens']} comp_ratio={r['completion']:.3f} "
              f"s0_identical={rep['s0_identical']} "
              f"finish={rep['batch_slot1']['finish']}", flush=True)

    gpu_wall_s = time.monotonic() - t_gpu0
    print(f"[gpu] CLI wall time {gpu_wall_s:.1f} s", flush=True)

    n = len(reps)
    comp_ratios = [r["ratios"]["completion"] for r in reps]
    ttft_ratios = [r["ratios"]["ttft"] for r in reps]
    med_comp, mean_comp = med(comp_ratios), statistics.mean(comp_ratios)
    med_ttft, mean_ttft = med(ttft_ratios), statistics.mean(ttft_ratios)
    check_totals = {k: sum(r["checks"][k] for r in reps)
                    for k in ("y2024_present", "tesla_present",
                              "tesla_2019_absent", "honda_absent",
                              "seattle_preserved", "buster_preserved",
                              "stray_2019_anywhere",
                              "y2024_on_vehicle_or_tesla_line")}
    seeds_ok = sum(r["seed_ok"] for r in reps)
    core_ok = sum(r["core_checks_ok"] for r in reps)
    token_ok = sum(r["profile_token_ok"] for r in reps)
    ident_ok = sum(r["s0_identical"] for r in reps)

    criteria = {
        "y2024_present_5_of_5": check_totals["y2024_present"] == n,
        "tesla_2019_absent_5_of_5": check_totals["tesla_2019_absent"] == n,
        "honda_absent_5_of_5": check_totals["honda_absent"] == n,
        "seattle_buster_preserved_5_of_5": (
            check_totals["seattle_preserved"] == n
            and check_totals["buster_preserved"] == n),
        "profile_tokens_lt_75_all": token_ok == n,
        "slot0_completion_median_le_1_10": med_comp <= 1.10,
    }
    verdict = "meets" if all(criteria.values()) else "miss"

    present = check_totals["y2024_present"]
    if present >= 4:
        decision = ("sentence_is_cause: 2024 present "
                    f"{present}/{n} without the anti-bleed sentence -> the "
                    "sentence IS the cause of the suppression (control "
                    "confirms)")
    elif present == 0:
        decision = ("sentence_exonerated: 2024 present 0/5 without the "
                    "sentence -> suppression persists with the sentence "
                    "removed (sentence exonerated; dialogue/prompt-driven)")
    else:
        decision = (f"inconclusive: 2024 present {present}/{n} - neither "
                    "the >=4/5 confirm nor the 0/5 exoneration threshold "
                    "reached")

    lat = {
        "completion_ratio_median": med_comp, "completion_ratio_mean": mean_comp,
        "ttft_ratio_median": med_ttft, "ttft_ratio_mean": mean_ttft,
        "solo0_completion_ms_median": med([r["solo_slot0"]["completion_ms"] for r in reps]),
        "batch_slot0_completion_ms_median": med([r["batch_slot0"]["completion_ms"] for r in reps]),
        "ratio_of_completion_medians": (
            med([r["batch_slot0"]["completion_ms"] for r in reps])
            / med([r["solo_slot0"]["completion_ms"] for r in reps])),
        "per_rep_completion_ratios": comp_ratios,
        "per_rep_ttft_ratios": ttft_ratios,
    }

    summary = {
        "experiment": ("Angle 36 control arm for 32b attribution (k=2, "
                       "angle-29 compaction WITHOUT the anti-bleed sentence "
                       "+ angle-32b's 2024 dialogue)"),
        "parent": "results/ingame_2024_gate.json (angle 32b)",
        "model": MODEL, "system": SYSTEM, "seeds": seeds,
        "max_tokens": MAX_TOKENS, "temperature": TEMP, "top_p": TOP_P,
        "n_reps": n,
        "metadata": cli_provenance(),
        "data_files": {"solo_slot0": SOLO_TSV, "batch": BATCH_TSV,
                       "note": ("derived from data/ingame_compact_batch.tsv "
                                "by replacing ONLY the final user message "
                                "(byte-substring, verified at generation "
                                "time); solo TSV is the derived s0 row only "
                                "(angle-32b arm design)")},
        "derivation_verification": checks,
        "unified_diffs": diffs,
        "final_user_message": NEW_FINAL,
        "profile_token_max": PROFILE_TOKEN_MAX,
        "acceptance": criteria,
        "seeds_passing_all_checks": seeds_ok,
        "core_checks_passing": core_ok,
        "check_totals": check_totals,
        "profile_tokens_ok_seeds": token_ok,
        "profile_tokens": [r["profile_tokens"] for r in reps],
        "profile_words": [r["profile_words"] for r in reps],
        "y2024_lines": [r["checks"]["y2024_lines"] for r in reps],
        "y2019_lines": [r["checks"]["y2019_lines"] for r in reps],
        "slot0_text_identical_seeds": ident_ok,
        "slot1_finish": [r["batch_slot1"]["finish"] for r in reps],
        "latency": lat,
        "s0_prompt_identical_all_reps": all(r["s0_prompt_identical"] for r in reps),
        "decision": decision,
        "decision_rule": ("2024 present >=4/5 -> sentence IS the cause "
                          "(control confirms); 0/5 -> suppression persists "
                          "without the sentence (sentence exonerated; "
                          "dialogue/prompt-driven); 1-3/5 inconclusive"),
        "gpu_wall_s": round(gpu_wall_s, 1),
        "verdict": verdict,
        "reps": reps,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                        encoding="utf-8")

    L = []
    L.append("Angle 36 control arm (32b attribution, NO anti-bleed sentence): "
             "k=2 parallel::Runtime")
    L.append(f"model={MODEL}  max_tokens={MAX_TOKENS} temp={TEMP} top_p={TOP_P} "
             f"system={SYSTEM!r}")
    L.append(f"cli_sha256={summary['metadata']['cli_sha256']}  "
             f"built={summary['metadata']['cli_built_mtime']}  "
             f"tree={summary['metadata']['git_tree']}")
    L.append(f"reps={n} seeds={seeds}  solo0=serial({SOLO_TSV})  "
             f"batch=k=2({BATCH_TSV})  GPU wall {gpu_wall_s:.1f} s")
    L.append("Slot 1 = angle-29 compaction (no anti-bleed sentence); final "
             "user message = angle-32b's 2024 Tesla line")
    L.append("derivation verified: " + ", ".join(
        f"{k}={'OK' if v else 'FAIL'}" for k, v in checks.items()))
    L.append("")
    L.append(f"{'seed':>6} | {'s0_solo_ms':>10} {'s0_batch_ms':>11} {'comp_r':>7} "
             f"{'ttft_r':>7} {'s0same':>6} | {'tok':>4} {'fin':>11} {'2024':>4} "
             f"{'2024ln':>6} {'no19T':>5} {'hondX':>5} {'sea':>5} {'bst':>4} "
             f"{'yrAny':>5} {'ok':>5}")
    for r in reps:
        c = r["checks"]
        L.append(f"{r['seed']:>6} | {r['solo_slot0']['completion_ms']:>10.1f} "
                 f"{r['batch_slot0']['completion_ms']:>11.1f} "
                 f"{r['ratios']['completion']:>7.3f} {r['ratios']['ttft']:>7.3f} "
                 f"{str(r['s0_identical']):>6} | {r['profile_tokens']:>4} "
                 f"{r['batch_slot1']['finish']:>11} {str(c['y2024_present']):>4} "
                 f"{str(c['y2024_on_vehicle_or_tesla_line']):>6} "
                 f"{str(c['tesla_2019_absent']):>5} {str(c['honda_absent']):>5} "
                 f"{str(c['seattle_preserved']):>5} {str(c['buster_preserved']):>4} "
                 f"{str(c['stray_2019_anywhere']):>5} {str(r['seed_ok']):>5}")
    L.append(f"{'median':>6} | {lat['solo0_completion_ms_median']:>10.1f} "
             f"{lat['batch_slot0_completion_ms_median']:>11.1f} {med_comp:>7.3f} "
             f"{med_ttft:>7.3f} | {med([r['profile_tokens'] for r in reps]):>4.0f}")
    L.append("")
    L.append("Per-seed checks: 2024-present="
             f"{check_totals['y2024_present']}/{n}  "
             f"2024-on-vehicle-line={check_totals['y2024_on_vehicle_or_tesla_line']}/{n}  "
             f"2019-Tesla-absent={check_totals['tesla_2019_absent']}/{n}  "
             f"honda/civic-absent={check_totals['honda_absent']}/{n}  "
             f"seattle={check_totals['seattle_preserved']}/{n}  "
             f"buster={check_totals['buster_preserved']}/{n}  "
             f"stray-2019-anywhere={check_totals['stray_2019_anywhere']}/{n}  "
             f"tokens<75={token_ok}/{n}")
    L.append(f"2024 lines per rep: {summary['y2024_lines']}")
    L.append(f"2019 lines per rep: {summary['y2019_lines']}")
    L.append(f"Profile tokens per rep: {summary['profile_tokens']}  "
             f"words: {summary['profile_words']}  finish={summary['slot1_finish']}")
    L.append(f"Slot 0 completion ratio batch/solo: median={med_comp:.3f} "
             f"mean={mean_comp:.3f} (solo med "
             f"{lat['solo0_completion_ms_median']:.1f} ms, batch med "
             f"{lat['batch_slot0_completion_ms_median']:.1f} ms, ratio of "
             f"medians {lat['ratio_of_completion_medians']:.3f})")
    L.append(f"Slot 0 ttft ratio batch/solo:       median={med_ttft:.3f} "
             f"mean={mean_ttft:.3f}")
    L.append(f"Slot 0 text identical solo vs batch (same seed): "
             f"{ident_ok}/{n} reps")
    L.append("ACCEPTANCE: " + ", ".join(f"{k}={'PASS' if v else 'FAIL'}"
                                        for k, v in criteria.items()))
    L.append(f"VERDICT: {verdict}")
    L.append(f"DECISION: {decision}")
    text = "\n".join(L) + "\n"
    OUT_TXT.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
