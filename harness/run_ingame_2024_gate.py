"""Angle 32b - false-suppression boundary (RecycleLM, follow-up to 31).

Angle 31's anti-bleed sentence ("Do not carry over adjectives, years, or
specs from replaced items unless explicitly stated") fixed the "2019 Tesla"
leak, but the false-suppression direction was untested: what if the NEW item
carries its own legitimately stated year? This run swaps the dialogue's final
user message for one that states the new vehicle's year explicitly:

  "I traded the 2019 Honda for a 2024 Tesla last week, so I need charging
   stations along my commute."

Everything else is the angle-31 prompt, derived at generation time by
byte-substring replacement in data/ingame_bleed_{batch,solo}.tsv (verified:
exactly one replacement per prompt; the solo TSV here is Slot 0 only with
the same new dialogue). Seed profile is unchanged (Vehicle: 2019 Honda
Civic), protocol unchanged: seeds 1234-1238, temp 0.2, max_tokens=160,
fresh solo arm (serial, Slot 0 only) + batch k=2.

Gates: "2024" present 5/5 (false-suppression does NOT occur); "2019 Tesla"
absent 5/5 (bleed still fixed); honda/civic absent 5/5; seattle/buster
preserved 5/5; profile tokens <75; Slot 0 completion median <=1.10x.
Also recorded: where "2024" lands (vehicle line vs elsewhere) and any
stray "2019" anywhere in the profile.

Harness fixes vs angle 31: Slot 0 text-identity + generated-token columns
(reviewer M1); parallel_cli.exe SHA256 + dirty-tree note (M2); run_cli
decodes UTF-8 (M3).

Writes results/ingame_2024_gate.json and results/ingame_2024_gate.txt.
Raw per-rep CLI JSON in results/ingame_2024_gate_raw/.
Run: python harness\\run_ingame_2024_gate.py [--reps N] [--rescore] [--dry-run]
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ingame_common import (  # noqa: E402
    MAX_TOKENS, MODEL, ROOT, SEEDS, S0_MARKER, S1_PROFILE_MARKER, SYSTEM,
    TEMP, TOP_P, PROFILE_TOKEN_MAX, cli_provenance, get_stream, load_raw, med,
    run_cli, s0_identity, summarize_branch, write_tsv,
)

SRC_BATCH = "data/ingame_bleed_batch.tsv"
BATCH_TSV = "data/ingame_2024_batch.tsv"
SOLO_TSV = "data/ingame_2024_solo.tsv"
RAW_DIR = ROOT / "results" / "ingame_2024_gate_raw"
OUT_JSON = ROOT / "results" / "ingame_2024_gate.json"
OUT_TXT = ROOT / "results" / "ingame_2024_gate.txt"

OLD_FINAL = ("Also, I sold my Honda last week and got a Tesla, so I'll need "
             "charging stations along my commute instead of gas stations.")
NEW_FINAL = ("I traded the 2019 Honda for a 2024 Tesla last week, so I need "
             "charging stations along my commute.")


def derive_prompts():
    """Angle-31 prompts with ONLY the final user message swapped."""
    src = (ROOT / SRC_BATCH).read_text(encoding="utf-8")
    if src.count(OLD_FINAL) != 2:
        raise SystemExit(f"expected 2 occurrences of OLD_FINAL in {SRC_BATCH}, "
                         f"found {src.count(OLD_FINAL)}")
    derived = src.replace(OLD_FINAL, NEW_FINAL)
    rows = []
    for line in derived.splitlines():
        if not line.strip():
            continue
        rid, prompt = line.split("\t", 1)
        rows.append((rid, prompt.replace("\\n", "\n").replace("\\t", "\t")))
    batch_rows = rows
    solo_rows = [r for r in rows if r[0] == "s0"]
    # sanity: derived s1 must still carry the anti-bleed sentence + profile
    s1 = dict(rows)["s1"]
    assert "Do not carry over adjectives, years, or specs" in s1
    assert "- Vehicle: 2019 Honda Civic" in s1
    assert NEW_FINAL in s1 and OLD_FINAL not in s1
    assert NEW_FINAL in dict(rows)["s0"]
    return batch_rows, solo_rows


def gate_checks(text):
    t = text.lower()
    lines_2024 = [ln.strip() for ln in text.splitlines() if "2024" in ln]
    lines_2019 = [ln.strip() for ln in text.splitlines() if "2019" in ln]
    y2024_on_entity = any(("tesla" in ln.lower() or "vehicle" in ln.lower())
                          for ln in lines_2024)
    return {
        "y2024_present": "2024" in t,
        "tesla_present": "tesla" in t,
        "y2024_lines": lines_2024,
        "y2024_on_vehicle_or_tesla_line": y2024_on_entity,
        "tesla_2019_absent": "2019 tesla" not in t,
        "honda_absent": "honda" not in t and "civic" not in t,
        "seattle_preserved": "seattle" in t,
        "buster_preserved": "buster" in t,
        "stray_2019_anywhere": "2019" in t,
        "y2019_lines": lines_2019,
    }


def build_rep(seed, solo0, batch):
    s0_solo = get_stream(solo0, S0_MARKER)
    s0_batch = get_stream(batch, S0_MARKER)
    s1_batch = get_stream(batch, S1_PROFILE_MARKER)

    checks = gate_checks(s1_batch["text"])
    tokens = s1_batch["generated"]
    core_keys = ("tesla_present", "honda_absent", "seattle_preserved",
                 "buster_preserved")
    core_ok = all(checks[k] for k in core_keys)
    return {
        "seed": seed,
        "solo_slot0": summarize_branch(s0_solo),
        "batch_slot0": summarize_branch(s0_batch),
        "batch_slot1": summarize_branch(s1_batch),
        **s0_identity(s0_solo["text"], s0_batch["text"],
                      s0_solo["generated"], s0_batch["generated"]),
        "s0_prompt_identical": s0_solo["prompt"] == s0_batch["prompt"],
        "checks": checks,
        "core_checks_ok": core_ok,
        "profile_tokens": tokens,
        "profile_words": len(s1_batch["text"].split()),
        "profile_token_ok": tokens < PROFILE_TOKEN_MAX,
        "seed_ok": (core_ok and tokens < PROFILE_TOKEN_MAX
                    and checks["y2024_present"] and checks["tesla_2019_absent"]),
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

    batch_rows, solo_rows = derive_prompts()
    if args.dry_run:
        print("plan:", seeds, "derive", SRC_BATCH, "->", BATCH_TSV, SOLO_TSV,
              f"(replacement {OLD_FINAL[:40]!r}... -> {NEW_FINAL[:40]!r}...)")
        return 0

    write_tsv(ROOT / BATCH_TSV, batch_rows)
    write_tsv(ROOT / SOLO_TSV, solo_rows)
    print(f"[derive] wrote {BATCH_TSV} ({len(batch_rows)} rows) and "
          f"{SOLO_TSV} ({len(solo_rows)} rows); s1 keeps anti-bleed sentence "
          f"+ angle-29 profile seed; only final user message differs vs "
          f"{SRC_BATCH}", flush=True)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    reps = []
    for seed in seeds:
        if args.rescore:
            solo0 = load_raw(RAW_DIR / f"rep{seed}_solo.json")
            batch = load_raw(RAW_DIR / f"rep{seed}_batch.json")
        else:
            print(f"[run] seed={seed} solo (serial, Slot 0, new dialogue)", flush=True)
            solo0 = run_cli("serial", SOLO_TSV, seed, RAW_DIR / f"rep{seed}_solo.json")
            print(f"[run] seed={seed} batch k=2 (2024 gate)", flush=True)
            batch = run_cli("batch", BATCH_TSV, seed, RAW_DIR / f"rep{seed}_batch.json")
        rep = build_rep(seed, solo0, batch)
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
        "experiment": ("Angle 32b false-suppression gate (k=2, angle-31 "
                       "compaction + dialogue stating the new item's own year)"),
        "parent": "results/ingame_bleed_gate.json (angle 31)",
        "model": MODEL, "system": SYSTEM, "seeds": seeds,
        "max_tokens": MAX_TOKENS, "temperature": TEMP, "top_p": TOP_P,
        "n_reps": n,
        "metadata": cli_provenance(),
        "data_files": {"solo_slot0": SOLO_TSV, "batch": BATCH_TSV,
                       "note": ("derived from data/ingame_bleed_batch.tsv by "
                                "replacing ONLY the final user message "
                                "(byte-substring, verified at generation time); "
                                "solo TSV is the derived s0 row only")},
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
        "verdict": verdict,
        "reps": reps,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                        encoding="utf-8")

    L = []
    L.append("Angle 32b - false-suppression boundary: k=2 parallel::Runtime")
    L.append(f"model={MODEL}  max_tokens={MAX_TOKENS} temp={TEMP} top_p={TOP_P} "
             f"system={SYSTEM!r}")
    L.append(f"cli_sha256={summary['metadata']['cli_sha256']}  "
             f"built={summary['metadata']['cli_built_mtime']}  "
             f"tree={summary['metadata']['git_tree']}")
    L.append(f"reps={n} seeds={seeds}  solo0=serial({SOLO_TSV})  batch=k=2({BATCH_TSV})")
    L.append("Slot 1 = angle-31 anti-bleed compaction; final user message states "
             "the new item's own year (2024 Tesla)")
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
    text = "\n".join(L) + "\n"
    OUT_TXT.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
