"""Angle 32c - multi-turn compaction stability, 10 turns (RecycleLM, 31->32).

Angle 31's anti-bleed fix was validated on a single 3-turn dialogue. This
run stress-tests the same Slot 1 compaction rule (anti-bleed sentence
included) across a scripted 10-turn conversation with introduce / alter /
untouched structure, feeding profile_t back into turn t+1.

Scripted conversation (builder-final wording; user turns only — no assistant
feedback is fed back, which keeps Slot 0 prompts independent of prior Slot 0
batch-divergence and isolates compaction stability):

  seed profile: - Location: Seattle / - Pet: Buster (golden retriever, knee
  issues) / - Vehicle: 2019 Honda Civic / - Commute: 25 minutes highway

  1 introduce : I started learning piano last month, and I practice at home
                in the evenings.
  2 alter     : I moved to Portland last week, so my new place is there.
  3 untouched : The weather has been really nice this week.
  4 introduce : I also adopted a cat named Miso, so now I have two pets.
  5 alter     : I traded the 2019 Honda for a 2024 Tesla last week, so I
                need charging stations along my commute.
  6 untouched : How long is my commute again?
  7 introduce : I'm fostering a dog named Ranger for the next few weeks.
  8 introduce : I started a new job at Nimbus downtown.
  9 alter     : I moved back to Seattle last month, so Portland is behind me.
  10 untouched: Thanks for keeping track of all that for me.

Driver loop per turn t=1..10: Slot 1 prompt = angle-31 compaction rule
(with anti-bleed sentence) + profile_{t-1} + dialogue turns 1..t; Slot 0
prompt = conversational reply instruction + dialogue 1..t. Batch k=2 at
seed 3000+t AND a fresh serial Slot 0 run (same prompt, same seed) for the
latency baseline. profile_t = Slot 1 batch output; fed to t+1.

Hand-labeled expected profile state per turn (also in the JSON):
  buster 1-10; piano 1-10; commute line 1-10 (attrition probe); Miso 4-10;
  Ranger 7-10; Nimbus 8-10; Seattle 1 and 9-10 (Portland 2-8); vehicle
  2019 Honda 1-4, 2024 Tesla 5-10.

Per-turn gates: all currently-valid concepts present (no attrition);
superseded concepts absent (no resurrection — Seattle must NOT reappear at
turns 2-8); profile tokens <=75. Overall: Slot 0 completion ratio median
(batch/solo across turns) <=1.10.

Line-diff profile_{t-1} vs profile_t (normalized line sets; unmatched old
= INVALIDATE, unmatched new = ADD) vs the hand-labeled transition list;
event precision/recall reported descriptively.

Harness fixes vs angles 28-31: Slot 0 identity + generated-token columns
(reviewer M1); parallel_cli.exe SHA256 + dirty-tree note (M2); UTF-8 run_cli
(M3).

Writes results/ingame_multiturn.json and results/ingame_multiturn.txt.
Raw per-turn CLI JSON + prompt TSVs in results/ingame_multiturn_raw/.
Run: python harness\\run_ingame_multiturn.py [--turns N] [--rescore] [--dry-run]
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ingame_common import (  # noqa: E402
    COMPACT_RULE, MAX_TOKENS, MODEL, PROFILE_TOKEN_MAX, ROOT, S0_INSTRUCTION,
    S0_MARKER, S1_PROFILE_MARKER, SEED_PROFILE, SYSTEM, TEMP, TOP_P,
    cli_provenance, get_stream, load_raw, med, run_cli, s0_identity,
    summarize_branch, write_tsv,
)

RAW_DIR = ROOT / "results" / "ingame_multiturn_raw"
OUT_JSON = ROOT / "results" / "ingame_multiturn.json"
OUT_TXT = ROOT / "results" / "ingame_multiturn.txt"
SEED_BASE = 3000

TURNS = [
    (1, "introduce", "I started learning piano last month, and I practice at home in the evenings."),
    (2, "alter", "I moved to Portland last week, so my new place is there."),
    (3, "untouched", "The weather has been really nice this week."),
    (4, "introduce", "I also adopted a cat named Miso, so now I have two pets."),
    (5, "alter", "I traded the 2019 Honda for a 2024 Tesla last week, so I need charging stations along my commute."),
    (6, "untouched", "How long is my commute again?"),
    (7, "introduce", "I'm fostering a dog named Ranger for the next few weeks."),
    (8, "introduce", "I started a new job at Nimbus downtown."),
    (9, "alter", "I moved back to Seattle last month, so Portland is behind me."),
    (10, "untouched", "Thanks for keeping track of all that for me."),
]

# Hand label: concept -> keywords (case-insensitive substring, any-of).
CONCEPTS = {
    "location_seattle": ("seattle",),
    "location_portland": ("portland",),
    "pet_buster": ("buster",),
    "pet_miso": ("miso",),
    "pet_ranger_foster": ("ranger",),
    "vehicle_2019_honda": ("honda", "civic"),
    "vehicle_2024_tesla": ("tesla",),
    "job_nimbus": ("nimbus",),
    "hobby_piano": ("piano",),
    "commute_25min": ("commute",),
}
_ALL10 = set(range(1, 11))
EXPECTED = {
    "location_seattle": {1, 9, 10},
    "location_portland": {2, 3, 4, 5, 6, 7, 8},
    "pet_buster": _ALL10,
    "pet_miso": {4, 5, 6, 7, 8, 9, 10},
    "pet_ranger_foster": {7, 8, 9, 10},
    "vehicle_2019_honda": {1, 2, 3, 4},
    "vehicle_2024_tesla": {5, 6, 7, 8, 9, 10},
    "job_nimbus": {8, 9, 10},
    "hobby_piano": _ALL10,
    "commute_25min": _ALL10,
}
EXPECTED_EVENTS = {
    1: {"add": ["hobby_piano"], "invalidate": []},
    2: {"add": ["location_portland"], "invalidate": ["location_seattle"]},
    3: {"add": [], "invalidate": []},
    4: {"add": ["pet_miso"], "invalidate": []},
    5: {"add": ["vehicle_2024_tesla"], "invalidate": ["vehicle_2019_honda"]},
    6: {"add": [], "invalidate": []},
    7: {"add": ["pet_ranger_foster"], "invalidate": []},
    8: {"add": ["job_nimbus"], "invalidate": []},
    9: {"add": ["location_seattle"], "invalidate": ["location_portland"]},
    10: {"add": [], "invalidate": []},
}


def dialogue_upto(t):
    lines = ["Conversation:"]
    for i in range(t):
        lines.append(f"User: {TURNS[i][2]}")
    return "\n".join(lines) + "\n\n"


def concept_hits(text):
    t = text.lower()
    return {name for name, kws in CONCEPTS.items()
            if any(kw in t for kw in kws)}


def norm_lines(text):
    """Normalize profile lines for diffing: the seed profile uses ASCII '- '
    bullets but the model emits U+2022 (and sometimes en-dashes); strip any
    leading non-alphanumeric run so unchanged lines match across bullets.
    """
    out = []
    for ln in text.splitlines():
        s = " ".join(ln.strip().lower().split())
        s = s.lstrip("-\u2022\u2013\u2014*#\u00bb ")
        if not s or s.startswith("[") or s.startswith("you maintain"):
            continue
        out.append(s)
    return out


def line_diff(prev_text, cur_text):
    prev_left = list(norm_lines(prev_text))
    added, invalidated = [], []
    for ln in norm_lines(cur_text):
        if ln in prev_left:
            prev_left.remove(ln)
        else:
            added.append(ln)
    invalidated = prev_left
    return {"added": added, "invalidated": invalidated}


def score_events(turn, diff):
    """Match observed diff lines to hand-labeled events; TP/FP/FN."""
    exp = EXPECTED_EVENTS[turn]
    remaining_add = set(exp["add"])
    remaining_inv = set(exp["invalidate"])
    obs = []
    tp = fp = 0
    for kind, lines in (("add", diff["added"]), ("invalidate", diff["invalidated"])):
        for ln in lines:
            hits = {c for c, kws in CONCEPTS.items() if any(k in ln for k in kws)}
            pool = remaining_add if kind == "add" else remaining_inv
            got = hits & pool
            if got:
                tp += 1
                pool -= got
            else:
                fp += 1
            obs.append({"type": kind, "line": ln, "concepts": sorted(hits),
                        "matched": sorted(got)})
    fn = len(remaining_add) + len(remaining_inv)
    return {"observed": obs, "tp": tp, "fp": fp, "fn": fn,
            "missed_expected": sorted(remaining_add | remaining_inv)}


def turn_checks(t, profile_text):
    hits = concept_hits(profile_text)
    expected_present = sorted(c for c, turns in EXPECTED.items() if t in turns)
    expected_absent = sorted(c for c, turns in EXPECTED.items() if t not in turns)
    attrition = [c for c in expected_present if c not in hits]
    resurrection = [c for c in expected_absent if c in hits]
    return {
        "hits": sorted(hits),
        "expected_present": expected_present,
        "expected_absent": expected_absent,
        "attrition_missing": attrition,
        "resurrection_present": resurrection,
        "no_attrition": not attrition,
        "no_resurrection": not resurrection,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=int, default=10)
    ap.add_argument("--rescore", action="store_true",
                    help="re-analyze saved raw CLI JSON without re-running the GPU")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    turns = TURNS[: args.turns]
    if args.dry_run:
        print("plan:", len(turns), "turns; seed_base", SEED_BASE,
              "profile_0 = seed profile; batch k=2 + fresh solo Slot 0 per turn")
        return 0

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    profile_prev = SEED_PROFILE
    per_turn = []
    for t, kind, user_msg in turns:
        seed = SEED_BASE + t
        dialogue = dialogue_upto(t)
        s0_prompt = dialogue + S0_INSTRUCTION
        s1_prompt = (dialogue + S1_PROFILE_MARKER + "\n" + profile_prev
                     + "\n\n" + COMPACT_RULE)
        batch_tsv = RAW_DIR / f"turn{t:02d}_batch.tsv"
        solo_tsv = RAW_DIR / f"turn{t:02d}_solo.tsv"
        if not args.rescore:
            write_tsv(batch_tsv, [("s0", s0_prompt), ("s1", s1_prompt)])
            write_tsv(solo_tsv, [("s0", s0_prompt)])
            print(f"[run] turn={t} ({kind}) seed={seed} solo (serial, Slot 0)",
                  flush=True)
            solo = run_cli("serial", solo_tsv, seed, RAW_DIR / f"turn{t:02d}_solo.json")
            print(f"[run] turn={t} ({kind}) seed={seed} batch k=2", flush=True)
            batch = run_cli("batch", batch_tsv, seed, RAW_DIR / f"turn{t:02d}_batch.json")
        else:
            solo = load_raw(RAW_DIR / f"turn{t:02d}_solo.json")
            batch = load_raw(RAW_DIR / f"turn{t:02d}_batch.json")

        s0_solo = get_stream(solo, S0_MARKER)
        s0_batch = get_stream(batch, S0_MARKER)
        s1_batch = get_stream(batch, S1_PROFILE_MARKER)
        profile_cur = s1_batch["text"]

        checks = turn_checks(t, profile_cur)
        diff = line_diff(profile_prev, profile_cur)
        events = score_events(t, diff)
        tokens = s1_batch["generated"]
        rec = {
            "turn": t, "kind": kind, "user_message": user_msg, "seed": seed,
            "profile_prev": profile_prev,
            "profile": profile_cur,
            "profile_tokens": tokens,
            "profile_token_ok": tokens < PROFILE_TOKEN_MAX,
            "solo_slot0": summarize_branch(s0_solo),
            "batch_slot0": summarize_branch(s0_batch),
            "batch_slot1": summarize_branch(s1_batch),
            **s0_identity(s0_solo["text"], s0_batch["text"],
                          s0_solo["generated"], s0_batch["generated"]),
            "s0_prompt_identical": s0_solo["prompt"] == s0_batch["prompt"],
            "checks": checks,
            "diff": diff,
            "events": events,
            "expected_events": EXPECTED_EVENTS[t],
            "turn_ok": (checks["no_attrition"] and checks["no_resurrection"]
                        and tokens <= PROFILE_TOKEN_MAX),
            "ratios": {
                "completion": s0_batch["completion_ms"] / s0_solo["completion_ms"],
                "ttft": s0_batch["ttft_ms"] / s0_solo["ttft_ms"],
            },
        }
        per_turn.append(rec)
        profile_prev = profile_cur
        print(f"[turn {t}] ok={rec['turn_ok']} tok={tokens} "
              f"attrition={checks['attrition_missing']} "
              f"resurrect={checks['resurrection_present']} "
              f"adds={diff['added']} invs={diff['invalidated']} "
              f"ev_tp/fp/fn={events['tp']}/{events['fp']}/{events['fn']} "
              f"comp_ratio={rec['ratios']['completion']:.3f} "
              f"s0_identical={rec['s0_identical']} "
              f"finish={s1_batch['finish']}", flush=True)

    n = len(per_turn)
    comp_ratios = [r["ratios"]["completion"] for r in per_turn]
    ttft_ratios = [r["ratios"]["ttft"] for r in per_turn]
    med_comp, mean_comp = med(comp_ratios), statistics.mean(comp_ratios)
    med_ttft, mean_ttft = med(ttft_ratios), statistics.mean(ttft_ratios)
    turns_ok = sum(r["turn_ok"] for r in per_turn)
    token_ok = sum(r["profile_token_ok"] for r in per_turn)
    attrition_turns = [r["turn"] for r in per_turn if not r["checks"]["no_attrition"]]
    resurrect_turns = [r["turn"] for r in per_turn if not r["checks"]["no_resurrection"]]
    ident_ok = sum(r["s0_identical"] for r in per_turn)
    ev_tp = sum(r["events"]["tp"] for r in per_turn)
    ev_fp = sum(r["events"]["fp"] for r in per_turn)
    ev_fn = sum(r["events"]["fn"] for r in per_turn)
    precision = ev_tp / (ev_tp + ev_fp) if ev_tp + ev_fp else None
    recall = ev_tp / (ev_tp + ev_fn) if ev_tp + ev_fn else None

    criteria = {
        "no_attrition_all_turns": not attrition_turns,
        "no_resurrection_all_turns": not resurrect_turns,
        "profile_tokens_le_75_all_turns": token_ok == n,
        "slot0_completion_median_le_1_10": med_comp <= 1.10,
    }
    verdict = "meets" if all(criteria.values()) else "miss"

    summary = {
        "experiment": ("Angle 32c multi-turn compaction stability (10 turns, "
                       "k=2, angle-31 rule, profile feedback loop)"),
        "parent": "results/ingame_bleed_gate.json (angle 31)",
        "model": MODEL, "system": SYSTEM,
        "max_tokens": MAX_TOKENS, "temperature": TEMP, "top_p": TOP_P,
        "n_turns": n,
        "metadata": cli_provenance(),
        "seed_base": SEED_BASE,
        "seed_profile": SEED_PROFILE,
        "compaction_rule": COMPACT_RULE,
        "dialogue_format": ("user turns only (no assistant feedback into "
                            "subsequent prompts; Slot 0 prompt stays "
                            "independent of prior Slot 0 batch-divergence)"),
        "conversation_script": [
            {"turn": t, "kind": k, "user_message": m} for t, k, m in TURNS],
        "hand_labels": {
            "concept_keywords": {k: list(v) for k, v in CONCEPTS.items()},
            "expected_presence_turns": {k: sorted(v) for k, v in EXPECTED.items()},
            "expected_events": EXPECTED_EVENTS,
        },
        "profile_token_max": PROFILE_TOKEN_MAX,
        "acceptance": criteria,
        "turns_passing": turns_ok,
        "attrition_turns": attrition_turns,
        "resurrection_turns": resurrect_turns,
        "profile_tokens": [r["profile_tokens"] for r in per_turn],
        "slot0_text_identical_turns": ident_ok,
        "line_diff_events": {"tp": ev_tp, "fp": ev_fp, "fn": ev_fn,
                             "precision": precision, "recall": recall},
        "latency": {
            "completion_ratio_median": med_comp,
            "completion_ratio_mean": mean_comp,
            "ttft_ratio_median": med_ttft, "ttft_ratio_mean": mean_ttft,
            "solo0_completion_ms_median": med([r["solo_slot0"]["completion_ms"] for r in per_turn]),
            "batch_slot0_completion_ms_median": med([r["batch_slot0"]["completion_ms"] for r in per_turn]),
            "ratio_of_completion_medians": (
                med([r["batch_slot0"]["completion_ms"] for r in per_turn])
                / med([r["solo_slot0"]["completion_ms"] for r in per_turn])),
            "per_turn_completion_ratios": comp_ratios,
            "per_turn_ttft_ratios": ttft_ratios,
        },
        "verdict": verdict,
        "turns": per_turn,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                        encoding="utf-8")

    L = []
    L.append("Angle 32c - multi-turn compaction stability: k=2 parallel::Runtime")
    L.append(f"model={MODEL}  max_tokens={MAX_TOKENS} temp={TEMP} top_p={TOP_P} "
             f"system={SYSTEM!r}")
    L.append(f"cli_sha256={summary['metadata']['cli_sha256']}  "
             f"built={summary['metadata']['cli_built_mtime']}  "
             f"tree={summary['metadata']['git_tree']}")
    L.append(f"turns={n} seeds={SEED_BASE + 1}..{SEED_BASE + n}  "
             f"batch k=2 + fresh solo Slot 0 per turn; profile feedback loop")
    L.append("dialogue = user turns only; Slot 1 = angle-31 anti-bleed compaction "
             "rule + profile_{t-1}")
    L.append("")
    L.append("SCRIPT:")
    for t, k, m in TURNS[:n]:
        L.append(f"  {t:>2} {k:<10} {m}")
    L.append("")
    L.append(f"{'turn':>4} {'kind':<10} {'tok':>4} {'s0_r':>6} {'ident':>6} "
             f"{'attr':>4} {'resu':>4} {'ev':>7} {'gate':>5}")
    for r in per_turn:
        c = r["checks"]
        ev = r["events"]
        L.append(f"{r['turn']:>4} {r['kind']:<10} {r['profile_tokens']:>4} "
                 f"{r['ratios']['completion']:>6.3f} {str(r['s0_identical']):>6} "
                 f"{str(c['no_attrition']):>4} {str(c['no_resurrection']):>4} "
                 f"{ev['tp']}/{ev['fp']}/{ev['fn']:>1}  {str(r['turn_ok']):>5}")
    L.append(f"{'med':>4} {'':<10} {med([r['profile_tokens'] for r in per_turn]):>4.0f} "
             f"{med_comp:>6.3f} {ident_ok}/{n}")
    L.append("")
    for r in per_turn:
        L.append(f"turn {r['turn']} profile ({r['profile_tokens']} tok):")
        for ln in r["profile"].splitlines():
            if ln.strip():
                L.append(f"    {ln}")
        if r["diff"]["added"] or r["diff"]["invalidated"]:
            L.append(f"  diff ADD={r['diff']['added']} "
                     f"INVALIDATE={r['diff']['invalidated']} "
                     f"(expected add={r['expected_events']['add']} "
                     f"inv={r['expected_events']['invalidate']})")
        if not r["turn_ok"]:
            L.append(f"  FAIL attrition={r['checks']['attrition_missing']} "
                     f"resurrection={r['checks']['resurrection_present']} "
                     f"tokens={r['profile_tokens']}")
    L.append("")
    L.append(f"Attrition turns: {attrition_turns or 'none'}   "
             f"Resurrection turns: {resurrect_turns or 'none'}")
    L.append(f"Line-diff events vs hand labels: TP={ev_tp} FP={ev_fp} FN={ev_fn} "
             f"precision={precision:.2f} recall={recall:.2f}"
             if precision is not None else "no diff events")
    L.append(f"Slot 0 completion ratio batch/solo: median={med_comp:.3f} "
             f"mean={mean_comp:.3f} (solo med "
             f"{summary['latency']['solo0_completion_ms_median']:.1f} ms, "
             f"batch med "
             f"{summary['latency']['batch_slot0_completion_ms_median']:.1f} ms)")
    L.append(f"Slot 0 ttft ratio batch/solo:       median={med_ttft:.3f} "
             f"mean={mean_ttft:.3f}")
    L.append("ACCEPTANCE: " + ", ".join(f"{k}={'PASS' if v else 'FAIL'}"
                                        for k, v in criteria.items()))
    L.append(f"VERDICT: {verdict}")
    text = "\n".join(L) + "\n"
    OUT_TXT.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
