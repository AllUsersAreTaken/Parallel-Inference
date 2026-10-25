"""Angle 36 - delta compaction, 10 turns (RecycleLM; the angle-32c fix).

Angle 32c (run_ingame_multiturn.py) missed: the full-profile rewrite across
the scripted 10-turn conversation lost concepts on 10/10 turns (piano dropped
t1-2, Buster overwritten by Miso t4 and never recovered, the Honda line
emptied t1-4 by anti-bleed collateral, Portland resurrected t10 as a synonym
"Current Residence: Portland"; events P 0.53 / R 0.80; tokens <=75; Slot 0
median 1.057). This run changes ONLY Slot 1's task and adds a pure-Python
host state machine - parallel/ and app/parallel_cli.exe are untouched:

  Slot 1 stops rewriting the profile. It emits ONLY an atomic delta:

    {\"upsert\": {\"key\": \"new value\"}, \"delete\": [\"key\"]}

  and the host applies it (state.update(upsert); pop each delete key with
  pop(key, None) - never error). No attrition is possible by construction;
  the gate becomes DELTA correctness (missing upsert for a new fact = fail).

Slot 1 prompt (instruction-first like the angle-33/34/35 scribe prompts,
with the proven angle-27 "No markdown, no commentary." tail at the end of
the instruction block, embedded in the user text):

  {S1_INSTR} {dialogue_upto(t)}[User Profile]\\n{render_state(state_{t-1})}
  \\n\\nOutput:

where S1_INSTR is the prompt quoted in the angle-36 spec VERBATIM (JSON
schema + rules + tail), dialogue_upto(t) is angle-32c's dialogue builder
byte-identical (turns 1..t, user-only, "Conversation:" header), and
render_state is angle-32c's profile rendering style (S1_PROFILE_MARKER +
"- key: value" bullet lines, keys in the lowercase_snake_case the delta
schema mandates). The instruction sits first, then the same
conversation-then-profile context 32c showed, then "Output:" (angle-33
scribe shape). Slot 0 prompt is angle-32c's byte-for-byte.

Script: the angle-32c 10-turn script VERBATIM (asserted equal to
run_ingame_multiturn.TURNS at import; per-turn s0 prompts additionally
asserted equal to the prompts stored in angle 32c's raw solo JSONs).
Seed profile: angle 32c's seed profile as a dict (asserted equal to
run_ingame_multiturn.SEED_PROFILE); state_0 = location=Seattle,
pet="Buster (golden retriever, knee issues)", vehicle="2019 Honda
Civic", commute="25 minutes highway". Fresh solo arm (serial, Slot 0
only) + batch k=2 per turn, seed 3000+t, temp 0.2, max_tokens 160 - the
32c protocol unchanged.

Hand labels translated to KEY-LEVEL expectations (documented in the JSON
under hand_labels.key_level):
  t1  hobby key contains "piano"
  t2  location key contains "portland", not "seattle" (same-key upsert
      supersedes; no delete needed)
  t3  state unchanged (exact dict equality; prior deltas persist)
  t4  BOTH "buster" and "miso" present somewhere in state values (merged
      pet value OR a second pet key - either accepted if both animals
      present)
  t5  vehicle key contains "tesla"; "honda"/"civic" gone from every value
      ("2024" presence recorded as an OBSERVATION, not a gate)
  t6  state unchanged
  t7  "ranger" present and "buster" present (somewhere)
  t8  job key contains "nimbus"
  t9  location key contains "seattle", not "portland"; "portland" gone
      everywhere
  t10 state unchanged
Every turn: commute key present; no previously-established key lost (host
keeps untouched keys alive, but a model delete would lose one - that is
delta error and counts).

Gates (EXPLORATORY, non-preregistered - chain lesson):
  G1 delta bare json.loads 10/10 (robust_parse = bounded bracket repair,
     NO fence stripping, recorded as secondary diagnostic only)
  G2 key-level label checks all 10 turns (and concept-level no-attrition
     on the rendered state - a rename-key that hides a fact still counts
     as a label miss; both are reported)
  G3 no resurrection: concept-level expected-absent facts absent from the
     rendered state every turn (Seattle t2-8, honda t5+, portland t9-10);
     host keys are fixed so synonym resurrection is impossible; a
     popped-then-re-added key is recorded as a diagnostic
  G4 rendered state <=75 tokens every turn (counted with
     build/cuda-graphs/bin/llama-tokenize.exe on the exact embedded block
     text, BOS excluded; per-turn token files kept in raw/_state_tokens/)
  G5 Slot 0 completion ratio (batch/solo) median <=1.10

Also scored: delta sizes (Slot 1 generated tokens), event precision/recall
against the hand-labeled 32c transition list (events derived from the
state transition: a key added = ADD with its concepts; key removed =
INVALIDATE; value changed = ADD concepts(new)-concepts(old) and
INVALIDATE concepts(old)-concepts(new); TP/FP/FN accounting identical to
32c's score_events), and a final-state side-by-side vs angle 32c's final
profile (concept matrix: expected t10 | 32c final | delta-compaction
final).

Writes results/ingame_delta_compaction.{json,txt}. Raw per-turn CLI JSON
+ prompt TSVs in results/ingame_delta_compaction_raw/.
Run: python harness\\run_ingame_delta_compaction.py [--rescore] [--dry-run]
"""
import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ingame_common import (  # noqa: E402
    DLL_DIR, MAX_TOKENS, MODEL, PROFILE_TOKEN_MAX, ROOT, S0_INSTRUCTION,
    S0_MARKER, S1_PROFILE_MARKER, SYSTEM, TEMP, TOP_P, cli_provenance,
    get_stream, load_raw, med, run_cli, s0_identity, summarize_branch,
    write_tsv,
)
import run_ingame_multiturn as a32c  # noqa: E402

RAW_DIR = ROOT / "results" / "ingame_delta_compaction_raw"
OUT_JSON = ROOT / "results" / "ingame_delta_compaction.json"
OUT_TXT = ROOT / "results" / "ingame_delta_compaction.txt"
TOK_DIR = RAW_DIR / "_state_tokens"
TOKENIZE_EXE = ROOT / "build" / "cuda-graphs" / "bin" / "llama-tokenize.exe"
SEED_BASE = 3000

# Angle-36 Slot 1 instruction, VERBATIM from the spec.
S1_INSTR = (
    "You maintain the user's profile as key-value state. Output ONLY the "
    "atomic changes needed. JSON schema: {\"upsert\": {\"key\": \"new value\"}, "
    "\"delete\": [\"key\"]}. Rules: upsert keys whose value is new or changed; "
    "delete keys the user explicitly retracts or that are superseded; if "
    "nothing changed output {\"upsert\": {}, \"delete\": []}; keys you do not "
    "mention are left untouched \u2014 never restate the full profile; "
    "subject-style keys in lowercase_snake_case (e.g. location, pet, vehicle, "
    "commute, hobby, job). No markdown, no commentary."
)

# The angle-32c 10-turn script, copied VERBATIM from run_ingame_multiturn.py
# and asserted equal to it at startup (the task requires exact reuse).
SCRIPT_32C = [
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

# state_0 = the angle-32c seed profile as a dict.
SEED_STATE = {
    "location": "Seattle",
    "pet": "Buster (golden retriever, knee issues)",
    "vehicle": "2019 Honda Civic",
    "commute": "25 minutes highway",
}

KEY_LEVEL_LABELS = {
    1: "state['hobby'] (case-insensitive key match) contains 'piano'",
    2: "state['location'] contains 'portland' and not 'seattle' "
       "(same-key upsert supersedes Seattle; no delete needed)",
    3: "state == state after t2 (exact dict equality; prior deltas persist)",
    4: "'buster' AND 'miso' both present in state values (merged pet value "
       "OR a second pet key - both animals present is the acceptance)",
    5: "state['vehicle'] contains 'tesla'; 'honda'/'civic' absent from every "
       "value ('2024' presence = OBSERVATION, not gated)",
    6: "state == state after t5 (exact dict equality)",
    7: "'ranger' present in state values AND 'buster' present in state values",
    8: "state['job'] contains 'nimbus'",
    9: "state['location'] contains 'seattle' and not 'portland'; 'portland' "
       "absent from every value",
    10: "state == state after t9 (exact dict equality)",
}


def dialogue_upto(t):
    """Angle-32c dialogue builder, byte-identical (asserted in verify_reuse)."""
    return a32c.dialogue_upto(t)


def render_state(state):
    """Angle-32c profile rendering style: marker + '- key: value' bullets."""
    lines = [f"- {k}: {v}" for k, v in state.items()]
    return S1_PROFILE_MARKER + "\n" + "\n".join(lines)


def s0_prompt(dialogue):
    return dialogue + S0_INSTRUCTION


def s1_prompt(dialogue, state):
    return S1_INSTR + " " + dialogue + render_state(state) + "\n\nOutput:"


def bare_parse(text):
    """Direct json.loads on the stripped stream text - the G1 gate."""
    try:
        return json.loads(text.strip()), None
    except Exception as e:  # noqa: BLE001 - record the failure verbatim
        return None, f"{type(e).__name__}: {e}"


def robust_parse(text):
    """Bounded host-side bracket repair (secondary diagnostic only).

    Mirrors angles 34/35: NO fence stripping (a fenced output must fail);
    append closing brackets for at most two unbalanced opens.
    """
    t = text.strip()
    if t.startswith("```"):
        return None
    try:
        return json.loads(t)
    except Exception:  # noqa: BLE001
        pass
    fix = t
    for _ in range(2):
        if fix.count("[") > fix.count("]"):
            fix += "]"
        elif fix.count("{") > fix.count("}"):
            fix += "}"
        else:
            break
    try:
        return json.loads(fix)
    except Exception:  # noqa: BLE001
        return None


def shape_ok(delta):
    return (isinstance(delta, dict)
            and isinstance(delta.get("upsert"), dict)
            and isinstance(delta.get("delete"), list))


def apply_delta(state, delta):
    """Host state machine: state.update(upsert); pop deletes (never error)."""
    state = dict(state)
    rec = {"applied": False, "deleted_keys": [], "popped_missing": [],
           "contradictory_keys": [], "pop_errors": []}
    if not shape_ok(delta):
        rec["reason"] = "shape_invalid"
        return state, rec
    upsert = delta["upsert"]
    delete = delta["delete"]
    state.update(upsert)
    rec["contradictory_keys"] = sorted(
        set(map(str, upsert)) & set(map(str, delete)))
    for k in delete:
        try:
            if k in state:
                state.pop(k)
                rec["deleted_keys"].append(k)
            else:
                rec["popped_missing"].append(k)
        except TypeError as e:
            rec["pop_errors"].append(f"{k!r}: {e}")
    rec["applied"] = True
    return state, rec


def concepts_in(text):
    tl = str(text).lower()
    return {c for c, kws in a32c.CONCEPTS.items()
            if any(kw in tl for kw in kws)}


def state_events(prev, cur):
    """Key-level state-transition events (the delta's semantic effect)."""
    evs = []
    for k in sorted(set(prev) | set(cur)):
        old, new = prev.get(k), cur.get(k)
        if k not in prev:
            hits = concepts_in(f"{k}: {new}")
            evs.append({"type": "add", "key": k, "value": new,
                        "concepts": sorted(hits)})
        elif k not in cur:
            hits = concepts_in(f"{k}: {old}")
            evs.append({"type": "invalidate", "key": k, "value": old,
                        "concepts": sorted(hits)})
        elif old != new:
            oh, nh = concepts_in(f"{k}: {old}"), concepts_in(f"{k}: {new}")
            added, inv = nh - oh, oh - nh
            if added:
                evs.append({"type": "add", "key": k, "value": new,
                            "concepts": sorted(added)})
            if inv:
                evs.append({"type": "invalidate", "key": k, "value": old,
                            "concepts": sorted(inv)})
    return evs


def score_events(turn, evs):
    """TP/FP/FN vs the hand-labeled 32c transition list (32c semantics:
    one event matches at most one expected concept of its type)."""
    exp = a32c.EXPECTED_EVENTS[turn]
    pool_add = set(exp["add"])
    pool_inv = set(exp["invalidate"])
    tp = fp = 0
    obs = []
    for ev in evs:
        pool = pool_add if ev["type"] == "add" else pool_inv
        got = set(ev["concepts"]) & pool
        if got:
            tp += 1
            pool -= got
        else:
            fp += 1
        obs.append({**ev, "matched": sorted(got)})
    fn = len(pool_add) + len(pool_inv)
    return {"observed": obs, "tp": tp, "fp": fp, "fn": fn,
            "missed_expected": sorted(pool_add | pool_inv)}


def vget(state, key):
    for k, v in state.items():
        if str(k).lower() == key:
            return str(v).lower()
    return None


def label_checks(t, state, prev):
    """Key-level hand-label checks for turn t (all booleans -> label_ok)."""
    keys_now = {str(k).lower() for k in state}
    lost = sorted(str(k) for k in prev if str(k).lower() not in keys_now)
    vals = " || ".join(f"{k}: {v}" for k, v in state.items()).lower()
    ck = {"commute_key_present": "commute" in keys_now,
          "prev_keys_kept": not lost}
    diag = {"prev_keys_lost": lost, "state_values_lower": vals,
            "has_2019_anywhere": "2019" in vals,
            "has_2024_anywhere": "2024" in vals}
    if t == 1:
        hv = vget(state, "hobby")
        ck["hobby_piano"] = bool(hv) and "piano" in hv
        diag["piano_anywhere"] = "piano" in vals
    elif t == 2:
        lv = vget(state, "location")
        ck["location_portland_no_seattle"] = (
            bool(lv) and "portland" in lv and "seattle" not in lv)
        diag["portland_anywhere"] = "portland" in vals
    elif t in (3, 6, 10):
        ck["state_unchanged"] = state == prev
    elif t == 4:
        ck["both_animals_present"] = "buster" in vals and "miso" in vals
    elif t == 5:
        vv = vget(state, "vehicle")
        ck["vehicle_has_tesla"] = bool(vv) and "tesla" in vv
        ck["honda_civic_gone_everywhere"] = (
            "honda" not in vals and "civic" not in vals)
        diag["vehicle_value"] = vv
        diag["tesla_anywhere"] = "tesla" in vals
    elif t == 7:
        ck["ranger_and_buster_present"] = (
            "ranger" in vals and "buster" in vals)
    elif t == 8:
        jv = vget(state, "job")
        ck["job_nimbus"] = bool(jv) and "nimbus" in jv
        diag["nimbus_anywhere"] = "nimbus" in vals
    elif t == 9:
        lv = vget(state, "location")
        ck["location_seattle_no_portland"] = (
            bool(lv) and "seattle" in lv and "portland" not in lv)
        ck["portland_gone_everywhere"] = "portland" not in vals
        diag["location_value"] = lv
    return ck, diag, all(ck.values())


def state_tokens(text, out_file):
    """Exact BPE token count of the rendered block via llama-tokenize.exe.

    The tool prints one '<id> -> <piece>' line per token (a piece may span
    lines; each token still emits exactly one id-line). It prefixes BOS
    <|begin_of_text|> (128000), which is excluded - parallel_cli's
    `generated` counts also exclude BOS.
    """
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(text, encoding="utf-8", newline="\n")
    env = dict(os.environ)
    env["PATH"] = str(DLL_DIR) + os.pathsep + env.get("PATH", "")
    r = subprocess.run(
        [str(TOKENIZE_EXE), "-m", MODEL, "-f", str(out_file),
         "--n-gpu-layers", "0"],
        cwd=str(ROOT), env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=300)
    if r.returncode != 0:
        raise RuntimeError(f"llama-tokenize failed: {r.stderr.strip()}")
    ids = [int(m) for m in re.findall(r"^\s*(\d+) -> ",
                                      (r.stdout or "") + "\n" + (r.stderr or ""),
                                      flags=re.M)]
    if ids and ids[0] == 128000:
        ids = ids[1:]
    return len(ids)


def verify_reuse():
    """The 32c script + seed profile must be reused EXACTLY (task req)."""
    v = {"turns_equal_to_32c": a32c.TURNS == SCRIPT_32C,
         "seed_profile_equal_to_32c": a32c.SEED_PROFILE == (
             "- Location: Seattle\n"
             "- Pet: Buster (golden retriever, knee issues)\n"
             "- Vehicle: 2019 Honda Civic\n"
             "- Commute: 25 minutes highway"),
         "s0_prompt_matches_32c_raw": {}}
    for t, _kind, _msg in SCRIPT_32C:
        p32 = (ROOT / "results" / "ingame_multiturn_raw"
               / f"turn{t:02d}_solo.json")
        if p32.exists():
            doc = load_raw(p32)
            expected = s0_prompt(dialogue_upto(t))
            v["s0_prompt_matches_32c_raw"][t] = (
                doc["serial"][0]["prompt"] == expected)
    v["all_s0_prompts_match_32c"] = all(
        v["s0_prompt_matches_32c_raw"].values())
    v["rendered_seed_state"] = render_state(SEED_STATE)
    v["seed_render_matches_spec"] = v["rendered_seed_state"] == (
        "[User Profile]\n"
        "- location: Seattle\n"
        "- pet: Buster (golden retriever, knee issues)\n"
        "- vehicle: 2019 Honda Civic\n"
        "- commute: 25 minutes highway")
    if not (v["turns_equal_to_32c"] and v["seed_profile_equal_to_32c"]
            and v["all_s0_prompts_match_32c"]
            and v["seed_render_matches_spec"]):
        raise SystemExit(f"reuse verification FAILED: {v}")
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rescore", action="store_true",
                    help="re-analyze saved raw CLI JSON without re-running the GPU")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    reuse = verify_reuse()
    turns = SCRIPT_32C
    if args.dry_run:
        print("plan:", len(turns), "turns; seed_base", SEED_BASE,
              "state_0 = 32c seed profile dict; batch k=2 + fresh solo "
              "Slot 0 per turn; Slot 1 = delta prompt; reuse OK",
              json.dumps(reuse, indent=1))
        return 0

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    state_prev = dict(SEED_STATE)
    dead_keys = set()  # keys popped by model deletes in earlier turns
    per_turn = []
    t_gpu0 = time.monotonic()
    for t, kind, user_msg in turns:
        seed = SEED_BASE + t
        dialogue = dialogue_upto(t)
        s0p = s0_prompt(dialogue)
        s1p = s1_prompt(dialogue, state_prev)
        batch_tsv = RAW_DIR / f"turn{t:02d}_batch.tsv"
        solo_tsv = RAW_DIR / f"turn{t:02d}_solo.tsv"
        if not args.rescore:
            write_tsv(batch_tsv, [("s0", s0p), ("s1", s1p)])
            write_tsv(solo_tsv, [("s0", s0p)])
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
        delta_text = s1_batch["text"]

        delta, bare_err = bare_parse(delta_text)
        robust = robust_parse(delta_text)
        shape = shape_ok(delta) if delta is not None else False
        state_cur, apply_rec = apply_delta(state_prev, delta)
        rendered = render_state(state_cur)
        ck, diag, label_ok = label_checks(t, state_cur, state_prev)
        concept = a32c.turn_checks(t, rendered)
        evs = state_events(state_prev, state_cur)
        events = score_events(t, evs)
        stok = state_tokens(rendered, TOK_DIR / f"turn{t:02d}.txt")

        dead_key_readd = sorted(k for k in state_cur if k in dead_keys)
        dead_keys.update(map(str, apply_rec["deleted_keys"]))
        rec = {
            "turn": t, "kind": kind, "user_message": user_msg, "seed": seed,
            "state_prev": state_prev,
            "delta_raw": delta_text,
            "delta": delta,
            "delta_parse": {
                "bare_ok": delta is not None,
                "bare_error": bare_err,
                "robust_ok": robust is not None,
                "shape_ok": shape,
                "fences": "```" in delta_text,
                "ends_with": delta_text.strip()[-12:],
            },
            "apply": apply_rec,
            "dead_key_readd": dead_key_readd,
            "state": state_cur,
            "rendered_state": rendered,
            "state_tokens": stok,
            "state_tokens_ok": stok <= PROFILE_TOKEN_MAX,
            "label_checks": ck,
            "label_diag": diag,
            "label_ok": label_ok,
            "concept_checks": concept,
            "events": events,
            "expected_events": a32c.EXPECTED_EVENTS[t],
            "delta_tokens": s1_batch["generated"],
            "solo_slot0": summarize_branch(s0_solo),
            "batch_slot0": summarize_branch(s0_batch),
            "batch_slot1": summarize_branch(s1_batch),
            **s0_identity(s0_solo["text"], s0_batch["text"],
                          s0_solo["generated"], s0_batch["generated"]),
            "s0_prompt_identical": s0_solo["prompt"] == s0_batch["prompt"],
            "turn_ok": (delta is not None and label_ok
                        and concept["no_attrition"]
                        and concept["no_resurrection"]
                        and stok <= PROFILE_TOKEN_MAX),
            "ratios": {
                "completion": s0_batch["completion_ms"] / s0_solo["completion_ms"],
                "ttft": s0_batch["ttft_ms"] / s0_solo["ttft_ms"],
            },
        }
        per_turn.append(rec)
        state_prev = state_cur
        print(f"[turn {t}] ok={rec['turn_ok']} bare={rec['delta_parse']['bare_ok']} "
              f"label={label_ok} attr={concept['attrition_missing']} "
              f"resu={concept['resurrection_present']} "
              f"delta={delta_text.strip()!r} "
              f"state_tok={stok} delta_tok={rec['delta_tokens']} "
              f"ev={events['tp']}/{events['fp']}/{events['fn']} "
              f"comp_ratio={rec['ratios']['completion']:.3f} "
              f"finish={s1_batch['finish']}", flush=True)

    gpu_wall_s = time.monotonic() - t_gpu0
    print(f"[gpu] CLI wall time {gpu_wall_s:.1f} s", flush=True)

    # token counting is CPU-only but runs llama-tokenize loads; include it
    n = len(per_turn)
    comp_ratios = [r["ratios"]["completion"] for r in per_turn]
    ttft_ratios = [r["ratios"]["ttft"] for r in per_turn]
    med_comp, mean_comp = med(comp_ratios), statistics.mean(comp_ratios)
    med_ttft, mean_ttft = med(ttft_ratios), statistics.mean(ttft_ratios)
    bare_ok_n = sum(r["delta_parse"]["bare_ok"] for r in per_turn)
    robust_ok_n = sum(r["delta_parse"]["robust_ok"] for r in per_turn)
    shape_ok_n = sum(r["delta_parse"]["shape_ok"] for r in per_turn)
    fence_n = sum(r["delta_parse"]["fences"] for r in per_turn)
    label_turns = [r["turn"] for r in per_turn if not r["label_ok"]]
    attrition_turns = [r["turn"] for r in per_turn
                       if not r["concept_checks"]["no_attrition"]]
    resurrect_turns = [r["turn"] for r in per_turn
                       if not r["concept_checks"]["no_resurrection"]]
    state_tok_ok = sum(r["state_tokens_ok"] for r in per_turn)
    ident_ok = sum(r["s0_identical"] for r in per_turn)
    ev_tp = sum(r["events"]["tp"] for r in per_turn)
    ev_fp = sum(r["events"]["fp"] for r in per_turn)
    ev_fn = sum(r["events"]["fn"] for r in per_turn)
    precision = ev_tp / (ev_tp + ev_fp) if ev_tp + ev_fp else None
    recall = ev_tp / (ev_tp + ev_fn) if ev_tp + ev_fn else None

    criteria = {
        "G1_delta_bare_parse_10_of_10": bare_ok_n == n,
        "G2_key_level_labels_all_turns": not label_turns and not attrition_turns,
        "G3_no_resurrection_all_turns": not resurrect_turns,
        "G4_state_tokens_le_75_all_turns": state_tok_ok == n,
        "G5_slot0_completion_median_le_1_10": med_comp <= 1.10,
    }
    verdict = "meets" if all(criteria.values()) else "miss"

    # Final-state side-by-side vs angle 32c.
    p32c = load_raw(ROOT / "results" / "ingame_multiturn.json")
    final_32c = p32c["turns"][-1]["profile"]
    final_delta = per_turn[-1]["rendered_state"]
    concept_matrix = {}
    for c in sorted(a32c.CONCEPTS):
        concept_matrix[c] = {
            "expected_t10": 10 in a32c.EXPECTED[c],
            "in_32c_final": c in a32c.concept_hits(final_32c),
            "in_delta_final": c in a32c.concept_hits(final_delta),
        }

    summary = {
        "experiment": ("Angle 36 delta compaction (10 turns, k=2, Slot 1 "
                       "emits atomic {upsert,delete} deltas, pure-Python "
                       "host state machine; the angle-32c fix)"),
        "parent": "results/ingame_multiturn.json (angle 32c)",
        "model": MODEL, "system": SYSTEM,
        "max_tokens": MAX_TOKENS, "temperature": TEMP, "top_p": TOP_P,
        "n_turns": n,
        "metadata": cli_provenance(),
        "seed_base": SEED_BASE,
        "seed_profile_32c": a32c.SEED_PROFILE,
        "state_0": SEED_STATE,
        "s1_instruction": S1_INSTR,
        "s1_prompt_construction": (
            "S1_INSTR (spec-verbatim) + ' ' + angle-32c dialogue_upto(t) "
            "(Conversation: header, user turns 1..t, trailing blank line) + "
            "render_state(state_{t-1}) in angle-32c profile style "
            "('S1_PROFILE_MARKER' + '- key: value' bullets, "
            "lowercase_snake_case keys) + '\\n\\nOutput:'; instruction-first "
            "scribe shape (angles 33-35), proximal 'No markdown, no "
            "commentary.' tail closes the instruction block"),
        "s0_prompt": "angle-32c dialogue_upto(t) + S0_INSTRUCTION (byte-identical)",
        "reuse_verification": reuse,
        "gates_note": "exploratory (non-preregistered) per chain lesson",
        "hand_labels": {
            "key_level": {str(k): v for k, v in KEY_LEVEL_LABELS.items()},
            "concept_keywords": {k: list(v) for k, v in a32c.CONCEPTS.items()},
            "expected_presence_turns": {k: sorted(v)
                                        for k, v in a32c.EXPECTED.items()},
            "expected_events": a32c.EXPECTED_EVENTS,
            "universal": ("commute key present every turn; no "
                          "previously-established key lost; rendered state "
                          "<=75 tok; previously-established facts persist by "
                          "host construction"),
        },
        "host_state_machine": (
            "state.update(upsert); for k in delete: state.pop(k, None) "
            "(never error); apply only when shape_ok "
            "(upsert: dict, delete: list); contradictory upsert+delete "
            "same-key recorded (update-then-pop per spec)"),
        "profile_token_max": PROFILE_TOKEN_MAX,
        "acceptance": criteria,
        "turns_passing": sum(r["turn_ok"] for r in per_turn),
        "bare_parse_passing": bare_ok_n,
        "robust_parse_passing": robust_ok_n,
        "shape_ok_turns": shape_ok_n,
        "fence_turns": [r["turn"] for r in per_turn
                        if r["delta_parse"]["fences"]],
        "label_fail_turns": label_turns,
        "attrition_turns": attrition_turns,
        "resurrection_turns": resurrect_turns,
        "state_tokens": [r["state_tokens"] for r in per_turn],
        "delta_tokens": [r["delta_tokens"] for r in per_turn],
        "delta_tokens_median": med([r["delta_tokens"] for r in per_turn]),
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
        "final_state_side_by_side": {
            "concept_matrix": concept_matrix,
            "delta_final_state": final_delta,
            "delta_final_keys": list(per_turn[-1]["state"].keys()),
            "angle32c_final_profile": final_32c,
            "angle32c_final_attrition": a32c.turn_checks(10, final_32c)["attrition_missing"],
            "delta_final_attrition": per_turn[-1]["concept_checks"]["attrition_missing"],
        },
        "gpu_wall_s": round(gpu_wall_s, 1),
        "verdict": verdict,
        "turns": per_turn,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                        encoding="utf-8")

    L = []
    L.append("Angle 36 - delta compaction (the angle-32c fix): k=2 parallel::Runtime")
    L.append(f"model={MODEL}  max_tokens={MAX_TOKENS} temp={TEMP} top_p={TOP_P} "
             f"system={SYSTEM!r}")
    L.append(f"cli_sha256={summary['metadata']['cli_sha256']}  "
             f"built={summary['metadata']['cli_built_mtime']}  "
             f"tree={summary['metadata']['git_tree']}")
    L.append(f"turns={n} seeds={SEED_BASE + 1}..{SEED_BASE + n}  "
             f"batch k=2 + fresh solo Slot 0 per turn; host state machine "
             f"(update + pop); GPU wall {gpu_wall_s:.1f} s")
    L.append("dialogue = 32c script VERBATIM (asserted); Slot 0 = 32c s0 "
             "byte-identical (asserted vs 32c raw); Slot 1 = delta instruction")
    L.append("gates EXPLORATORY (non-preregistered)")
    L.append("")
    L.append("SCRIPT (angle-32c verbatim):")
    for t, k, m in turns:
        L.append(f"  {t:>2} {k:<10} {m}")
    L.append("")
    L.append(f"{'turn':>4} {'kind':<10} {'bare':>4} {'lbl':>4} {'attr':>4} "
             f"{'resu':>4} {'sTok':>4} {'dTok':>4} {'s0_r':>6} {'ident':>6} "
             f"{'ev':>7} {'gate':>5}")
    for r in per_turn:
        c = r["concept_checks"]
        ev = r["events"]
        L.append(f"{r['turn']:>4} {r['kind']:<10} "
                 f"{str(r['delta_parse']['bare_ok']):>4} "
                 f"{str(r['label_ok']):>4} {str(c['no_attrition']):>4} "
                 f"{str(c['no_resurrection']):>4} {r['state_tokens']:>4} "
                 f"{r['delta_tokens']:>4} {r['ratios']['completion']:>6.3f} "
                 f"{str(r['s0_identical']):>6} "
                 f"{ev['tp']}/{ev['fp']}/{ev['fn']:>1}  {str(r['turn_ok']):>5}")
    L.append(f"{'med':>4} {'':<10} {'':>4} {'':>4} {'':>4} {'':>4} "
             f"{med([r['state_tokens'] for r in per_turn]):>4.0f} "
             f"{med([r['delta_tokens'] for r in per_turn]):>4.0f} "
             f"{med_comp:>6.3f} {ident_ok}/{n}")
    L.append("")
    for r in per_turn:
        L.append(f"turn {r['turn']} delta ({r['delta_tokens']} tok, "
                 f"finish={r['batch_slot1']['finish']}, "
                 f"bare={r['delta_parse']['bare_ok']}, "
                 f"robust={r['delta_parse']['robust_ok']}):")
        L.append(f"    {r['delta_raw'].strip()}")
        L.append(f"  state after ({r['state_tokens']} tok):")
        for ln in r["rendered_state"].splitlines():
            if ln.strip():
                L.append(f"    {ln}")
        if r["events"]["observed"]:
            L.append(f"  events: {[(e['type'], e['key'], e['concepts']) for e in r['events']['observed']]} "
                     f"(expected add={r['expected_events']['add']} "
                     f"inv={r['expected_events']['invalidate']})")
        if not r["label_ok"]:
            L.append(f"  FAIL label checks: "
                     f"{[k for k, v in r['label_checks'].items() if not v]} "
                     f"diag={r['label_diag']}")
        if not r["concept_checks"]["no_attrition"]:
            L.append(f"  FAIL concept attrition={r['concept_checks']['attrition_missing']}")
        if not r["concept_checks"]["no_resurrection"]:
            L.append(f"  FAIL concept resurrection={r['concept_checks']['resurrection_present']}")
        if r["apply"]["deleted_keys"] or r["apply"]["contradictory_keys"]:
            L.append(f"  apply: deleted={r['apply']['deleted_keys']} "
                     f"contradictory={r['apply']['contradictory_keys']}")
    L.append("")
    L.append(f"Label-fail turns: {label_turns or 'none'}   "
             f"Concept attrition turns: {attrition_turns or 'none'}   "
             f"Resurrection turns: {resurrect_turns or 'none'}")
    L.append(f"Bare parse: {bare_ok_n}/{n}  robust parse: {robust_ok_n}/{n}  "
             f"shape ok: {shape_ok_n}/{n}  fences: {fence_n}/{n}")
    L.append(f"State-transition events vs hand labels: TP={ev_tp} FP={ev_fp} "
             f"FN={ev_fn} precision="
             f"{precision:.2f} recall={recall:.2f}"
             if precision is not None else "no events")
    L.append(f"Delta sizes (Slot 1 generated tok): "
             f"{[r['delta_tokens'] for r in per_turn]} "
             f"median={summary['delta_tokens_median']}")
    L.append("")
    L.append("FINAL STATE SIDE-BY-SIDE (concept | expected t10 | 32c final | delta final):")
    for c, row in concept_matrix.items():
        L.append(f"  {c:<22} {str(row['expected_t10']):>12} "
                 f"{str(row['in_32c_final']):>11} {str(row['in_delta_final']):>12}")
    L.append("  delta final state:")
    for ln in final_delta.splitlines():
        L.append(f"    {ln}")
    L.append("  angle-32c final profile:")
    for ln in final_32c.splitlines():
        L.append(f"    {ln}")
    L.append("")
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
