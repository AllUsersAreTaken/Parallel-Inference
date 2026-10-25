"""Angle 35 - tail-only ablation of angle 34 (k=3 in-game pipeline).

Attribution question: angle 34 (harness/run_ingame_k3_fix.py) fixed fences
5/5->0/5 and strict relations 0/5->4/5 with THREE Slot 1 patches (proximal
anti-fence tail + rules 5-6 + max_tokens 160->256). Did the proximal
anti-fence tail ALONE suppress fences, or did rules 5-6 (Subject must be
"User"; Maximum 3 relations) co-confound? This run keeps ONE patch only:

  Slot 1 prompt = angle-33's Slot 1 prompt VERBATIM (rules 1-4 + Schema,
  NO rules 5-6) + ONLY the proximal tail sentence
  "Output ONLY a JSON object. No markdown, no code fences, no commentary."
  placed exactly where angle 34 placed it: on its own line immediately
  BEFORE "Output:" (last instruction before decode).
  max_tokens = 160 (the angle-33 shared default; patch count stays minimal
  -- the cap is not a prompt patch and angle 33's budget was only folded
  into a parse failure at seed 1236 because of FENCES, which the tail is
  the candidate fix for).

Slot 0, Slot 2, dialogue, solo arm byte-identical to angle 33 AND angle 34
(verified: solo TSV byte-identical; batch s0/s2 rows byte-identical to
both). TSV derivation from angle 33 is checked strictly: the decoded s1
prompt must equal angle 33's s1 prompt with the tail line inserted
immediately before the final "Output:" (unified diff: exactly ONE added
line -- the tail -- and ZERO removed lines); the recorded diff is stored
in the results JSON.

Protocol identical to angles 33/34: batch row seeds follow
parallel_cli.cpp:108 (row i samples seed+i), fresh solo serial Slot 0 arm,
temp 0.2, top_p 0.9, seeds 1234-1238, model Llama-3.2-3B Q4_K_M on the
parallel::Runtime CLI (parallel_cli.exe, SHA256 recorded).

Scoring identical to angle 34 (byte-equal matchers/gates):
  - Gate 1 parses: BARE json.loads, NO fence stripping (comparable first).
  - robust_parse() is recorded as a SECONDARY diagnostic only (bounded
    host-side brace repair: append at most one '}' when the stripped text
    ends with ']]'; no fence stripping inside -- a fenced output must FAIL
    robust_parse and be reported as a fail). Gate verdicts never use it.
  - Fences: '```' anywhere in the raw slot text (expect 0 with the tail).
  - Relation matcher: strict subject {user, the user} (gate 2); lenient
    {user, the user, i, me} diagnostic; DRIVES/OWNS predicate families;
    object contains 'tesla' and neither 'honda' nor 'civic'.
  - Retrieval-hook matcher: regex \\b(tesla|ev|evs|charging|car|cars)\\b.
  - Host closure: deterministic mock graph update on the BARE-parsed
    strict match (comparability); robust-based closure recorded only as a
    diagnostic inside the robust block.
  - Slots 0/2 latency + identity: completion ratio batch(k=3)/solo median
    <= 1.12 (gate 5), s0 text identity, tokens, finish reasons,
    relations count (no max-3 rule in this prompt: over-3 is diagnostic).

Host processing (pure Python, post-EOS, no GPU) is unchanged from angles
33/34: mock graph seeded with (User, DRIVES, 2019 Honda Civic, valid_from=
t0, invalid_at=NULL); a new-vehicle Slot 1 relation invalidates the Honda
edge and inserts the new edge valid_from=now.

Writes results/ingame_k3_tail_ablation.json and
results/ingame_k3_tail_ablation.txt. Raw per-rep CLI JSON in
results/ingame_k3_tail_ablation_raw/; prompts in
data/ingame_k3_tail_{batch,solo}.tsv.
Run: python harness\\run_ingame_k3_tail.py [--reps N] [--rescore] [--dry-run]
"""
import argparse
import difflib
import json
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ingame_common import (  # noqa: E402
    MAX_TOKENS, MODEL, ROOT, S0_MARKER, SEEDS, SYSTEM, TEMP, TOP_P,
    cli_provenance, get_stream, load_raw, med, run_cli, s0_identity,
    summarize_branch, write_tsv,
)

SRC_S0 = "data/ingame_mem_solo.tsv"          # angle 27 s0 row (byte source)
A33_BATCH_TSV = "data/ingame_k3_batch.tsv"    # angle 33 (diff baseline)
A33_SOLO_TSV = "data/ingame_k3_solo.tsv"      # angle 33 (diff baseline)
A34_BATCH_TSV = "data/ingame_k3_fix_batch.tsv"  # angle 34 (cross-check)
A34_SOLO_TSV = "data/ingame_k3_fix_solo.tsv"    # angle 34 (cross-check)
A33_JSON = ROOT / "results" / "ingame_k3_pipeline.json"
A34_JSON = ROOT / "results" / "ingame_k3_fix.json"
BATCH_TSV = "data/ingame_k3_tail_batch.tsv"
SOLO_TSV = "data/ingame_k3_tail_solo.tsv"
RAW_DIR = ROOT / "results" / "ingame_k3_tail_ablation_raw"
OUT_JSON = ROOT / "results" / "ingame_k3_tail_ablation.json"
OUT_TXT = ROOT / "results" / "ingame_k3_tail_ablation.txt"

TAIL_MAX_TOKENS = MAX_TOKENS   # 160, angle-33 value (minimal patch count)

S1_MARKER = "Extract durable personal facts"
S2_MARKER = "hypothetical questions"

# Angle-33 instruction, VERBATIM: rules 1-4 + Schema line only, NO rules
# 5-6 (those belong to angle 34 and are the co-confound under test).
S1_INSTR = (
    "Extract durable personal facts, preferences, and explicit relationship "
    "triples from the user turn. Do not summarize dialogue or track "
    "conversational state. Extract atomic facts and entity relations. "
    "Rules: 1. \"facts\": Plain language statements of new information. "
    "2. \"relations\": Directed triples [Subject, Predicate, Object] using "
    "uppercase predicates (e.g., OWNS, LIKES, DISLIKES, LOCATED_IN, DRIVES). "
    "3. If no memorable information is present, output empty lists. "
    "4. Output strict JSON only. "
    "Schema: {\"facts\": [\"string\"], "
    "\"relations\": [[\"string\", \"string\", \"string\"]]}."
)
S2_INSTR = (
    "Given the user's statement, generate 3 hypothetical questions or search "
    "queries the user might ask in the future where this information would "
    "be critical to know. Output strict JSON only. "
    "Schema: {\"queries\": [\"string\", \"string\", \"string\"]}."
)
# The ONLY patch: proximal anti-fence tail, byte-identical to angle 34's,
# same placement (its own line immediately before "Output:").
S1_TAIL = ("Output ONLY a JSON object. No markdown, no code fences, "
           "no commentary.")

# Angle-27 facts label (run_ingame_memory.py), reused for facts-list recall.
FACTS_LABEL = {
    "buster_knee_issue": ["buster", "knee"],
    "stick_to_flat_river_trail": ["flat", ("trail", "river")],
    "user_bought_tesla": ["tesla"],
    "needs_charging_stations": ["charg", ("station", "commute")],
}

DRIVES_FAM = {"drive", "drives", "driving"}
OWNS_FAM = {"own", "owns", "owned", "owning", "has", "have", "bought",
            "purchased", "acquired"}
SUBJECTS_OK = {"user", "the user"}
SUBJECTS_OK_LENIENT = SUBJECTS_OK | {"i", "me"}   # diagnostic only
HOOK_RE = re.compile(r"\b(tesla|ev|evs|charging|car|cars)\b", re.IGNORECASE)

SEED_EDGE = {"id": "e0", "subject": "User", "predicate": "DRIVES",
             "object": "2019 Honda Civic", "valid_from": "t0",
             "invalid_at": None}

# Byte-equal to angle 34's matcher doc (which is angle 33's plus the
# lenient-diagnostic and gate-2-scope clarifications).
MATCHER_DOC = {
    "relation_subject": ("casefold + collapse whitespace + strip edge "
                         "punctuation; accept 'user' or 'the user'"),
    "relation_predicate": ("DRIVES family {drive, drives, driving} or OWNS "
                           "family {own, owns, owned, owning, has, have, "
                           "bought, purchased, acquired}, casefold-normalized"),
    "relation_object_new_vehicle": ("contains 'tesla' and contains neither "
                                    "'honda' nor 'civic' (case-insensitive "
                                    "substring)"),
    "relation_gate": ("triple is a 3-string JSON array AND subject AND "
                      "predicate AND object all pass; first match wins; "
                      "gate 2 uses the STRICT subject set on the direct "
                      "parse only"),
    "relation_lenient_diagnostic": ("non-gating: same scoring with subject "
                                    "set {'user', 'the user', 'i', 'me'}"),
    "retrieval_hook": (r"regex \b(tesla|ev|evs|charging|car|cars)\b "
                       "(case-insensitive) on each Slot 2 query; seed hits "
                       "if >=1 query matches"),
    "fence": "'```' anywhere in the raw slot text",
    "comment": ("any line whose stripped form starts with '//' or '/*' "
                "(angle-28 style comment check)"),
    "host_closure": ("mock graph seeded with (User, DRIVES, 2019 Honda "
                     "Civic, valid_from=t0, invalid_at=NULL); on a "
                     "new-vehicle triple set invalid_at=now on the open "
                     "Honda/DRIVES-family edge and append the new edge "
                     "valid_from=now; closure_ok requires both"),
}

ROBUST_PARSE_RULE = (
    "cleaned = raw.strip(); if cleaned.endswith(']]') and not "
    "cleaned.endswith('}'): cleaned += '}'; json.loads(cleaned)"
)


def tsv_unescape(s):
    """Mirror parallel_cli.cpp unescape() (single left-to-right pass)."""
    o, i = [], 0
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s):
            d = s[i + 1]
            if d == "n":
                o.append("\n"); i += 2; continue
            if d == "t":
                o.append("\t"); i += 2; continue
            if d == "r":
                o.append("\r"); i += 2; continue
            if d == "\\":
                o.append("\\"); i += 2; continue
        o.append(c)
        i += 1
    return "".join(o)


def load_angle27_s0():
    """Unescape the angle-27 s0 row; assert it carries the expected shape."""
    line = (ROOT / SRC_S0).read_text(encoding="utf-8").splitlines()[0]
    rid, esc = line.split("\t", 1)
    if rid != "s0":
        raise SystemExit(f"{SRC_S0}: expected row id 's0', got {rid!r}")
    prompt = tsv_unescape(esc)
    if not prompt.startswith("Conversation:\n"):
        raise SystemExit(f"{SRC_S0}: s0 prompt lacks 'Conversation:\\n' header")
    if "\n\nReply to the user" not in prompt:
        raise SystemExit(f"{SRC_S0}: s0 prompt lacks reply instruction")
    return prompt


def read_tsv_rows(path):
    rows = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        rid, esc = line.split("\t", 1)
        rows[rid] = esc
    return rows


def verify_tsvs(old_s1_prompt):
    """Strict derivation check vs angle 33 (+ cross-check vs angle 34).

    Required: solo byte-identical to both, batch s0/s2 rows byte-identical
    to both, s1 row differs from angle 33 ONLY by the tail line inserted
    immediately before the final 'Output:' (exactly one added line, zero
    removed lines).
    """
    report = {}
    ok = True

    for tag, path in (("angle33", A33_SOLO_TSV), ("angle34", A34_SOLO_TSV)):
        same = (ROOT / path).read_bytes() == (ROOT / SOLO_TSV).read_bytes()
        report[f"solo_byte_identical_vs_{tag}"] = same
        ok &= same

    a33 = read_tsv_rows(ROOT / A33_BATCH_TSV)
    a34 = read_tsv_rows(ROOT / A34_BATCH_TSV)
    b = read_tsv_rows(ROOT / BATCH_TSV)
    for rows, tag in ((a33, "angle33"), (a34, "angle34")):
        if set(rows) != {"s0", "s1", "s2"} or set(b) != {"s0", "s1", "s2"}:
            raise SystemExit(f"TSV verify: expected row ids {{s0,s1,s2}} "
                             f"({tag} vs ours)")
        for slot in ("s0", "s2"):
            same = rows[slot] == b[slot]
            report[f"batch_{slot}_row_identical_vs_{tag}"] = same
            ok &= same

    s1_differs = a33["s1"] != b["s1"]
    report["batch_s1_row_differs_vs_angle33"] = s1_differs
    ok &= s1_differs

    new_s1 = tsv_unescape(b["s1"])
    # Byte-exact expectation: angle-33 s1 + tail line before final Output:
    anchor = "\nOutput:"
    i = old_s1_prompt.rfind(anchor)
    expected = (old_s1_prompt[:i + 1] + S1_TAIL + old_s1_prompt[i:])
    tail_line_exact = new_s1 == expected
    report["s1_equals_angle33_plus_tail_before_output"] = tail_line_exact
    ok &= tail_line_exact

    dlines = list(difflib.unified_diff(
        old_s1_prompt.splitlines(), new_s1.splitlines(),
        fromfile="angle33_s1", tofile="tail_s1", lineterm=""))
    added = [l[1:] for l in dlines
             if l.startswith("+") and not l.startswith("+++")]
    removed = [l[1:] for l in dlines
               if l.startswith("-") and not l.startswith("---")]
    tail_only = added == [S1_TAIL] and not removed
    report["s1_diff_tail_line_only"] = tail_only
    report["s1_diff_added_lines"] = added
    report["s1_diff_removed_lines"] = removed
    report["s1_prompt_unified_diff"] = dlines
    ok &= tail_only

    a34_s1_differs = a34["s1"] != b["s1"]
    report["batch_s1_row_differs_vs_angle34_rules_removed"] = a34_s1_differs
    ok &= a34_s1_differs

    report["ok"] = bool(ok)
    report["note"] = ("verified derivation: solo + s0/s2 rows byte-identical "
                      "to angles 33 and 34; s1 row = angle-33 s1 with ONLY "
                      "the proximal tail line inserted before 'Output:' "
                      "(rules 5-6 absent)")
    if not ok:
        raise SystemExit("TSV verify FAILED: " + json.dumps(
            {k: v for k, v in report.items() if isinstance(v, bool)}))
    return report


S0_PROMPT = load_angle27_s0()
DIALOGUE_TURNS = S0_PROMPT[len("Conversation:\n"):].split("\n\n")[0]
S1_PROMPT = f"{S1_INSTR} Conversation:\n{DIALOGUE_TURNS}\n\n{S1_TAIL}\nOutput:"
S2_PROMPT = f"{S2_INSTR} Conversation:\n{DIALOGUE_TURNS}\n\nOutput:"
A33_S1_PROMPT = tsv_unescape(read_tsv_rows(ROOT / A33_BATCH_TSV)["s1"])


def match_concepts(text):
    """Angle-27 concept matcher, restricted to the facts label."""
    t = text.lower()
    hits = []
    for name, groups in FACTS_LABEL.items():
        ok = True
        for g in groups:
            opts = (g,) if isinstance(g, str) else tuple(g)
            if not any(o in t for o in opts):
                ok = False
                break
        if ok:
            hits.append(name)
    return hits


def direct_json(text):
    """Gate-1 parse: bare json.loads, NO fence stripping."""
    try:
        return json.loads(text), None
    except ValueError as e:
        return None, str(e)[:200]


def robust_parse(raw):
    """SECONDARY diagnostic: bounded deterministic host-side brace repair.

    Exact rule: strip; if the text ends with "]]" append ONE "}" and parse.
    At most one "}" is appended and NOTHING else is edited; fences are never
    stripped inside robust_parse -- a fenced or otherwise-malformed output
    must fail here and be reported as a fail. Never used for gate verdicts
    (bare parse is reported first for comparability with angles 33/34).
    """
    cleaned = raw.strip()
    repair_applied = False
    if cleaned.endswith("]]") and not cleaned.endswith("}"):
        cleaned += "}"
        repair_applied = True
    try:
        return json.loads(cleaned), None, repair_applied
    except ValueError as e:
        return None, str(e)[:200], repair_applied


def diag_parse(text):
    """Non-gating fallback probe recorded only when direct parse failed."""
    t = text.strip()
    if t.startswith("```"):
        lines = [ln for ln in t.splitlines() if not ln.strip().startswith("```")]
        try:
            json.loads("\n".join(lines).strip())
            return "fence_stripped_ok"
        except ValueError:
            pass
    i = t.find("{")
    if i >= 0:
        depth = 0
        for j in range(i, len(t)):
            if t[j] == "{":
                depth += 1
            elif t[j] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        json.loads(t[i:j + 1])
                        return "brace_scan_ok"
                    except ValueError:
                        break
    return "failed"


def fence_strip_json(text):
    """Non-gating: strip ``` fence lines, then bare json.loads."""
    t = text.strip()
    lines = [ln for ln in t.splitlines() if not ln.strip().startswith("```")]
    try:
        return json.loads("\n".join(lines).strip()), None
    except ValueError as e:
        return None, str(e)[:200]


def has_comment(text):
    return any(ln.lstrip().startswith("//") or ln.lstrip().startswith("/*")
               for ln in text.splitlines())


def _norm(s):
    s = re.sub(r"\s+", " ", s.casefold().strip())
    return s.strip(" .,;:!?\"'()[]{}")


def score_relations(obj, subjects=SUBJECTS_OK):
    out = {"relations_is_list": False, "triples": [], "malformed": 0,
           "subject_user_triples": [], "matched": None, "matched_norm": None,
           "subjects_accepted": sorted(subjects)}
    if not isinstance(obj, dict):
        return out
    rels = obj.get("relations")
    if not isinstance(rels, list):
        return out
    out["relations_is_list"] = True
    for r in rels:
        if not (isinstance(r, list) and len(r) == 3
                and all(isinstance(x, str) for x in r)):
            out["malformed"] += 1
            out["triples"].append({"raw": r, "ok": False,
                                   "why": "not a 3-string array"})
            continue
        s_n, p_n, o_n = _norm(r[0]), _norm(r[1]), _norm(r[2])
        subj_ok = s_n in subjects
        fam = ("drives" if p_n in DRIVES_FAM
               else "owns" if p_n in OWNS_FAM else None)
        obj_new = ("tesla" in o_n) and ("honda" not in o_n) and ("civic" not in o_n)
        out["triples"].append({
            "raw": r, "ok": True, "norm": [s_n, p_n, o_n],
            "subject_ok": subj_ok, "predicate_family": fam,
            "predicate_ok": fam is not None, "new_vehicle_object": obj_new,
        })
        if subj_ok:
            out["subject_user_triples"].append(r)
        if subj_ok and fam and obj_new and out["matched"] is None:
            out["matched"] = r
            out["matched_norm"] = {"subject": s_n, "predicate": p_n,
                                   "family": fam, "object": o_n}
    return out


def score_queries(obj):
    out = {"queries_is_list": False, "queries": [], "non_string": 0,
           "any_hit": False}
    if not isinstance(obj, dict):
        return out
    qs = obj.get("queries")
    if not isinstance(qs, list):
        return out
    out["queries_is_list"] = True
    for q in qs:
        if not isinstance(q, str):
            out["non_string"] += 1
            continue
        m = HOOK_RE.search(q)
        out["queries"].append({"q": q, "hit": bool(m),
                               "hit_term": m.group(1).lower() if m else None})
    out["any_hit"] = any(x["hit"] for x in out["queries"])
    return out


def score_facts(obj):
    """Angle-27-style facts recall on Slot 1's "facts" list (context only)."""
    out = {"facts_tp": 0, "facts_fp": 0, "matched": [], "unmatched": [],
           "coverage": {}}
    facts = obj.get("facts") if isinstance(obj, dict) else None
    if not isinstance(facts, list):
        return out
    seen = set()
    for s in facts:
        if not isinstance(s, str):
            continue
        hits = match_concepts(s)
        if not hits:
            out["facts_fp"] += 1
            out["unmatched"].append(s)
            continue
        out["matched"].append(s)
        for h in hits:
            out["coverage"].setdefault(h, []).append(s)
            seen.add(h)
    out["facts_tp"] = len(seen)
    return out


def host_closure(matched_triple):
    """Deterministic mock graph update (pure Python, no GPU)."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    edges = [dict(SEED_EDGE)]
    invalidated = []
    for e in edges:
        if (_norm(e["subject"]) in SUBJECTS_OK
                and e["predicate"].upper() in {"DRIVES", "DRIVING", "OWNS"}
                and ("honda" in e["object"].casefold()
                     or "civic" in e["object"].casefold())
                and e["invalid_at"] is None):
            e["invalid_at"] = now
            invalidated.append(e["id"])
    new_edge = {"id": "e1", "subject": matched_triple[0],
                "predicate": matched_triple[1].upper(),
                "object": matched_triple[2],
                "valid_from": now, "invalid_at": None}
    edges.append(new_edge)
    honda_closed = any(e["id"] == "e0" and e["invalid_at"] for e in edges)
    inserted = any(e["id"] == "e1" and e["valid_from"] == now for e in edges)
    return {"executed": True, "now": now, "invalidated_ids": invalidated,
            "inserted": new_edge, "edges_after": edges,
            "closure_ok": bool(honda_closed and inserted)}


def fence_strip_diag(text, direct_obj):
    """Always-on non-gating diagnostic (angle-34 style, per seed).

    Reparses the raw Slot 1 text after ``` fence stripping and rescores
    relations strict + lenient. Gate verdicts never use this.
    """
    fobj, ferr = fence_strip_json(text)
    d = {"strip_parse_ok": fobj is not None, "strip_parse_error": ferr}
    if fobj is None:
        d.update({"relations_strict": None, "relations_lenient": None,
                  "new_vehicle_strict": False, "new_vehicle_lenient": False,
                  "matched_lenient": None, "facts_recall": None})
        return d
    rel_s = score_relations(fobj)
    rel_l = score_relations(fobj, subjects=SUBJECTS_OK_LENIENT)
    d.update({
        "relations_strict": rel_s,
        "relations_lenient": rel_l,
        "new_vehicle_strict": rel_s["matched"] is not None,
        "new_vehicle_lenient": rel_l["matched"] is not None,
        "matched_lenient": rel_l["matched"],
        "facts_recall": score_facts(fobj),
        "same_as_direct": direct_obj is not None and fobj == direct_obj,
    })
    return d


def build_rep(seed, solo0, batch, walls):
    s0_solo = get_stream(solo0, S0_MARKER)
    s0_batch = get_stream(batch, S0_MARKER)
    s1 = get_stream(batch, S1_MARKER)
    s2 = get_stream(batch, S2_MARKER)

    # Bare parse FIRST (gate verdicts, comparability with angles 33/34).
    s1_obj, s1_err = direct_json(s1["text"])
    s2_obj, s2_err = direct_json(s2["text"])
    # robust_parse is a SECONDARY diagnostic only.
    s1_robust, s1_robust_err, s1_repair_applied = robust_parse(s1["text"])
    rel = score_relations(s1_obj)
    rel_len = score_relations(s1_obj, subjects=SUBJECTS_OK_LENIENT)
    qry = score_queries(s2_obj)
    facts = score_facts(s1_obj)
    new_vehicle = rel["matched"] is not None
    new_vehicle_len = rel_len["matched"] is not None

    rel_r = score_relations(s1_robust) if s1_robust is not None else None
    rel_r_len = (score_relations(s1_robust, subjects=SUBJECTS_OK_LENIENT)
                 if s1_robust is not None else None)
    robust_matched = rel_r["matched"] if rel_r else None
    robust_closure = (host_closure(robust_matched) if robust_matched else
                      {"executed": False, "closure_ok": False})
    robust_diag = {
        "parse_ok": s1_robust is not None,
        "error": s1_robust_err,
        "repair_applied": s1_repair_applied,
        "matches_bare": s1_robust == s1_obj,
        "new_vehicle_strict": (rel_r["matched"] is not None) if rel_r else None,
        "new_vehicle_lenient": (rel_r_len["matched"] is not None)
                                if rel_r_len else None,
        "matched_strict": robust_matched,
        "relations_count": (len(s1_robust["relations"])
                            if isinstance(s1_robust, dict)
                            and isinstance(s1_robust.get("relations"), list)
                            else None),
        "closure_ok_if_scored": bool(robust_closure["closure_ok"]),
    }

    # Always-on non-gating diagnostic: fence-stripped reparse + rescore.
    fdiag = fence_strip_diag(s1["text"], s1_obj)

    closure = (host_closure(rel["matched"]) if new_vehicle else
               {"executed": False, "edges_after": [dict(SEED_EDGE)],
                "closure_ok": False,
                "reason": "no new-vehicle triple; graph left at seed state"})
    return {
        "seed": seed,
        "slot_seeds": {"s0": seed, "s1": seed + 1, "s2": seed + 2},
        "solo_slot0": summarize_branch(s0_solo),
        "batch_slot0": summarize_branch(s0_batch),
        "batch_slot1": summarize_branch(s1),
        "batch_slot2": summarize_branch(s2),
        **s0_identity(s0_solo["text"], s0_batch["text"],
                      s0_solo["generated"], s0_batch["generated"]),
        "s0_prompt_identical": s0_solo["prompt"] == s0_batch["prompt"],
        "s1_direct_ok": s1_obj is not None,
        "s1_direct_error": s1_err,
        "s1_bare_parse": s1_obj is not None,
        "s1_repaired_parse": s1_robust is not None,
        "s1_repair_applied": s1_repair_applied,
        "s1_robust_error": s1_robust_err,
        "s1_robust_json": s1_robust,
        "robust_parse_diagnostic": robust_diag,
        "s1_diag": None if s1_obj is not None else diag_parse(s1["text"]),
        "s1_json": s1_obj,
        "diagnostic_fence_strip": fdiag,
        "s1_fence_strip_ok": fdiag["strip_parse_ok"],
        "s2_direct_ok": s2_obj is not None,
        "s2_direct_error": s2_err,
        "s2_diag": None if s2_obj is not None else diag_parse(s2["text"]),
        "s2_json": s2_obj,
        "relations": rel,
        "relations_lenient": rel_len,
        "queries": qry,
        "facts_recall": facts,
        "facts_empty": (isinstance(s1_obj, dict)
                        and isinstance(s1_obj.get("facts"), list)
                        and len(s1_obj["facts"]) == 0),
        "relations_empty": (isinstance(s1_obj, dict)
                            and isinstance(s1_obj.get("relations"), list)
                            and len(s1_obj["relations"]) == 0),
        "relations_count": (len(s1_obj["relations"])
                            if isinstance(s1_obj, dict)
                            and isinstance(s1_obj.get("relations"), list)
                            else None),
        "queries_count": (len(qry["queries"]) if qry["queries_is_list"]
                          else None),
        "new_vehicle_present": new_vehicle,
        "new_vehicle_lenient": new_vehicle_len,
        "matched_lenient_direct": rel_len["matched"],
        "retrieval_hit": qry["any_hit"],
        "host_closure": closure,
        "fences": {"s1": "```" in s1["text"], "s2": "```" in s2["text"]},
        "comments": {"s1": has_comment(s1["text"]),
                     "s2": has_comment(s2["text"])},
        "gpu_wall_s": walls,
        "ratios": {
            "completion": s0_batch["completion_ms"] / s0_solo["completion_ms"],
            "ttft": s0_batch["ttft_ms"] / s0_solo["ttft_ms"],
        },
    }


def angle_matrix(this_run):
    """Side-by-side: angle 33 (no tail) / 34 (tail+rules) / 35 (tail only)."""
    m = {}

    def load(path):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    a = load(A33_JSON)
    if a:
        gc = a.get("gate_counts", {})
        m["angle33_no_tail"] = {
            "source": A33_JSON.name,
            "patches_vs_angle33": "none (baseline)",
            "fence_seeds": len(gc.get("fence_seeds", [])),
            "s1_bare_parse": gc.get("s1_direct_parse"),
            "s1_robust_parse": None,
            "strict_relation": gc.get("new_vehicle_relation"),
            "max_tokens": a.get("max_tokens"),
        }
    else:
        m["angle33_no_tail"] = {"available": False}

    a = load(A34_JSON)
    if a:
        gc = a.get("gate_counts", {})
        rp = (a.get("robust_parse") or {}).get("counts") or {}
        m["angle34_tail_plus_rules"] = {
            "source": A34_JSON.name,
            "patches_vs_angle33": ("proximal anti-fence tail + rules 5-6 "
                                   "(subject User, max 3 relations) + "
                                   "max_tokens 256"),
            "fence_seeds": len(gc.get("fence_seeds", [])),
            "s1_bare_parse": gc.get("s1_direct_parse"),
            "s1_robust_parse": rp.get("repaired_parse"),
            "strict_relation": gc.get("new_vehicle_relation_strict"),
            "max_tokens": a.get("max_tokens"),
        }
    else:
        m["angle34_tail_plus_rules"] = {"available": False}

    m["angle35_tail_only"] = {
        "source": OUT_JSON.name,
        "patches_vs_angle33": ("proximal anti-fence tail ONLY (rules 5-6 "
                               "absent; max_tokens 160 = angle-33 value)"),
        "fence_seeds": this_run["fence_seed_count"],
        "s1_bare_parse": this_run["s1_bare_parse"],
        "s1_robust_parse": this_run["s1_robust_parse"],
        "strict_relation": this_run["strict_relation"],
        "max_tokens": this_run["max_tokens"],
    }
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rescore", action="store_true",
                    help="re-analyze saved raw CLI JSON without re-running the GPU")
    args = ap.parse_args()
    seeds = SEEDS[: args.reps]

    batch_rows = [("s0", S0_PROMPT), ("s1", S1_PROMPT), ("s2", S2_PROMPT)]
    solo_rows = [("s0", S0_PROMPT)]
    print(f"[derive] s0 prompt loaded from {SRC_S0} and asserted "
          f"byte-identical; dialogue turns={len(DIALOGUE_TURNS.splitlines())}",
          flush=True)
    print(f"[derive] s1 patch: ONLY the proximal anti-fence tail before "
          f"Output:; NO rules 5-6; max_tokens={TAIL_MAX_TOKENS} "
          f"(angle-33 value)", flush=True)
    if args.dry_run:
        print("plan:", seeds, "batch k=3 + solo serial Slot 0 per seed; "
              "s0 prompt bytes OK; prompts ready for", BATCH_TSV, SOLO_TSV)
        return 0

    write_tsv(ROOT / BATCH_TSV, batch_rows)
    write_tsv(ROOT / SOLO_TSV, solo_rows)
    print(f"[derive] wrote {BATCH_TSV} (3 rows) and {SOLO_TSV} (1 row)",
          flush=True)
    tsv_report = verify_tsvs(A33_S1_PROMPT)
    print(f"[derive] TSV verify OK: solo vs 33/34="
          f"{tsv_report['solo_byte_identical_vs_angle33']}/"
          f"{tsv_report['solo_byte_identical_vs_angle34']} "
          f"s0/s2 rows vs 33="
          f"{tsv_report['batch_s0_row_identical_vs_angle33']}/"
          f"{tsv_report['batch_s2_row_identical_vs_angle33']} "
          f"vs 34="
          f"{tsv_report['batch_s0_row_identical_vs_angle34']}/"
          f"{tsv_report['batch_s2_row_identical_vs_angle34']} "
          f"s1 tail-line-only diff="
          f"{tsv_report['s1_diff_tail_line_only']}", flush=True)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    prev_walls = {}
    if args.rescore and OUT_JSON.exists():
        try:
            prev = json.loads(OUT_JSON.read_text(encoding="utf-8"))
            prev_walls = {w["seed"]: w for w in prev.get("gpu_wall", {}).get("per_rep", [])}
        except (ValueError, KeyError):
            prev_walls = {}
    reps = []
    walls_total = 0.0
    walls_log = []
    for seed in seeds:
        if args.rescore:
            solo0 = load_raw(RAW_DIR / f"rep{seed}_solo.json")
            batch = load_raw(RAW_DIR / f"rep{seed}_batch.json")
            pw = prev_walls.get(seed)
            walls = ({"solo_s": pw["solo_s"], "batch_s": pw["batch_s"],
                      "total_s": pw["total_s"], "preserved_from_prior_run": True}
                     if pw else None)
            if walls:
                walls_total += walls["total_s"]
                walls_log.append({"seed": seed, **walls})
        else:
            t0 = time.monotonic()
            print(f"[run] seed={seed} solo (serial, Slot 0)", flush=True)
            solo0 = run_cli("serial", SOLO_TSV, seed,
                            RAW_DIR / f"rep{seed}_solo.json")
            w_solo = time.monotonic() - t0
            t0 = time.monotonic()
            print(f"[run] seed={seed} batch k=3 (voice + scribe + indexer)",
                  flush=True)
            batch = run_cli("batch", BATCH_TSV, seed,
                            RAW_DIR / f"rep{seed}_batch.json")
            w_batch = time.monotonic() - t0
            walls = {"solo_s": round(w_solo, 3), "batch_s": round(w_batch, 3),
                     "total_s": round(w_solo + w_batch, 3)}
            walls_total += w_solo + w_batch
            walls_log.append({"seed": seed, **walls})
        rep = build_rep(seed, solo0, batch, walls)
        reps.append(rep)
        r, c = rep["ratios"], rep["host_closure"]
        print(f"[rep {seed}] s1_ok={rep['s1_direct_ok']} "
              f"robust={rep['s1_repaired_parse']} "
              f"repaired={rep['s1_repair_applied']} "
              f"fstrip={rep['s1_fence_strip_ok']} "
              f"s2_ok={rep['s2_direct_ok']} rel={rep['new_vehicle_present']} "
              f"rel_len={rep['new_vehicle_lenient']} "
              f"hook={rep['retrieval_hit']} "
              f"clos={c['closure_ok'] if c['executed'] else '-'} "
              f"comp_ratio={r['completion']:.3f} "
              f"s1_tok={rep['batch_slot1']['generated']} "
              f"s2_tok={rep['batch_slot2']['generated']} "
              f"s0_identical={rep['s0_identical']} "
              f"fin=({rep['batch_slot1']['finish']},{rep['batch_slot2']['finish']})",
              flush=True)

    n = len(reps)
    comp_ratios = [r["ratios"]["completion"] for r in reps]
    ttft_ratios = [r["ratios"]["ttft"] for r in reps]
    med_comp, mean_comp = med(comp_ratios), statistics.mean(comp_ratios)
    med_ttft, mean_ttft = med(ttft_ratios), statistics.mean(ttft_ratios)

    s1_direct = sum(r["s1_direct_ok"] for r in reps)
    s1_robust_ok = sum(r["s1_repaired_parse"] for r in reps)
    repair_seeds = [r["seed"] for r in reps if r["s1_repair_applied"]]
    robust_nv_strict = sum(1 for r in reps
                           if r["robust_parse_diagnostic"]["new_vehicle_strict"])
    robust_nv_lenient = sum(1 for r in reps
                            if r["robust_parse_diagnostic"]["new_vehicle_lenient"])
    s2_direct = sum(r["s2_direct_ok"] for r in reps)
    gate2 = sum(r["new_vehicle_present"] for r in reps)
    gate2_len = sum(r["new_vehicle_lenient"] for r in reps)
    gate3 = sum(r["retrieval_hit"] for r in reps)
    g2_reps = [r for r in reps if r["new_vehicle_present"]]
    closure_all = bool(g2_reps) and all(r["host_closure"]["closure_ok"]
                                        for r in g2_reps)
    fstrip_ok = sum(r["s1_fence_strip_ok"] for r in reps)
    fstrip_nv_strict = sum(r["diagnostic_fence_strip"]["new_vehicle_strict"]
                           for r in reps)
    fstrip_nv_len = sum(r["diagnostic_fence_strip"]["new_vehicle_lenient"]
                        for r in reps)
    fence_seeds = [r["seed"] for r in reps
                   if r["fences"]["s1"] or r["fences"]["s2"]]
    comment_seeds = [r["seed"] for r in reps
                     if r["comments"]["s1"] or r["comments"]["s2"]]
    facts_tp = sum(r["facts_recall"]["facts_tp"] for r in reps)
    facts_fp = sum(r["facts_recall"]["facts_fp"] for r in reps)
    ident_ok = sum(r["s0_identical"] for r in reps)
    facts_empty_seeds = [r["seed"] for r in reps if r["facts_empty"]]
    rels_empty_seeds = [r["seed"] for r in reps if r["relations_empty"]]
    over3_rel_seeds = [r["seed"] for r in reps
                       if isinstance(r["relations_count"], int)
                       and r["relations_count"] > 3]

    criteria = {
        "gate1_json_integrity_direct_5_of_5_both_slots":
            s1_direct == n and s2_direct == n,
        "gate2_new_vehicle_relation_ge_4_of_5": gate2 >= 4,
        "gate3_retrieval_hook_ge_4_of_5": gate3 >= 4,
        "gate4_host_closure_all_gate2_seeds": closure_all,
        "gate5_slot0_completion_median_le_1_12": med_comp <= 1.12,
    }
    verdict = "meets" if all(criteria.values()) else "miss"

    this_run = {
        "fence_seed_count": len(fence_seeds),
        "s1_bare_parse": s1_direct,
        "s1_robust_parse": s1_robust_ok,
        "strict_relation": gate2,
        "max_tokens": TAIL_MAX_TOKENS,
    }
    matrix = angle_matrix(this_run)

    summary = {
        "experiment": ("Angle 35 tail-only ablation of the angle-34 k=3 "
                       "pipeline: Slot 1 prompt = angle 33's VERBATIM + "
                       "ONLY the proximal anti-fence tail (no rules 5-6), "
                       "max_tokens 160; attribution of angle 34's "
                       "fence/strict-relation fixes to the tail alone"),
        "parent": ("results/ingame_k3_fix.json (angle 34, bundle) and "
                   "results/ingame_k3_pipeline.json (angle 33, baseline); "
                   "s0 prompt byte source data/ingame_mem_solo.tsv "
                   "(angle 27); protocol unchanged: fresh solo arm, "
                   "seeds 1234-1238, temp 0.2, top_p 0.9, max_tokens 160"),
        "patches_vs_angle33": [
            ("proximal anti-fence tail on its own line immediately before "
             f"'Output:' in the s1 row: {S1_TAIL!r} (byte-identical to "
             "angle 34's tail sentence and placement)"),
            ("NO rules 5-6 (Subject must be \"User\" / Maximum 3 "
             "relations) -- the co-confound under test; rules 1-4 and the "
             "Schema line stay angle-33 verbatim"),
            (f"max_tokens = {TAIL_MAX_TOKENS} (angle-33 value; angle 34's "
             "256 raise is NOT applied, keeping the patch count minimal)"),
        ],
        "model": MODEL, "system": SYSTEM, "seeds": seeds,
        "max_tokens": TAIL_MAX_TOKENS, "temperature": TEMP, "top_p": TOP_P,
        "n_reps": n,
        "metadata": cli_provenance(),
        "data_files": {"solo_slot0": SOLO_TSV, "batch": BATCH_TSV,
                       "note": ("s0/s2 rows and solo TSV byte-identical to "
                                "angles 33 AND 34; only the s1 row differs "
                                "from angle 33 (tail line only)")},
        "tsv_verification": tsv_report,
        "slot_seeds_note": ("parallel_cli.cpp:108: batch row i samples "
                            "seed+i; solo serial 1-row run samples seed+0, "
                            "so batch Slot 0 and solo Slot 0 share the seed"),
        "prompt_construction": {
            "s0": S0_PROMPT,
            "s1": S1_PROMPT,
            "s2": S2_PROMPT,
            "template_note": ("angle-33 template '<instr> Conversation:"
                              "\\n{turns}\\n\\nOutput:'; s1 inserts the "
                              "anti-fence tail on its own line between the "
                              "dialogue block and 'Output:' exactly as "
                              "angle 34 did, with NO rules 5-6 in the "
                              "instruction"),
        },
        "matchers": MATCHER_DOC,
        "seed_graph_edge": SEED_EDGE,
        "acceptance": criteria,
        "verdict": verdict,
        "gate_counts": {
            "s1_direct_parse": s1_direct, "s2_direct_parse": s2_direct,
            "new_vehicle_relation_strict": gate2,
            "new_vehicle_relation_lenient_diagnostic": gate2_len,
            "retrieval_hook": gate3,
            "host_closure_ok_on_gate2_seeds":
                sum(1 for r in g2_reps if r["host_closure"]["closure_ok"]),
            "gate2_seeds": [r["seed"] for r in g2_reps],
            "gate2_seeds_lenient": [r["seed"] for r in reps
                                    if r["new_vehicle_lenient"]],
            "s0_text_identical": ident_ok,
            "fence_seeds": fence_seeds, "comment_seeds": comment_seeds,
            "facts_empty_seeds": facts_empty_seeds,
            "relations_empty_seeds": rels_empty_seeds,
            "relations_over_3_seeds": over3_rel_seeds,
        },
        "comparison_matrix": {
            "note": ("same model/seeds/temp/matchers/gates across the three "
                     "angles; only the Slot 1 prompt differs (and angle "
                     "34's max_tokens raise)"),
            "angle33_no_tail": matrix.get("angle33_no_tail"),
            "angle34_tail_plus_rules": matrix.get("angle34_tail_plus_rules"),
            "angle35_tail_only": matrix.get("angle35_tail_only"),
        },
        "robust_parse_diagnostic": {
            "rule": ROBUST_PARSE_RULE,
            "note": ("SECONDARY diagnostic only (bare parse reported "
                     "first for comparability): bounded host-side brace "
                     "repair, at most ONE '}', no fence stripping -- a "
                     "fenced output fails robust_parse and is reported as "
                     "a fail; gate verdicts never use it"),
            "counts": {"bare_parse": s1_direct, "repaired_parse": s1_robust_ok,
                       "of_n": n},
            "repair_applied_seeds": repair_seeds,
            "per_seed": [{"seed": r["seed"],
                          "bare_parse": r["s1_bare_parse"],
                          "repaired_parse": r["s1_repaired_parse"],
                          "repair_applied": r["s1_repair_applied"],
                          "new_vehicle_strict":
                              r["robust_parse_diagnostic"]["new_vehicle_strict"],
                          "new_vehicle_lenient":
                              r["robust_parse_diagnostic"]["new_vehicle_lenient"],
                          "relations_count":
                              r["robust_parse_diagnostic"]["relations_count"]}
                         for r in reps],
            "new_vehicle_strict_total": robust_nv_strict,
            "new_vehicle_lenient_total": robust_nv_lenient,
        },
        "facts_recall_totals": {
            "facts_tp": facts_tp, "facts_fp": facts_fp,
            "facts_possible": len(FACTS_LABEL) * n,
            "concept_hits": {
                name: sum(1 for r in reps
                          if name in r["facts_recall"]["coverage"])
                for name in FACTS_LABEL},
        },
        "diagnostic_fence_strip": {
            "note": ("NON-GATING: fence-stripped reparse of Slot 1, computed "
                     "for EVERY seed (angle-34 style); isolates format "
                     "failure from semantics. Gate verdicts use "
                     "direct-parse strict scores only."),
            "strip_parse_ok": fstrip_ok,
            "new_vehicle_strict_subject": fstrip_nv_strict,
            "new_vehicle_lenient_subject_i_me": fstrip_nv_len,
            "per_rep": {str(r["seed"]): r["diagnostic_fence_strip"]
                        for r in reps},
        },
        "s1_tokens": [r["batch_slot1"]["generated"] for r in reps],
        "s2_tokens": [r["batch_slot2"]["generated"] for r in reps],
        "s0_batch_tokens": [r["batch_slot0"]["generated"] for r in reps],
        "s1_finish": [r["batch_slot1"]["finish"] for r in reps],
        "s2_finish": [r["batch_slot2"]["finish"] for r in reps],
        "latency": {
            "completion_ratio_median": med_comp,
            "completion_ratio_mean": mean_comp,
            "ttft_ratio_median": med_ttft, "ttft_ratio_mean": mean_ttft,
            "solo0_completion_ms_median":
                med([r["solo_slot0"]["completion_ms"] for r in reps]),
            "batch_slot0_completion_ms_median":
                med([r["batch_slot0"]["completion_ms"] for r in reps]),
            "ratio_of_completion_medians": (
                med([r["batch_slot0"]["completion_ms"] for r in reps])
                / med([r["solo_slot0"]["completion_ms"] for r in reps])),
            "per_rep_completion_ratios": comp_ratios,
            "per_rep_ttft_ratios": ttft_ratios,
        },
        "gpu_wall": {"total_s": round(walls_total, 3) if walls_log else None,
                     "per_rep": walls_log,
                     "note": ("wall time of the 10 parallel_cli invocations "
                              "(5 serial + 5 batch), includes model load "
                              "each time; under --rescore, preserved from "
                              "the prior run's JSON when present")},
        "s0_prompt_identical_all_reps": all(r["s0_prompt_identical"]
                                             for r in reps),
        "reps": reps,
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                        encoding="utf-8")

    L = []
    L.append("Angle 35 - tail-only ablation of angle 34 (k=3 pipeline): "
             "parallel::Runtime")
    L.append(f"model={MODEL}  max_tokens={TAIL_MAX_TOKENS} temp={TEMP} "
             f"top_p={TOP_P} system={SYSTEM!r}")
    L.append(f"cli_sha256={summary['metadata']['cli_sha256']}  "
             f"built={summary['metadata']['cli_built_mtime']}  "
             f"tree={summary['metadata']['git_tree']}")
    L.append(f"reps={n} seeds={seeds}  solo=serial({SOLO_TSV})  "
             f"batch=k=3({BATCH_TSV}; rows sample seed, seed+1, seed+2)")
    L.append("Patches vs angle 33 (Slot 1 row only; s0/s2 rows and solo TSV "
             "verified byte-identical to angles 33 AND 34):")
    for p in summary["patches_vs_angle33"]:
        L.append(f"  - {p}")
    L.append(f"TSV verify: solo byte-identical vs 33/34="
             f"{tsv_report['solo_byte_identical_vs_angle33']}/"
             f"{tsv_report['solo_byte_identical_vs_angle34']}, batch s0/s2 "
             f"rows identical vs 33="
             f"{tsv_report['batch_s0_row_identical_vs_angle33']}/"
             f"{tsv_report['batch_s2_row_identical_vs_angle33']} vs 34="
             f"{tsv_report['batch_s0_row_identical_vs_angle34']}/"
             f"{tsv_report['batch_s2_row_identical_vs_angle34']}, s1 row "
             f"differs (tail line only)="
             f"{tsv_report['s1_diff_tail_line_only']}")
    L.append("")
    L.append("MATCHERS (byte-equal to angle 34; raws keep every "
             "triple/query for rescoring):")
    for k, v in MATCHER_DOC.items():
        L.append(f"  {k}: {v}")
    L.append("")
    L.append(f"{'seed':>6} | {'solo_ms':>9} {'batch_ms':>10} {'comp_r':>7} "
             f"{'ttft_r':>7} {'same':>5} | {'s1ok':>4} {'fstrp':>5} {'s2ok':>4} "
             f"{'rel':>3} {'len':>3} {'hook':>4} {'clos':>4} | {'s1tok':>5} "
             f"{'s2tok':>5} {'s0b':>4} {'rels':>4} | {'fin1':>11} {'fin2':>10}")
    for r in reps:
        c = r["host_closure"]
        L.append(f"{r['seed']:>6} | {r['solo_slot0']['completion_ms']:>9.1f} "
                 f"{r['batch_slot0']['completion_ms']:>10.1f} "
                 f"{r['ratios']['completion']:>7.3f} "
                 f"{r['ratios']['ttft']:>7.3f} {str(r['s0_identical']):>5} | "
                 f"{str(r['s1_direct_ok']):>4} "
                 f"{str(r['s1_fence_strip_ok']):>5} "
                 f"{str(r['s2_direct_ok']):>4} "
                 f"{str(r['new_vehicle_present']):>3} "
                 f"{str(r['new_vehicle_lenient']):>3} "
                 f"{str(r['retrieval_hit']):>4} "
                 f"{(str(c['closure_ok']) if c['executed'] else '-'):>4} | "
                 f"{r['batch_slot1']['generated']:>5} "
                 f"{r['batch_slot2']['generated']:>5} "
                 f"{r['batch_slot0']['generated']:>4} "
                 f"{(str(r['relations_count']) if r['relations_count'] is not None else '-'):>4} | "
                 f"{r['batch_slot1']['finish']:>11} "
                 f"{r['batch_slot2']['finish']:>10}")
    L.append(f"{'med/sum':>6} | "
             f"{summary['latency']['solo0_completion_ms_median']:>9.1f} "
             f"{summary['latency']['batch_slot0_completion_ms_median']:>10.1f} "
             f"{med_comp:>7.3f} {med_ttft:>7.3f} {ident_ok}/{n} | "
             f"{s1_direct:>4}/{n:<3} {fstrip_ok:>5}/{n:<3} "
             f"{s2_direct:>4}/{n:<3} {gate2:>3}/{n:<2} {gate2_len:>3}/{n:<2} "
             f"{gate3:>4}/{n:<2} "
             f"{sum(1 for r in g2_reps if r['host_closure']['closure_ok']):>4}/{len(g2_reps):<2} | "
             f"{med(summary['s1_tokens']):>5.0f} "
             f"{med(summary['s2_tokens']):>5.0f}")
    L.append("")
    for r in reps:
        L.append(f"--- seed {r['seed']} "
                 f"(slot seeds s0={r['slot_seeds']['s0']} "
                 f"s1={r['slot_seeds']['s1']} s2={r['slot_seeds']['s2']}) ---")
        if not r["s1_direct_ok"]:
            L.append(f"  s1 direct parse FAILED: {r['s1_direct_error']} "
                     f"(diag={r['s1_diag']})")
            L.append(f"  [robust-secondary] repaired parse ok="
                     f"{r['s1_repaired_parse']} repair_applied="
                     f"{r['s1_repair_applied']} error={r['s1_robust_error']}")
        d = r["diagnostic_fence_strip"]
        if d["strip_parse_ok"]:
            L.append(f"  [diag] fence-stripped parse OK "
                     f"(same_as_direct={d.get('same_as_direct')}); "
                     f"new_vehicle(strict)={d['new_vehicle_strict']} "
                     f"new_vehicle(lenient i/me)={d['new_vehicle_lenient']} "
                     f"facts tp/fp="
                     f"{d['facts_recall']['facts_tp']}/"
                     f"{d['facts_recall']['facts_fp']}")
            for t in d["relations_strict"]["triples"]:
                L.append(f"  [diag] rel: {t}")
            if d.get("matched_lenient"):
                L.append(f"  [diag] lenient match: {d['matched_lenient']}")
        else:
            L.append(f"  [diag] fence-stripped parse FAILED: "
                     f"{d['strip_parse_error']}")
        if not r["s2_direct_ok"]:
            L.append(f"  s2 direct parse FAILED: {r['s2_direct_error']} "
                     f"(diag={r['s2_diag']})")
        fr = r["facts_recall"]
        L.append(f"  facts recall tp/fp={fr['facts_tp']}/{fr['facts_fp']} "
                 f"empty={r['facts_empty']} matched={fr['matched']} "
                 f"unmatched={fr['unmatched']}")
        for t in r["relations"]["triples"]:
            L.append(f"  rel: {t}")
        if r["relations"]["matched"]:
            L.append(f"  NEW-VEHICLE TRIPLE (strict, direct parse): "
                     f"{r['relations']['matched']}")
        elif r["new_vehicle_lenient"]:
            L.append(f"  new-vehicle only under LENIENT subject set: "
                     f"{r['matched_lenient_direct']} (diagnostic, "
                     f"non-gating)")
        for q in r["queries"]["queries"]:
            L.append(f"  q: hit={q['hit']} term={q['hit_term']} {q['q']!r}")
        if not r["queries"]["queries_is_list"]:
            L.append("  queries key missing or not a list")
        c = r["host_closure"]
        if c["executed"]:
            L.append(f"  host closure executed at {c['now']}: "
                     f"invalidated={c['invalidated_ids']} "
                     f"inserted={c['inserted']} closure_ok={c['closure_ok']}")
        else:
            L.append("  host closure NOT executed (no new-vehicle triple)")
        if r["fences"]["s1"] or r["fences"]["s2"] or r["comments"]["s1"] \
                or r["comments"]["s2"]:
            L.append(f"  FENCE/COMMENT: fences={r['fences']} "
                     f"comments={r['comments']}")
        L.append(f"  s1 raw: {r['batch_slot1']['text']!r}")
        L.append(f"  s2 raw: {r['batch_slot2']['text']!r}")
    L.append("")
    L.append(f"Gate counts: s1 direct parse {s1_direct}/{n}, s2 direct "
             f"parse {s2_direct}/{n}; new-vehicle relation (strict) {gate2}/{n} "
             f"(seeds {[r['seed'] for r in reps if r['new_vehicle_present']]}); "
             f"new-vehicle lenient (diag) {gate2_len}/{n}; "
             f"retrieval hook {gate3}/{n} "
             f"(seeds {[r['seed'] for r in reps if r['retrieval_hit']]})")
    L.append(f"Host closure: executed+ok on "
             f"{sum(1 for r in g2_reps if r['host_closure']['closure_ok'])}/"
             f"{len(g2_reps)} gate-2 seeds; skipped on seeds "
             f"{[r['seed'] for r in reps if not r['new_vehicle_present']] or 'none'}")
    L.append(f"Fences {len(fence_seeds)}/5 seeds {fence_seeds}; comments "
             f"{len(comment_seeds)}/5 seeds {comment_seeds} (expect 0)")
    L.append(f"Facts recall (angle-27 label, facts list only): TP={facts_tp} "
             f"FP={facts_fp} of {len(FACTS_LABEL) * n} possible; "
             f"concept hits="
             f"{summary['facts_recall_totals']['concept_hits']}")
    L.append(f"Empty lists: facts {facts_empty_seeds or 'none'}; "
             f"relations {rels_empty_seeds or 'none'}; relations count per "
             f"rep={[r['relations_count'] for r in reps]} (NO max-3 rule in "
             f"this prompt; over-3 seeds {over3_rel_seeds or 'none'}); "
             f"queries count per rep={[r['queries_count'] for r in reps]}")
    L.append(f"DIAGNOSTIC (non-gating) fence-stripped Slot 1, all seeds: "
             f"parse ok {fstrip_ok}/{n}; new-vehicle strict-subject "
             f"{fstrip_nv_strict}/{n}; new-vehicle lenient (adds 'i'/'me') "
             f"{fstrip_nv_len}/{n}")
    L.append(f"ROBUST PARSE (SECONDARY diagnostic; bare parse above is "
             f"reported first for comparability): rule: {ROBUST_PARSE_RULE}")
    L.append(f"  bare {s1_direct}/{n} vs robust {s1_robust_ok}/{n}; repair "
             f"applied on seeds {repair_seeds or 'none'}; strict relation "
             f"on repaired JSON {robust_nv_strict}/{n} "
             f"(lenient {robust_nv_lenient}/{n})")
    L.append(f"Slot 1 tokens per rep: {summary['s1_tokens']} "
             f"finish={summary['s1_finish']}")
    L.append(f"Slot 2 tokens per rep: {summary['s2_tokens']} "
             f"finish={summary['s2_finish']}")
    L.append(f"Slot 0 completion ratio batch(k=3)/solo: median={med_comp:.3f} "
             f"mean={mean_comp:.3f} (solo med "
             f"{summary['latency']['solo0_completion_ms_median']:.1f} ms, "
             f"batch med "
             f"{summary['latency']['batch_slot0_completion_ms_median']:.1f} ms, "
             f"ratio of medians "
             f"{summary['latency']['ratio_of_completion_medians']:.3f})")
    L.append(f"Slot 0 ttft ratio batch/solo: median={med_ttft:.3f} "
             f"mean={mean_ttft:.3f}")
    L.append(f"Slot 0 text identical solo vs batch (same seed): "
             f"{ident_ok}/{n} (identity caveat: non-identical seeds differ "
             f"by sampling under batched-GEMM numerics, not by the s1 "
             f"prompt patch)")
    if walls_log:
        L.append(f"GPU wall time (10 CLI invocations, incl. model loads): "
                 f"{walls_total:.1f} s total; per rep "
                 f"{[w['total_s'] for w in walls_log]}")
    L.append("")
    L.append("COMPARISON MATRIX (identical seeds/temp/matchers/gates; only "
             "the Slot 1 prompt differs):")
    L.append(f"  {'angle':<26} | {'fences':>7} | {'bare parse':>11} | "
             f"{'strict relation':>15} | {'max_tok':>7}")

    def x5(v):
        return f"{v}/5" if isinstance(v, int) else str(v)

    for key, label in (("angle33_no_tail", "33 no tail (baseline)"),
                       ("angle34_tail_plus_rules", "34 tail+rules (bundle)"),
                       ("angle35_tail_only", "35 tail only (this run)")):
        row = matrix.get(key) or {}
        if row.get("available") is False:
            L.append(f"  {label:<26} | {'n/a':>7} | {'n/a':>11} | "
                     f"{'n/a':>15} | {'n/a':>7}")
            continue
        bare = row.get("s1_bare_parse")
        robust = row.get("s1_robust_parse")
        bare_s = x5(bare)
        if isinstance(robust, int) and robust != bare:
            bare_s += f" (robust {robust}/5)"
        L.append(f"  {label:<26} | {x5(row.get('fence_seeds')):>7} | "
                 f"{bare_s:>11} | {x5(row.get('strict_relation')):>15} | "
                 f"{row.get('max_tokens', '?'):>7}")
    L.append("ATTRIBUTION: fences stayed 0/5 without rules 5-6? strict "
             "relation held without the subject/bloat rules? see the matrix "
             "rows above (35 vs 33 isolates the tail; 35 vs 34 isolates "
             "rules 5-6 + max_tokens).")
    L.append("ACCEPTANCE: " + ", ".join(f"{k}={'PASS' if v else 'FAIL'}"
                                        for k, v in criteria.items()))
    L.append(f"VERDICT: {verdict}")
    text = "\n".join(L) + "\n"
    OUT_TXT.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
