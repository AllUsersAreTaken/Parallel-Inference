#!/usr/bin/env python3
r"""Playground packaging of the frozen k=3 edge-memory engine (rows 33-35).

Slot 0 = voice, Slot 1 = scribe (frozen angle-34 recipe), Slot 2 = reverse-HyDE
indexer, host = robust_parse (frozen one-brace rule) + bi-temporal edge store.
Reads the frozen prompt texts from data/ingame_k3_fix_batch.tsv so nothing drifts.

Usage:
  python app\memory_engine.py --turn "I sold the Honda and bought a Tesla."
  python app\memory_engine.py --demo          # frozen 3-turn vehicle dialogue
  python app\memory_engine.py --reset         # clear state
  python app\memory_engine.py --show          # print current graph/index
"""
import argparse
import datetime
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_TSV = os.path.join(ROOT, "data", "ingame_k3_fix_batch.tsv")
CLI = os.path.join(ROOT, "app", "parallel_cli.exe")
MODEL = os.path.join(ROOT, "models", "Llama-3.2-3B-Instruct-Q4_K_M.gguf")
BIN = os.path.join(ROOT, "build", "cuda-graphs", "bin")
SCRATCH = os.path.join(ROOT, "app", "_mem_tmp.tsv")
DEFAULT_STATE = os.path.join(ROOT, "app", "memory_state.json")

DEMO_TURNS = [
    "I've been hiking with my dog Buster, but his knee is acting up, so we should stick to the flat trail by the river.",
    "Also, I sold my Honda last week and got a Tesla, so I need charging stations along my commute instead of gas stations.",
]

# Dated amendment (2026-10-21, user directive): Rule 5 of the frozen scribe
# instruction is relaxed from strict-User subjects to a scoped entity rule so
# entity-to-entity triples (e.g. [pi4, RUNS, Plex]) are expressible without
# reopening the verb-in-subject failure that Rule 5 was added to stop.
RULE5_OLD = '5. Subject must be "User" in every relation.'
RULE5_NEW = ('5. Subject must be "User" OR a concrete object/hardware entity '
             '(e.g., pi4, car). Never use verbs, actions, or pronouns '
             '(I, it, they) as the Subject.')

# Dated amendment (2026-10-21, user directive): Slot 1 delta-scoping. Under
# continuous batching the batch wall is bounded by the slowest stream; the
# scribe was re-extracting the whole transcript every turn (growing laundry
# list -> growing straggler). Scope it to the latest user message only.
# Tail-only placement: head insertions were measured not to bind and
# correlated with a rise in shifted [verb, entity, x] triples; the final
# sentence before Output: is the load-bearing zone (angle 34).
S1_DELTA_RULE = (" Extract ONLY from the latest user message; do not repeat "
                 "facts or relations mentioned in earlier turns. Relations are "
                 "[Subject, PREDICATE, Object] in exactly that order, with an "
                 "UPPERCASE PREDICATE.")

# Object strings sometimes carry predicate residue from the model
# (["User", "WANTS", "USE FOR homelab/server"]); strip these leading phrases
# when followed by whitespace or end-of-string.
LEADING_PHRASES = ("use for", "used for", "using for", "to be", "running")


def clean_entity(s):
    """Strip articles and predicate-bleed prefixes from an extracted entity.

    'a pi4' -> 'pi4' (so re-extraction dedupes), 'USE FOR homelab/server' ->
    'homelab/server'. Remainder keeps its case (all keying/comparison is
    lowercase). Returns '' if nothing meaningful is left.
    """
    out = (s or "").strip()
    for _ in range(3):
        before = out
        out = re.sub(r"^(a|an|the)\s+", "", out, flags=re.I)
        for ph in LEADING_PHRASES:
            out = re.sub(r"^" + re.escape(ph) + r"(?=\s|$)\s*", "", out, flags=re.I)
        if out == before:
            break
    return out


def load_frozen_prompts():
    """Split the frozen angle-34 rows into (instruction, dialogue-template).

    The TSV stores literal backslash-n sequences (the CLI unescapes them),
    so we split on the two-character sequence, then work with real newlines.
    """
    s0_in, s1_in, s2_in = None, None, None
    with open(TEMPLATE_TSV, encoding="utf-8") as f:
        for line in f:
            rid, prompt = line.rstrip("\n").split("\t", 1)
            prompt = prompt.replace("\\n", "\n")  # unescape like the CLI does
            head, dialogue = prompt.split("Conversation:\n", 1)
            body, tail = dialogue.split("\n\n", 1)
            if rid == "s0":
                s0_in = tail  # reply instruction
            elif rid == "s1":
                s1_in = head + "Conversation:\n{D}\n\n" + tail.split("Output:")[0] + "Output:"
                if RULE5_OLD not in s1_in:
                    raise SystemExit(
                        "frozen Rule 5 text not found in s1 prompt - "
                        "update RULE5_OLD/NEW or restore the recorded TSV")
                s1_in = s1_in.replace(RULE5_OLD, RULE5_NEW)
                # Delta scoping lives ONLY in the proximal tail (angle 34:
                # tails bind; head insertions were measured not to bind and
                # correlated with a rise in shifted [verb, entity, x] triples).
                if "Output:" not in s1_in.split("Conversation:\n{D}\n\n", 1)[1]:
                    raise SystemExit("s1 proximal tail lost its Output: marker")
                s1_in = s1_in.replace("\nOutput:", S1_DELTA_RULE + "\nOutput:", 1)
            elif rid == "s2":
                s2_in = head + "Conversation:\n{D}\n\n" + tail
        if not all([s0_in, s1_in, s2_in]):
            raise SystemExit("could not read frozen prompts from " + TEMPLATE_TSV)
    return s0_in, s1_in, s2_in


_PROMPTS = None


def get_prompts():
    global _PROMPTS
    if _PROMPTS is None:
        _PROMPTS = load_frozen_prompts()
    return _PROMPTS


def render_dialogue(history):
    lines = ["Conversation:"]
    for role, text in history:
        lines.append(f"{role.capitalize()}: {text}")
    return "\n".join(lines)


def robust_parse(raw):
    """Frozen row-34 rule: append one '}' iff stripped raw ends with ']]'."""
    cleaned = raw.strip()
    if cleaned.endswith("]]") and not cleaned.endswith("}"):
        cleaned += "}"
        repaired = True
    else:
        repaired = False
    return json.loads(cleaned), repaired


def recover_parse(raw):
    """Fallback when json.loads fails: take the first complete JSON object.

    Seen failure: Slot 2 emitting a stray trailing quote after the closing
    brace (\"...diet\"]}\") -> 'Extra data'. Only used on the failure path,
    after the primary parse (incl. the frozen brace rule) already failed, so
    no recorded-gate behavior changes; callers log the recovery.
    """
    dec = json.JSONDecoder()
    text = raw.strip()
    i = text.find("{")
    if i < 0:
        raise ValueError("no JSON object in output")
    obj, end = dec.raw_decode(text[i:])
    return obj, text[i + end:].strip()


def run_k3(dialogue_block, s0_in, s1_in, s2_in):
    # parallel_cli reads one physical line per row and unescapes literal \n.
    def esc(s):
        return s.replace("\n", "\\n").replace("\t", " ")
    with open(SCRATCH, "w", encoding="utf-8", newline="\n") as f:
        f.write("id\tprompt\n")
        f.write("1\t" + esc(dialogue_block + "\n\n" + s0_in) + "\n")
        f.write("2\t" + esc(s1_in.replace("{D}", dialogue_block)) + "\n")
        f.write("3\t" + esc(s2_in.replace("{D}", dialogue_block)) + "\n")
    out = SCRATCH.replace(".tsv", ".json")
    env = dict(os.environ)
    env["PATH"] = BIN + os.pathsep + env.get("PATH", "")
    r = subprocess.run([CLI, "--model", MODEL, "--data", SCRATCH, "--out", out,
                        "--mode", "batch", "--max-tokens", "256",
                        "--temperature", "0.2", "--top-p", "0.9", "--seed", "1234"],
                       env=env, capture_output=True, text=True, timeout=300)
    if r.returncode != 0 or not os.path.exists(out):
        raise SystemExit(f"parallel_cli failed ({r.returncode}):\n{r.stderr[-800:]}")
    return json.load(open(out, encoding="utf-8"))


# Predicates where (subject, predicate) is functional: at most one open value
# at a time (a new value closes the old one). Everything else is set-valued:
# keyed by (subject, predicate, object), members coexist until individually
# superseded. Unknown predicates default to set-valued - a missed close is
# recoverable, a false close silently drops a fact.
SINGLE_VALUED_PREDICATES = {
    "drives", "lives_in", "employed_by", "current_location", "marital_status",
    "located_in",
}


def apply_graph(state, triples, now):
    """Bi-temporal closure with functional/set predicate routing.

    Functional predicates (SINGLE_VALUED_PREDICATES): keyed (subject,predicate);
    a new object closes the open version and appends the new one. Within one
    turn the first value wins and later different values are reported as
    conflicts (the model sometimes echoes the stale entity).

    Set-valued predicates (default): keyed (subject,predicate,object); members
    coexist - emitting [HAS, butter] and [HAS, flour] in one turn keeps both,
    and a later [HAS, cookies] never closes butter.
    """
    events = []
    conflicts = []
    edges = state.setdefault("edges", {})
    seen = {}  # key -> lowercased object seen this turn
    for s, p, o in triples:
        subj = clean_entity(s)
        obj = clean_entity(o)
        pred_raw = p.strip()
        # Schema (rule 2) says uppercase predicates. A non-uppercase predicate
        # means the triple is shifted (observed: ["Owns", "cheddar", ...] put
        # the verb in the subject slot and an entity in the predicate slot);
        # keying it would insert garbage nodes like owns|cheddar.
        if not (pred_raw and pred_raw == pred_raw.upper() and pred_raw != pred_raw.lower()):
            conflicts.append(
                f"dropped misaligned triple {[s, p, o]!r}: predicate {p!r} not uppercase")
            continue
        pred = pred_raw.lower()
        if not subj or not obj:
            conflicts.append(f"dropped junk triple {[s, p, o]!r} after entity cleaning")
            continue
        single = pred in SINGLE_VALUED_PREDICATES
        sl, ol = subj.lower(), obj.lower()
        key = f"{sl}|{pred}" if single else f"{sl}|{pred}|{ol}"
        if key in seen:
            if single and seen[key] != ol:
                conflicts.append(f"{key}: kept earlier value, dropped {obj!r}")
            continue
        seen[key] = ol
        versions = edges.setdefault(key, [])
        open_v = next((v for v in versions if v["invalid_at"] is None), None)
        if open_v and open_v["object"].strip().lower() == ol:
            continue
        if open_v:
            open_v["invalid_at"] = now
            events.append(("CLOSE", open_v["object"]))
        versions.append({"object": obj, "valid_from": now, "invalid_at": None})
        events.append(("ADD", obj))
    return events, conflicts


def normalize_state(state):
    """Upgrade older state files (single dict per key) to version lists, in place."""
    for k, v in list(state.get("edges", {}).items()):
        if isinstance(v, dict):
            state["edges"][k] = [v]
    return state


def run_turn(state, text):
    """Run one user turn through the k=3 chain; mutates state, returns details."""
    normalize_state(state)
    state.setdefault("history", []).append(["user", text])
    dialogue = render_dialogue(state["history"])
    s0_in, s1_in, s2_in = get_prompts()
    res = run_k3(dialogue, s0_in, s1_in, s2_in)
    streams = res["streams"]
    now = datetime.datetime.now().isoformat(timespec="seconds")
    out = {
        "turn": text,
        "reply": streams[0]["text"].strip(),
        "facts": [],
        "events": [],
        "conflicts": [],
        "repaired": False,
        "s1_error": None,
        "s1_raw": "",
        "s1_recovered": None,
        "s2_error": None,
        "s2_recovered": None,
        "queries": [],
        "wall_ms": float(res.get("wall_ms", 0)),
        "slot_ms": [float(s.get("completion_ms", 0)) for s in streams[:3]],
    }
    try:
        s1, rep1 = robust_parse(streams[1]["text"])
        out["s1_recovered"] = None
    except Exception as primary:
        try:
            s1, rest = recover_parse(streams[1]["text"])
            rep1 = False
            out["s1_recovered"] = rest or str(primary)
        except Exception:
            out["s1_error"] = str(primary)
            out["s1_raw"] = streams[1]["text"][:200]
            s1 = None
    if s1 is not None:
        out["facts"] = list(s1.get("facts", []))
        out["repaired"] = rep1
        events, conflicts = apply_graph(state, [tuple(t) for t in s1.get("relations", [])], now)
        out["events"] = events
        out["conflicts"] = conflicts
    try:
        s2, _ = robust_parse(streams[2]["text"])
        out["s2_recovered"] = None
        for q in s2.get("queries", []):
            state["queries"].append({"q": q, "at": now})
            out["queries"].append(q)
    except Exception as primary:
        try:
            s2, rest = recover_parse(streams[2]["text"])
            out["s2_recovered"] = rest or "trailing junk after JSON"
            for q in s2.get("queries", []):
                state["queries"].append({"q": q, "at": now})
                out["queries"].append(q)
        except Exception:
            out["s2_error"] = str(primary)
    state["assistant"] = out["reply"]
    state["history"].append(["assistant", out["reply"]])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turn", help="append this user turn and run k=3")
    ap.add_argument("--demo", action="store_true", help="run the frozen vehicle dialogue")
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--state", default=DEFAULT_STATE)
    args = ap.parse_args()

    if args.reset:
        if os.path.exists(args.state):
            os.remove(args.state)
        print("state cleared:", args.state)
        return

    state = {"history": [], "edges": {}, "queries": []}
    if os.path.exists(args.state):
        state = normalize_state(json.load(open(args.state, encoding="utf-8")))

    if args.show or not (args.turn or args.demo):
        print(json.dumps(state, indent=2))
        if not (args.turn or args.demo):
            return

    turns = DEMO_TURNS if args.demo else [args.turn]
    for text in turns:
        r = run_turn(state, text)
        print(f"\n--- turn: {text[:70]}{'...' if len(text) > 70 else ''} ---")
        print(f"Slot 0 reply:\n{r['reply']}\n")
        if r["s1_error"]:
            print(f"  WARNING: Slot 1 unparseable ({r['s1_error']}); state NOT updated")
            print(f"  raw: {r['s1_raw']!r}")
        else:
            for fact in r["facts"]:
                print(f"  fact: {fact}")
            for kind, obj in r["events"]:
                print(f"  graph {kind}: {obj}")
            for c in r["conflicts"]:
                print(f"  graph CONFLICT (dropped): {c}")
            if r["repaired"]:
                print("  (robust_parse repaired a missing brace)")
            if r.get("s1_recovered"):
                print(f"  (Slot 1 recovered by first-JSON fallback: {r['s1_recovered']!r})")
        if r["s2_error"]:
            print(f"  WARNING: Slot 2 unparseable ({r['s2_error']})")
        elif r.get("s2_recovered"):
            print(f"  (Slot 2 recovered by first-JSON fallback: {r['s2_recovered']!r})")
        for q in r["queries"]:
            print(f"  indexed query: {q}")
        json.dump(state, open(args.state, "w", encoding="utf-8"), indent=2)
        print(f"  wall: {r['wall_ms']:.0f} ms")

    print("\nopen edges:")
    for k, versions in state["edges"].items():
        for e in versions:
            if e["invalid_at"] is None:
                print(f"  {k} -> {e['object']}  (from {e['valid_from']})")


if __name__ == "__main__":
    main()
