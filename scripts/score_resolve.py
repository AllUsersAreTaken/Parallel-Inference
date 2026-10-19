"""Score messy-request resolution: single stream vs N-way ensemble.

Parses each candidate JSON, validates it against the context, and scores
intended event / resolved datetime / action / clarification / unsupported
message ids. The ensemble takes the majority (intended_event, datetime, action)
tuple; a weak majority triggers clarification instead of a guess.
"""
import csv
import json
import re
import statistics
import sys

CAND = sys.argv[1] if len(sys.argv) > 1 else "results/resolve_3b.tsv"
TRUTH = sys.argv[2] if len(sys.argv) > 2 else "data/resolve_truth.jsonl"
N = 8

cands = {}
with open(CAND, encoding="utf-8") as f:
    for row in csv.reader(f, delimiter="\t"):
        if len(row) >= 3:
            cands.setdefault(int(row[0]), {})[int(row[1])] = row[2]

truth = {}
with open(TRUTH, encoding="utf-8") as f:
    for line in f:
        o = json.loads(line)
        truth[o["id"]] = o["truth"]


def parse(text):
    # try every '{' position, taking the slice to the last '}' (handles a stray
    # leading brace from the unconstrained seed token)
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        sub = text[i:]
        j = sub.rfind("}")
        if j <= 0:
            continue
        cand = sub[:j + 1]
        for c in (cand, cand.replace("'", '"').replace(",}", "}").replace(",]", "]")):
            try:
                o = json.loads(c)
                if isinstance(o, dict):
                    return o
            except Exception:
                pass
    return None


def norm_dt(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


ids = sorted(i for i in cands if i in truth)
valid_ids = {i: set(range(1, 6)) for i in ids}


def score_one(o, t):
    if o is None:
        return {"schema": 0, "event": 0, "dt": 0, "action": 0, "clar": 0, "unsup": 0}
    ev = str(o.get("intended_event", "")).lower()
    dt = norm_dt(o.get("resolved_date_time", ""))
    ac = str(o.get("requested_action", "")).lower()
    sm = o.get("supporting_message_ids", [])
    mi = o.get("missing_information", [])
    nc = bool(o.get("needs_clarification", False))
    unsup = 0
    if isinstance(sm, list):
        unsup = sum(1 for x in sm if not isinstance(x, int) or x not in valid_ids_by_id.get(cur_id, set()))
    want_dt = norm_dt(t["resolved_date_time"])
    clar = 1 if ((nc or (isinstance(mi, list) and len(mi) > 0)) == t["needs_clarification"]) else 0
    REQ = ["intended_event", "resolved_date_time", "requested_action", "supporting_message_ids", "missing_information", "needs_clarification"]
    if not all(k in o for k in REQ):
        return {"schema": 0, "event": 0, "dt": 0, "action": 0, "clar": 0, "unsup": 0}
    return {
        "schema": 1,
        "event": 1 if "alex rivera" in ev else 0,
        "dt": 1 if want_dt in dt else 0,
        "action": 1 if "adapter" in ac else 0,
        "clar": clar,
        "unsup": unsup,
    }


valid_ids_by_id = {}
cur_id = 0
res = {}
for i in ids:
    valid_ids_by_id[i] = valid_ids[i]
    parsed = [parse(cands[i].get(j, "")) for j in range(N)]
    sc = [score_one(p, truth[i]) for p in parsed]

    def agg(key, subset):
        return 100 * sum(s[key] for s in subset) / len(subset) if subset else 0.0

    single = sc[:1]
    alls = sc

    # ensemble: majority tuple among valid parses
    tuples = []
    for p in parsed:
        if p is not None:
            tuples.append((str(p.get("intended_event", "")).lower()[:40],
                           norm_dt(p.get("resolved_date_time", "")),
                           str(p.get("requested_action", "")).lower()[:40]))
    ens = {"schema": 1 if any(p is not None for p in parsed) else 0, "event": 0, "dt": 0, "action": 0, "clar": 0, "unsup": 0}
    if tuples:
        best, bc = max(((x, tuples.count(x)) for x in set(tuples)), key=lambda z: z[1])
        if bc >= 3:
            ens["event"] = 1 if "alex rivera" in best[0] else 0
            ens["dt"] = 1 if norm_dt(truth[i]["resolved_date_time"]) in best[1] else 0
            ens["action"] = 1 if "adapter" in best[2] else 0
            ens["clar"] = 1 if not truth[i]["needs_clarification"] else 0
        else:
            ens["clar"] = 1 if truth[i]["needs_clarification"] else 0   # weak majority -> clarify
    res[i] = (single[0], ens, alls)

print(f"scenarios={len(ids)}")
print(f"{'metric':>16} {'single':>8} {'ensemble':>9} {'streams':>8}")
for key, name in [("schema", "schema valid"), ("event", "intended event"), ("dt", "resolved dt"),
                  ("action", "action"), ("clar", "clarification")]:
    s = 100 * sum(res[i][0][key] for i in ids) / len(ids)
    e = 100 * sum(res[i][1][key] for i in ids) / len(ids)
    a = 100 * sum(sum(x[key] for x in res[i][2]) / N for i in ids) / len(ids)
    print(f"{name:>16} {s:>7.0f}% {e:>8.0f}% {a:>7.0f}%")
unsup_s = sum(res[i][0]["unsup"] for i in ids)
unsup_e = sum(res[i][1]["unsup"] for i in ids)
print(f"{'unsupported ids':>16} {unsup_s:>8} {unsup_e:>9}")

# breakdown by family
for fam, pred in [("ambiguous", lambda i: truth[i]["needs_clarification"]), ("straightforward", lambda i: not truth[i]["needs_clarification"])]:
    sub = [i for i in ids if pred(i)]
    if not sub:
        continue
    e = 100 * sum(res[i][1]["event"] and res[i][1]["dt"] and res[i][1]["action"] for i in sub) / len(sub)
    s = 100 * sum(res[i][0]["event"] and res[i][0]["dt"] and res[i][0]["action"] for i in sub) / len(sub)
    print(f"  {fam:>15} (n={len(sub)}): single full-correct {s:.0f}%  ensemble {e:.0f}%")
