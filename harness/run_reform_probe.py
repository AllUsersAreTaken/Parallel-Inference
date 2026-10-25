#!/usr/bin/env python3
"""Reformulation-intervention probe (row 39; feature for row 38's blind spot).

Hypothesis: agreement stability under a meaning-preserving intervention is a
second uncertainty feature that can flag unanimous-wrong runs (the agreement
mechanism's blind spot - see disagree_probe100_n5: unanimous 45/47, the 2
misses are ids 28/29).

Design:
  * 30 problems stratified from results/disagree_probe100_n5.json:
    all 2 unanimous-wrong + 14 unanimous-right + 14 split (evenly spaced
    within stratum, deterministic, no within-stratum cherry-picking).
  * 5 arms x 3 diagnostic streams = 15 rows/problem (one parallel_cli
    invocation per problem; seeds 1234+row, same sampling chain as row 38):
      CTRL  original prompt          (noise floor: no intervention)
      REN   person names renamed     (mechanical, stoplist+position gated)
      REO   first two premises swapped (gated on >=3 sentences, no temporal
            markers, query sentence kept last)
      REP   superficial phrasing swaps (how many/old, in total, total->sum,
            buy family, per->each, get->receive; quantity-safe regex only)
      COM   all applicable transforms combined
  * Every transform gated on number-multiset preservation (loud 'fail').
    No-op transforms still run: they act as extra CTRL replicates.
  * arm_stable := orig majority appears >=2/3 in arm; stability := mean over
    non-noop intervention arms. Flag = stability below threshold.

Usage: python harness/run_reform_probe.py [--preview]
Writes: results/reform_probe30.{json,csv}
"""
import argparse
import itertools
import json
import os
import re
import subprocess
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "harness"))
import analyze_modality_vote as M  # noqa: E402

CLI = os.path.join(ROOT, "app", "parallel_cli.exe")
BIN = os.path.join(ROOT, "build", "cuda-graphs", "bin")
SRC_JSON = os.path.join(ROOT, "results", "disagree_probe100_n5.json")
SCRATCH = os.path.join(ROOT, "data", "_reform_probe.tsv")
OUT = os.path.join(ROOT, "data", "_reform_probe.json")

MODEL = os.path.join(ROOT, "models", "Llama-3.2-3B-Instruct-Q4_K_M.gguf")
ARMS = ("CTRL", "REN", "REO", "REP", "COM")
INTERV = ("REN", "REO", "REP", "COM")
K = 3
MAX_TOK, TEMP, TOP_P, SEED = 512, 0.6, 0.95, 1234
INSTR = ("Solve the problem step by step. Then end with a line in the "
         "exact form: The answer is: <number>\n\nProblem: ")

STOP = {"If", "How", "What", "When", "Where", "The", "This", "That", "There",
        "He", "She", "They", "We", "You", "A", "An", "In", "On", "At", "By",
        "For", "With", "And", "But", "Or", "To", "Of", "Is", "Are", "Was",
        "Were", "Can", "Could", "Will", "Would", "Each", "Every", "Some",
        "After", "Before", "His", "Her", "Its", "It", "Be", "Do", "Does",
        "Did", "Many", "Much", "More", "Most", "Such", "Than", "Then",
        "These", "Those", "Us", "I", "My", "Our", "Your", "All", "Any",
        "Both", "Few", "Other", "Over", "Under", "Between", "During", "From",
        "Into", "Through", "About", "As", "Not", "No", "Yes", "Find",
        "Calculate", "Total", "Here", "Also", "Only", "Once", "Two"}
NUMW = {"One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight",
        "Nine", "Ten", "Eleven", "Twelve", "First", "Second", "Third",
        "Fourth", "Fifth", "Half", "Quarter", "Last", "Next", "Both"}
POOL = ["Maya", "Theo", "Priya", "Jonas", "Lena", "Omar", "Nina", "Felix"]


def esc(s):
    return (s.replace("\\", "\\\\").replace("\n", "\\n")
             .replace("\r", "\\r").replace("\t", "\\t"))


def nums(s):
    return Counter(x.replace(",", "") for x in
                   re.findall(r"\d[\d,]*(?:\.\d+)?", s))


def find_names(q):
    caps = list(re.finditer(r"\b[A-Z][a-z]{1,12}\b", q))
    counts = Counter(m.group(0) for m in caps)
    names = []
    for m in caps:
        w = m.group(0)
        if w in STOP or w in NUMW or w in names:
            continue
        poss = q[m.end():m.end() + 2] == "'s"
        pre = q[:m.start()].rstrip()
        at_start = pre == "" or pre[-1] in ".?!"
        if poss or counts[w] >= 2 or not at_start:
            names.append(w)
    return names


def t_rename(q):
    names = find_names(q)
    if not names:
        return q, "noop"
    out = q
    for i, w in enumerate(names):
        rep = POOL[i % len(POOL)]
        while rep in out and i < len(POOL) - 1:
            i += 1
            rep = POOL[i % len(POOL)]
        out = re.sub(r"\b" + w + r"\b", rep, out)
    return out, ("ok" if out != q else "noop")


def t_reorder(q):
    sents = [s for s in re.split(r"(?<=\.)\s+", q.strip()) if s]
    if len(sents) < 3:
        return q, "noop"
    if re.search(r"\b(then|after\b|next\b|afterwards|finally|before\b)", q,
                 re.I):
        return q, "temporal"
    sents[0], sents[1] = sents[1], sents[0]
    return " ".join(sents), "ok"


def t_rephrase(q):
    out, n = q, 0

    def sub(pat, rep):
        nonlocal out, n
        o = out
        out = re.sub(pat, rep, out)
        n += out != o

    sub(r"\bHow many\b", "What number of")
    sub(r"\bhow many\b", "what number of")
    sub(r"\bHow old\b", "What age")
    sub(r"\bhow old\b", "what age")
    sub(r"\bin total\b", "altogether")
    sub(r"\btotal(?! number)\b", "sum")
    sub(r"\bTotals?(?! number)\b", "Sum")
    sub(r"\bbuys\b", "purchases")
    sub(r"\bbuy\b", "purchase")
    sub(r"\bbuying\b", "purchasing")
    sub(r"\bbought\b", "purchased")
    if re.search(r"\bper (day|week|month|year|hour|minute)\b", out):
        sub(r"\bper (day|week|month|year|hour|minute)\b", r"each \1")
    else:
        sub(r"\beach (day|week|month|year|hour|minute)\b", r"per \1")
    sub(r"\bget\b", "receive")
    return out, ("ok" if n else "noop")


def build_arms(q):
    """Returns [(arm, prompt_text, status)]; status ok/noop/temporal/fail."""
    ref = nums(q)
    rn, s1 = t_rename(q)
    ro, s2 = t_reorder(q)
    rp, s3 = t_rephrase(q)
    co, s4 = q, "ok"
    for fn in (t_rename, t_reorder, t_rephrase):
        nxt, stt = fn(co)
        if stt == "ok":
            co = nxt
    if co == q:
        s4 = "noop"
    arms = [("CTRL", q, "ok"), ("REN", rn, s1), ("REO", ro, s2),
            ("REP", rp, s3), ("COM", co, s4)]
    out = []
    for a, t, s in arms:
        if nums(t) != ref and a != "CTRL":
            s = "fail"
        out.append((a, t, s))
    return out


def pick(ids, k):
    ids = sorted(ids)
    n = len(ids)
    return sorted({ids[int((i + 0.5) * n / k)] for i in range(k)})


def select_problems():
    d = json.load(open(SRC_JSON, encoding="utf-8"))
    recs = d["records"]
    uw = [r for r in recs if r["unanimous"] and not r["maj_ok"]]
    ur = [r for r in recs if r["unanimous"] and r["maj_ok"]]
    sp = [r for r in recs if not r["unanimous"]]
    ur_ids, sp_ids = pick([r["id"] for r in ur], 14), \
        pick([r["id"] for r in sp], 14)
    chosen = ([(r, "UW") for r in uw]
              + [(r, "UR") for r in ur if r["id"] in ur_ids]
              + [(r, "SP") for r in sp if r["id"] in sp_ids])
    chosen.sort(key=lambda t: t[0]["id"])
    return chosen


def chunks_for(arms, texts):
    """Split arms into invocations whose capacity fits parallel_cli's
    context check: sum over streams of (est_prompt_tokens + max_tokens)
    must stay <= 8192 (the harness refuses larger batches)."""
    out, cur, cur_cost = [], [], 0
    for arm, t in zip([a for a, _, _ in arms], texts):
        est = len(INSTR + t) // 3 + 60
        add = K * est + K * MAX_TOK
        if cur and (cur_cost + add) > 8192 * 0.95:
            out.append(cur)
            cur, cur_cost = [], 0
        cur.append(arm)
        cur_cost += add
    if cur:
        out.append(cur)
    return out


def run():
    selected = select_problems()
    print(f"selected {len(selected)} problems: "
          + ", ".join(f"{s}:{r['id']}" for r, s in selected))
    env = dict(os.environ)
    env["PATH"] = BIN + os.pathsep + env.get("PATH", "")
    all_recs = []
    for r, stratum in selected:
        arms = build_arms(r["q"])
        texts = [t for _, t, _ in arms]
        groups = chunks_for(arms, texts)
        answers_by_arm, walls = {}, 0.0
        for gi, grp in enumerate(groups):
            ai0 = [a for a, _, _ in arms].index(grp[0])
            with open(SCRATCH, "w", encoding="utf-8") as fh:
                for j, arm in enumerate(grp):
                    t = texts[ai0 + j]
                    for s in range(K):
                        fh.write(f"{j * K + s + 1}\t{esc(INSTR + t)}\n")
            cp = subprocess.run(
                [CLI, "--model", MODEL, "--data", SCRATCH, "--out", OUT,
                 "--mode", "batch", "--max-tokens", str(MAX_TOK),
                 "--temperature", str(TEMP), "--top-p", str(TOP_P),
                 "--seed", str(SEED + 20 * gi)],
                cwd=ROOT, env=env, capture_output=True, text=True,
                timeout=600)
            if cp.returncode != 0:
                raise SystemExit(f"parallel_cli failed ({cp.returncode}):\n"
                                 f"{cp.stderr[-800:]}")
            out = json.load(open(OUT, encoding="utf-8"))
            walls += out["wall_ms"]
            for j, arm in enumerate(grp):
                ans = []
                for s in range(K):
                    stx = out["streams"][j * K + s]
                    ans.append(M.canon(M.extract_cot(stx["text"])[0]))
                answers_by_arm[arm] = ans
        o_maj, o_gold = r["maj"], r["gold"]
        arm_data, stabs = {}, []
        for a, t, st in arms:
            ans = answers_by_arm[a]
            cnt = Counter(x for x in ans if x is not None)
            arm_maj = (max(cnt.items(), key=lambda kv: kv[1])[0]
                       if cnt else None)
            stable = (cnt.get(o_maj, 0) >= 2) if o_maj else False
            arm_data[a] = {"status": st, "answers": ans, "maj": arm_maj,
                           "stable": stable, "arm_ok": arm_maj == o_gold}
            if a in INTERV and st == "ok":
                stabs.append(stable)
        ctrl = arm_data["CTRL"]["stable"]
        stability = sum(stabs) / len(stabs) if stabs else None
        rec = {"id": r["id"], "stratum": stratum, "q": r["q"],
               "gold": o_gold, "orig_maj": o_maj, "maj_ok": r["maj_ok"],
               "orig_unanimous": r["unanimous"],
               "orig_agree": r["pairwise_agree"],
               "ctrl_stable": ctrl, "stability": stability,
               "n_interv": len(stabs), "arms": arm_data,
               "wall_ms": out["wall_ms"]}
        all_recs.append(rec)
        print(f"p{r['id']:>3} {stratum} strab={stability} ctrl={int(ctrl)} "
              f"arms=" + " ".join(f"{a}:{arm_data[a]['status'][:4]}"
                                  f"{'*' if arm_data[a]['stable'] else '!'}"
                                  for a in ARMS))
    return all_recs


def auroc(pos, neg):
    if not pos or not neg:
        return float("nan")
    gt = sum(1.0 * (p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return gt / (len(pos) * len(neg))


def report(recs):
    print("\n=== stratum x stability ===")
    for st in ("UW", "UR", "SP"):
        g = [r for r in recs if r["stratum"] == st]
        if not g:
            continue
        valid = [r for r in g if r["stability"] is not None]
        mean = (sum(r["stability"] for r in valid) / len(valid)
                if valid else float("nan"))
        fl = {t: sum(1 for r in valid if r["stability"] <= t)
              for t in (1.0, 0.75, 0.5)}
        ctrl = sum(r["ctrl_stable"] for r in g)
        print(f"  {st}: n={len(g)} mean_stab={mean:.2f} "
              f"flag<1={fl[1.0]}/{len(valid)} <=0.75={fl[0.75]} "
              f"<=0.5={fl[0.5]} ctrl_stable={ctrl}/{len(g)}")
    ok = [r["stability"] for r in recs
          if r["stability"] is not None and r["maj_ok"]]
    bad = [r["stability"] for r in recs
           if r["stability"] is not None and not r["maj_ok"]]
    print(f"\nAUROC(stability -> maj_ok) = {auroc(ok, bad):.3f} "
          f"(n_pos={len(ok)}, n_neg={len(bad)})")
    for thr in (1.0, 0.75, 0.5):
        flagged = [r for r in recs if r["stability"] is not None
                   and r["stability"] <= thr]
        caught = [r for r in flagged if not r["maj_ok"]]
        prec = (100 * len(caught) / len(flagged)) if flagged else float("nan")
        rec_ = 100 * len(caught) / len(bad) if bad else float("nan")
        print(f"flag stability<={thr}: {len(flagged)}/30 flagged, "
              f"caught {len(caught)}/{len(bad)} wrong "
              f"(recall {rec_:.0f}%, precision {prec:.0f}%)")
    # per-arm contribution + blind-spot detail
    print("\n=== per-arm disagreement rates (status==ok arms) ===")
    for a in INTERV:
        g = [r for r in recs if r["arms"][a]["status"] == "ok"]
        mis = sum(1 for r in g if not r["arms"][a]["stable"])
        flip = sum(1 for r in g if r["arms"][a]["arm_ok"] and not r["maj_ok"])
        print(f"  {a}: {len(g)} ok arms, mismatch {mis}, "
              f"wrong->gold flips {flip}")
    for r in recs:
        if r["stratum"] == "UW":
            print(f"\nBLIND SPOT p{r['id']}: gold={r['gold']} "
                  f"orig_maj={r['orig_maj']} stab={r['stability']}")
            for a in ARMS:
                ad = r["arms"][a]
                print(f"  {a} [{ad['status']}] ans={ad['answers']} "
                      f"maj={ad['maj']} stable={ad['stable']} "
                      f"gold_hit={ad['arm_ok']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true",
                    help="build+verify transforms only, no GPU")
    args = ap.parse_args()
    if args.preview:
        selected = select_problems()
        for r, stratum in selected:
            arms = build_arms(r["q"])
            print(f"\n== p{r['id']} [{stratum}] gold={r['gold']} "
                  f"maj={r['maj']} ==")
            for a, t, st in arms:
                diff = "SAME" if t == r["q"] else "CHANGED"
                print(f"  {a} [{st}/{diff}]: {t[:150]}")
        return
    recs = run()
    report(recs)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    jpath = os.path.join(ROOT, "results", "reform_probe30.json")
    json.dump({"src": os.path.basename(SRC_JSON), "k": K, "arms": list(ARMS),
               "temp": TEMP, "top_p": TOP_P, "seed": SEED,
               "max_tokens": MAX_TOK, "records": recs},
              open(jpath, "w", encoding="utf-8"), indent=1)
    cols = ["id", "stratum", "gold", "orig_maj", "maj_ok",
            "orig_unanimous", "orig_agree", "ctrl_stable", "stability",
            "n_interv", "wall_ms"]
    cpath = os.path.join(ROOT, "results", "reform_probe30.csv")
    with open(cpath, "w", encoding="utf-8", newline="") as fh:
        fh.write(",".join(cols) + "\n")
        for r in recs:
            fh.write(",".join(str(r[c]) for c in cols) + "\n")
    print(f"\nwrote {jpath}\nwrote {cpath}")


if __name__ == "__main__":
    main()
