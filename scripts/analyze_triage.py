#!/usr/bin/env python3
"""Analyze the dual-stream + referee triage run.

Ground truth:
  A/B correct  = the row's keywords (any hit) or answer appear in the text.
  A/B relation = both correct -> AGREE; exactly one -> CONFLICT; neither ->
                 AGREE iff the normalized texts are identical, else CONFLICT.
  C            = the grammar-locked AGREE/CONFLICT verdict.

The user's five-box matrix is A-centric. This script reports it explicitly and
also the plain NLI confusion (C vs the A/B relation), because the two differ
when both streams are wrong in the same way.
"""
import csv
import os
import re
import statistics as st
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "results", "triage_bench.csv")
OUT = os.path.join(ROOT, "results", "triage_summary.txt")
CATS = ("head", "ambiguous", "tail")


def fnum(r, k):
    try:
        return float(r.get(k, ""))
    except (TypeError, ValueError):
        return None


def inum(r, k):
    v = fnum(r, k)
    return int(v) if v is not None else None


def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else float("nan")


def ratio(num, den):
    return (num / den) if den else float("nan")


def norm(t):
    return re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip()


def correct(text, keywords, answer=""):
    t = (text or "").lower()
    ks = [k.strip().lower() for k in (keywords or "").split("|") if k.strip()]
    if not ks and answer:
        ks = [answer.strip().lower()]
    return any(k in t for k in ks)


def relation(ac, bc, a, b):
    if ac and bc:
        return "AGREE"
    if ac or bc:
        return "CONFLICT"
    return "AGREE" if norm(a) == norm(b) and norm(a) else "CONFLICT"


def main():
    rows = list(csv.DictReader(open(CSV, newline="", encoding="utf-8")))
    n = len(rows)
    assert n > 0
    lines = []
    w = lines.append

    for r in rows:
        r["_a"] = r.get("resp_a_pair", "")
        r["_b"] = r.get("resp_b_pair", "")
        r["_ab"] = r.get("resp_a_base", "")
        r["_ac"] = correct(r["_a"], r.get("keywords", ""), r.get("answer", ""))
        r["_bc"] = correct(r["_b"], r.get("keywords", ""), r.get("answer", ""))
        r["_rel"] = relation(r["_ac"], r["_bc"], r["_a"], r["_b"])
        r["_c"] = r.get("c_verdict", "")
        r["_same"] = bool(norm(r["_a"]) and norm(r["_a"]) == norm(r["_b"]))
        r["_base_same"] = r["_ab"] == r["_a"]
        r["_ab_same"] = r["_a"] == r["_b"]

    def count(pred, rs=None):
        return sum(1 for r in (rs if rs is not None else rows) if pred(r))

    def pct(k, d):
        return (100.0 * k / d) if d else float("nan")

    w("DUAL-STREAM + REFEREE TRIAGE")
    w(f"N={n}  " + "  ".join(f"{c}={count(lambda r, c=c: r['category'] == c)}" for c in CATS))
    w("")

    # correctness
    ac = count(lambda r: r["_ac"])
    bc = count(lambda r: r["_bc"])
    either = count(lambda r: r["_ac"] or r["_bc"])
    both = count(lambda r: r["_ac"] and r["_bc"])
    neither = count(lambda r: not r["_ac"] and not r["_bc"])
    shared = count(lambda r: not r["_ac"] and not r["_bc"] and r["_same"])
    identical = count(lambda r: r["_ab_same"])
    base_same = count(lambda r: r["_base_same"])
    w("FACTUAL / DIVERGENCE")
    w(f"  A correct {ac}/{n} ({pct(ac,n):.1f}%)   B correct {bc}/{n} ({pct(bc,n):.1f}%)")
    w(f"  either correct {either}/{n} ({pct(either,n):.1f}%)   both correct {both}/{n} ({pct(both,n):.1f}%)")
    w(f"  neither correct {neither}/{n} ({pct(neither,n):.1f}%)   identical wrong text (shared hallucination) {shared}/{n} ({pct(shared,n):.1f}%)")
    w(f"  A/B text identical {identical}/{n} ({pct(identical,n):.1f}%)   A_base==A_pair {base_same}/{n} ({pct(base_same,n):.1f}%)")
    w("")

    # NLI confusion
    rel_agree = count(lambda r: r["_rel"] == "AGREE")
    rel_conf = count(lambda r: r["_rel"] == "CONFLICT")
    rel_agree_c_agree = count(lambda r: r["_rel"] == "AGREE" and r["_c"] == "AGREE")
    rel_agree_c_conf = count(lambda r: r["_rel"] == "AGREE" and r["_c"] == "CONFLICT")
    rel_conf_c_conf = count(lambda r: r["_rel"] == "CONFLICT" and r["_c"] == "CONFLICT")
    rel_conf_c_agree = count(lambda r: r["_rel"] == "CONFLICT" and r["_c"] == "AGREE")
    nli_correct = rel_agree_c_agree + rel_conf_c_conf
    w("NLI CONFUSION (C verdict vs actual A/B relation)")
    w(f"  relation AGREE {rel_agree}: C AGREE {rel_agree_c_agree}, C CONFLICT {rel_agree_c_conf}")
    w(f"  relation CONFLICT {rel_conf}: C CONFLICT {rel_conf_c_conf}, C AGREE {rel_conf_c_agree}")
    w(f"  NLI accuracy {nli_correct}/{n} ({pct(nli_correct,n):.1f}%)   "
      f"conflict recall {pct(rel_conf_c_conf, rel_conf):.1f}%   "
      f"agree specificity {pct(rel_agree_c_agree, rel_agree):.1f}%")
    w("")

    # user matrix (A-centric)
    tp = count(lambda r: not r["_ac"] and r["_rel"] == "CONFLICT" and r["_c"] == "CONFLICT")
    fp = count(lambda r: r["_ac"] and r["_bc"] and r["_c"] == "CONFLICT")
    tn = count(lambda r: r["_ac"] and r["_bc"] and r["_c"] == "AGREE")
    fn1 = count(lambda r: not r["_ac"] and not r["_bc"] and r["_same"] and r["_c"] == "AGREE")
    fn2 = count(lambda r: r["_rel"] == "CONFLICT" and r["_c"] == "AGREE")
    catch = count(lambda r: not r["_ac"] and r["_c"] == "CONFLICT")
    miss = count(lambda r: not r["_ac"] and r["_c"] == "AGREE")
    fa = count(lambda r: r["_ac"] and r["_c"] == "CONFLICT")
    quiet = count(lambda r: r["_ac"] and r["_c"] == "AGREE")
    rescue = count(lambda r: not r["_ac"] and r["_bc"] and r["_c"] == "CONFLICT")
    b_error = count(lambda r: r["_ac"] and not r["_bc"] and r["_c"] == "CONFLICT")
    shared_conf = count(lambda r: not r["_ac"] and not r["_bc"] and r["_same"] and r["_c"] == "CONFLICT")
    w("USER MATRIX (A-centric)")
    w(f"  TP (A wrong, relation CONFLICT, C CONFLICT) {tp}/{n}")
    w(f"  FP (both correct, C CONFLICT; paraphrase trap) {fp}/{n}")
    w(f"  TN (both correct, C AGREE) {tn}/{n}")
    w(f"  FN1 (both wrong identical, C AGREE; shared hallucination) {fn1}/{n}")
    w(f"  FN2 (relation CONFLICT, C AGREE; referee blindness) {fn2}/{n}")
    w(f"  Also: A-wrong catches {catch}/{n-ac} ({pct(catch, n-ac):.1f}% of A-wrong); "
      f"A-wrong misses {miss}; A-correct false alarms {fa}/{ac} ({pct(fa, ac):.1f}% of A-correct); "
      f"A-correct quiet {quiet}")
    w(f"  Actionable: A wrong & B correct & C CONFLICT {rescue}; A correct & B wrong & C CONFLICT {b_error}; "
      f"shared wrong & C CONFLICT {shared_conf}")
    w("")

    # per category
    w("PER CATEGORY")
    w(f"{'cat':<10}{'n':>4}{'A ok':>6}{'B ok':>6}{'both':>6}{'same':>6}"
      f"{'C agr':>7}{'C con':>7}{'C=exp':>7}{'NLI':>7}{'trap':>6}{'shared':>7}"
      f"{'blind':>6}{'catch':>6}{'miss':>6}{'FA':>5}")
    for cat in CATS:
        rs = [r for r in rows if r["category"] == cat]
        m = len(rs)
        if not m:
            continue
        a = count(lambda r: r["_ac"], rs)
        b = count(lambda r: r["_bc"], rs)
        bo = count(lambda r: r["_ac"] and r["_bc"], rs)
        sa = count(lambda r: r["_same"], rs)
        ca = count(lambda r: r["_c"] == "AGREE", rs)
        cc = count(lambda r: r["_c"] == "CONFLICT", rs)
        cexp = count(lambda r: r["_c"] == r["expected"], rs)
        nli = count(lambda r: (r["_rel"] == "AGREE" and r["_c"] == "AGREE")
                    or (r["_rel"] == "CONFLICT" and r["_c"] == "CONFLICT"), rs)
        trap = count(lambda r: r["_ac"] and r["_bc"] and r["_c"] == "CONFLICT", rs)
        sh = count(lambda r: not r["_ac"] and not r["_bc"] and r["_same"], rs)
        blind = count(lambda r: r["_rel"] == "CONFLICT" and r["_c"] == "AGREE", rs)
        cat_catch = count(lambda r: not r["_ac"] and r["_c"] == "CONFLICT", rs)
        cat_miss = count(lambda r: not r["_ac"] and r["_c"] == "AGREE", rs)
        cat_fa = count(lambda r: r["_ac"] and r["_c"] == "CONFLICT", rs)
        w(f"{cat:<10}{m:>4}{a:>6}{b:>6}{bo:>6}{sa:>6}{ca:>7}{cc:>7}{cexp:>7}"
          f"{pct(nli,m):>6.1f}%{trap:>6}{sh:>7}{blind:>6}{cat_catch:>6}{cat_miss:>6}{cat_fa:>5}")
    w("")

    # engine
    wb = [fnum(r, "wall_base_ms") for r in rows]
    wp = [fnum(r, "wall_pair_ms") for r in rows]
    wt = [fnum(r, "triage_total_ms") for r in rows]
    cm = [fnum(r, "c_ms") for r in rows]
    cp = [fnum(r, "c_prefill_ms") for r in rows]
    tab = [fnum(r, "ttft_a_base_ms") for r in rows]
    tap = [fnum(r, "ttft_a_pair_ms") for r in rows]
    tbp = [fnum(r, "ttft_b_pair_ms") for r in rows]
    pb = [fnum(r, "prefill_base_ms") for r in rows]
    pp = [fnum(r, "prefill_pair_ms") for r in rows]
    sb = [inum(r, "steps_base") for r in rows]
    sp = [inum(r, "steps_pair") for r in rows]
    js = [inum(r, "joint_steps") for r in rows]
    pbr = [ratio(x, y) for x, y in zip(wp, wb)]
    tbr = [ratio(x, y) for x, y in zip(wt, wb)]
    tpr = [ratio(x, y) for x, y in zip(wt, wp)]
    cpr = [ratio(x, y) for x, y in zip(cm, wp)]
    ttr = [ratio(x, y) for x, y in zip(tap, tab)]
    jr = [ratio(x, y) for x, y in zip(js, sp)]
    w("ENGINE (medians)")
    w(f"  wall ms: baseline {med(wb):.0f}, pair {med(wp):.0f}, triage {med(wt):.0f}; "
      f"pair/base {med(pbr):.3f}, triage/base {med(tbr):.3f}, triage/pair {med(tpr):.3f}")
    w(f"  C: wall {med(cm):.0f} ms, prefill {med(cp):.0f} ms, C/pair {med(cpr):.3f}; "
      f"C tokens median {med([inum(r,'c_tokens') for r in rows])}")
    w(f"  TTFT ms: A_base {med(tab):.2f}, A_pair {med(tap):.2f}, B_pair {med(tbp):.2f}; "
      f"A pair/base {med(ttr):.3f}")
    w(f"  prefill ms: base {med(pb):.1f}, pair {med(pp):.1f}; "
      f"steps base {med(sb):.0f}, pair {med(sp):.0f}; joint {med(js):.0f} ({med(jr)*100:.1f}% of pair steps)")
    w(f"  pair/base <=1.25: {sum(1 for x in pbr if x <= 1.25)}/{n}; "
      f"TTFT pair/base <=1.25: {sum(1 for x in ttr if x <= 1.25)}/{n}; "
      f"A_base==A_pair {base_same}/{n}; A/B identical {identical}/{n}")
    w(f"  C finishes: {dict(Counter(r.get('c_finish','') for r in rows))}")
    w("")

    # examples
    def show(title, pred, k=3):
        w(title)
        got = 0
        for r in rows:
            if not pred(r):
                continue
            w(f"  {r['id']} [{r['category']}] A_ok={int(r['_ac'])} B_ok={int(r['_bc'])} "
              f"rel={r['_rel']} C={r['_c']} copy={int(r['_ab_same'])}")
            w(f"    A: {r['_a'][:180].replace(chr(10),' ')}")
            w(f"    B: {r['_b'][:180].replace(chr(10),' ')}")
            got += 1
            if got >= k:
                break
        w("")

    show("TP: A wrong, relation CONFLICT, C CONFLICT",
         lambda r: not r["_ac"] and r["_rel"] == "CONFLICT" and r["_c"] == "CONFLICT")
    show("FP: both correct, C CONFLICT (paraphrase trap)",
         lambda r: r["_ac"] and r["_bc"] and r["_c"] == "CONFLICT")
    show("FN1: both wrong identical, C AGREE (shared hallucination)",
         lambda r: not r["_ac"] and not r["_bc"] and r["_same"] and r["_c"] == "AGREE")
    show("FN2: relation CONFLICT, C AGREE (referee blindness)",
         lambda r: r["_rel"] == "CONFLICT" and r["_c"] == "AGREE")
    show("A wrong, C AGREE (miss)",
         lambda r: not r["_ac"] and r["_c"] == "AGREE")

    text = "\n".join(lines) + "\n"
    open(OUT, "w", encoding="utf-8", newline="\n").write(text)
    print(text)


if __name__ == "__main__":
    main()
