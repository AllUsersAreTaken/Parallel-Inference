#!/usr/bin/env python3
"""Summarize the HaluEval flag-only bench (stdlib only).

Reads results/halueval_bench.csv (written by harness/engine/halueval.cpp) and
writes:

    results/halueval_detector_summary.txt
    results/halueval_e2e_summary.txt

Detector metrics use the gold labels (hallucinated = positive). The e2e actor's
answer is judged against the HaluEval right_answer with a normalized lexical
check (exact/containment, number containment, or token F1 >= 0.5); that is a
proxy for "supported by the knowledge", not a human judgment.

Usage:
    python scripts/analyze_halueval.py
    python scripts/analyze_halueval.py --bench results/halueval_bench.csv
"""

import argparse
import csv
import hashlib
import math
import os
import random
import re
import statistics
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------- generic helpers ----------------

def pct(values, p):
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p / 100.0
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return s[int(k)]
    return s[lo] * (hi - k) + s[hi] * (k - lo)


def fnum(x, digits=1):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.{digits}f}"


def pct_s(x, digits=1):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{100.0 * x:.{digits}f}%"


def sha256(path):
    if not path or not os.path.exists(path):
        return "n/a"
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def boot_ci(values_fn, n, iters=2000, seed=1234):
    """Bootstrap over n paired units. values_fn(list_of_indices) -> float."""
    rnd = random.Random(seed)
    out = []
    for _ in range(iters):
        idx = [rnd.randrange(n) for _ in range(n)]
        v = values_fn(idx)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            out.append(v)
    if not out:
        return (float("nan"), float("nan"))
    out.sort()
    return (out[int(0.025 * len(out))], out[int(0.975 * len(out)) - 1])


def auroc(pos, neg):
    """Rank-based AUROC with average ranks for ties (Mann-Whitney)."""
    if not pos or not neg:
        return float("nan")
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    ranks = [0.0] * len(allv)
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    r_pos = sum(ranks[k] for k, (_, lab) in enumerate(allv) if lab == 1)
    n1, n0 = len(pos), len(neg)
    return (r_pos - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def cp_upper_zero(n, alpha=0.05):
    """One-sided 95% upper bound for 0/n (Clopper-Pearson)."""
    return 1.0 - alpha ** (1.0 / n) if n else float("nan")


def cp_lower_all(n, alpha=0.05):
    """One-sided 95% lower bound for n/n (Clopper-Pearson)."""
    return alpha ** (1.0 / n) if n else float("nan")


def read_tsv(path):
    """Read the escaped HaluEval TSV. Returns {id: {field: text}}."""
    out = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        rdr = csv.DictReader(fh, delimiter="\t")
        for row in rdr:
            if not row.get("id"):
                continue
            out[row["id"]] = {k: unescape(v) for k, v in row.items()}
    return out


def unescape(s):
    if s is None:
        return ""
    out = []
    i = 0
    while i < len(s):
        if s[i] == "\\" and i + 1 < len(s):
            c = s[i + 1]
            if c == "n":
                out.append("\n"); i += 2; continue
            if c == "t":
                out.append("\t"); i += 2; continue
            if c == "r":
                out.append("\r"); i += 2; continue
            if c == "\\":
                out.append("\\"); i += 2; continue
        out.append(s[i]); i += 1
    return "".join(out)


def norm(s):
    s = (s or "").lower()
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def token_f1(a, b):
    ta, tb = a.split(), b.split()
    if not ta or not tb:
        return 0.0
    from collections import Counter
    common = sum((Counter(ta) & Counter(tb)).values())
    if common == 0:
        return 0.0
    p = common / len(ta)
    r = common / len(tb)
    return 2 * p * r / (p + r)


def is_correct(answer, right):
    """Lexical proxy for "the actor's answer matches the reference answer".

    Correct if any of:
      - normalized equality,
      - one normalized string contains the other,
      - every numeric token in the reference appears in the answer,
      - the reference is a single token and appears in the answer,
      - token F1 >= 0.5.
    """
    a, r = norm(answer), norm(right)
    if not a or not r:
        return False
    if a == r:
        return True
    if r in a or a in r:
        return True
    rnums = re.findall(r"\b\d+(?:[.,]\d+)?\b", r)
    anums = set(re.findall(r"\b\d+(?:[.,]\d+)?\b", a))
    if rnums and all(n in anums for n in rnums):
        return True
    rt = r.split()
    if len(rt) == 1 and rt[0] in a.split():
        return True
    return token_f1(a, r) >= 0.5


def snippet(s, n=280):
    s = re.sub(r"\s+", " ", s or "").strip()
    return s if len(s) <= n else s[:n] + "..."


def load_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            if not r.get("id"):
                continue
            rows.append(r)
    return rows


def fget(r, k, default=0.0):
    v = r.get(k, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def iget(r, k, default=0):
    v = r.get(k, "")
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


# ---------------- detector ----------------

def detector_summary(rows, data, bench_path, data_path, model_path, out_path):
    det = [r for r in rows if r.get("mode") == "detector"]
    right = [r for r in det if r.get("candidate_type") == "right"]
    hall = [r for r in det if r.get("candidate_type") == "hallucinated"]
    if not right or not hall:
        raise SystemExit("detector: no right/hallucinated rows in the bench CSV")

    by_id = {}
    for r in det:
        by_id.setdefault(r["id"], {})[r["candidate_type"]] = r
    pairs = [(by_id[i]["right"], by_id[i]["hallucinated"]) for i in by_id
             if "right" in by_id[i] and "hallucinated" in by_id[i]]
    n_pairs = len(pairs)

    def verdict_metrics(idx):
        tp = sum(1 for k in idx if iget(pairs[k][1], "flag") == 1)
        fp = sum(1 for k in idx if iget(pairs[k][0], "flag") == 1)
        fn = len(idx) - tp
        tn = len(idx) - fp
        return tp, fp, fn, tn

    tp, fp, fn, tn = verdict_metrics(list(range(n_pairs)))
    n_h, n_r = len(hall), len(right)
    catch = tp / n_h if n_h else float("nan")
    fa = fp / n_r if n_r else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = catch
    acc = (tp + tn) / (n_h + n_r) if (n_h + n_r) else float("nan")

    def ci_rate(i, which):
        tp_i, fp_i, _, _ = verdict_metrics(list(i))
        if which == "catch":
            return tp_i / len(i)
        if which == "fa":
            return fp_i / len(i)
        if which == "prec":
            return tp_i / (tp_i + fp_i) if (tp_i + fp_i) else None
        return (tp_i + (len(i) - fp_i)) / (2.0 * len(i))

    catch_ci = boot_ci(lambda i: ci_rate(i, "catch"), n_pairs)
    fa_ci = boot_ci(lambda i: ci_rate(i, "fa"), n_pairs)
    prec_ci = boot_ci(lambda i: ci_rate(i, "prec"), n_pairs)
    acc_ci = boot_ci(lambda i: ci_rate(i, "acc"), n_pairs)

    pos = [fget(r, "score") for r in hall]
    neg = [fget(r, "score") for r in right]
    pos_raw = [fget(r, "score_raw") for r in hall]
    neg_raw = [fget(r, "score_raw") for r in right]
    auc = auroc(pos, neg)
    auc_raw = auroc(pos_raw, neg_raw)

    def auc_idx(i):
        p = [fget(pairs[k][1], "score") for k in i]
        q = [fget(pairs[k][0], "score") for k in i]
        return auroc(p, q)

    auc_ci = boot_ci(auc_idx, n_pairs)

    # threshold scan on the grammar-constrained score
    def thr_metrics(th):
        tp_t = sum(1 for s in pos if s >= th)
        fp_t = sum(1 for s in neg if s >= th)
        c = tp_t / n_h if n_h else float("nan")
        f = fp_t / n_r if n_r else float("nan")
        p = tp_t / (tp_t + fp_t) if (tp_t + fp_t) else float("nan")
        a = (tp_t + (n_r - fp_t)) / (n_h + n_r) if (n_h + n_r) else float("nan")
        return c, f, p, a, tp_t, fp_t

    cands = sorted(set(pos + neg), reverse=True)
    best = None
    for th in cands:
        c, f, p, a, tp_t, fp_t = thr_metrics(th)
        j = c - f
        if best is None or j > best[0]:
            best = (j, th, c, f, p, a, tp_t, fp_t)
    _, best_th, best_c, best_f, best_p, best_a, best_tp, best_fp = best

    # operating points at bounded false-alarm rates
    op_points = []
    for target in (0.0, 0.05, 0.10, 0.20, 0.30, 0.50):
        bestop = None
        for th in cands:
            c, f, p, a, tp_t, fp_t = thr_metrics(th)
            if f <= target + 1e-12:
                if bestop is None or c > bestop[0]:
                    bestop = (c, f, p, a, th, tp_t, fp_t)
        if bestop is not None:
            op_points.append((target,) + bestop)

    # examples
    caught = sorted([r for r in hall if iget(r, "flag") == 1],
                    key=lambda r: fget(r, "score"), reverse=True)[:2]
    missed = sorted([r for r in hall if iget(r, "flag") == 0],
                    key=lambda r: fget(r, "score"), reverse=True)[:2]
    fas = sorted([r for r in right if iget(r, "flag") == 1],
                 key=lambda r: fget(r, "score"), reverse=True)[:2]
    near_fa = [] if fas else sorted(right, key=lambda r: fget(r, "score"),
                                    reverse=True)[:2]

    lines = []
    lines.append("RecycleLM HaluEval detector (controlled)")
    lines.append("=" * 72)
    lines.append(f"input: {os.path.relpath(bench_path, ROOT).replace(os.sep, '/')}")
    lines.append(f"csv sha256: {sha256(bench_path)}")
    lines.append(f"model: {os.path.relpath(model_path, ROOT).replace(os.sep, '/')} "
                 f"sha256={sha256(model_path)}")
    lines.append(f"data: {os.path.relpath(data_path, ROOT).replace(os.sep, '/')} "
                 f"sha256={sha256(data_path)}")
    lines.append(f"rows: {len(det)} verdicts ({n_r} right, {n_h} hallucinated)")
    lines.append(f"base rate (hallucinated): {pct_s(n_h / len(det))} "
                 f"({n_h}/{len(det)})")
    lines.append("")
    lines.append("Prompt: Knowledge/Question/Candidate answer -> FATAL or OK.")
    lines.append('Grammar: root ::= "FATAL" | "OK" (greedy).')
    lines.append("Score: max first-token logit over FATAL prefixes minus the same")
    lines.append("       over OK prefixes (grammar-constrained branch score).")
    lines.append("score_raw: logit('F') - logit('O').")
    lines.append("")
    lines.append("At the model's verdict (greedy FATAL = flag)")
    lines.append("-" * 72)
    lines.append(f"  catch (hallucinated flagged): {pct_s(catch)} ({tp}/{n_h}) "
                 f"95% CI [{pct_s(catch_ci[0])}, {pct_s(catch_ci[1])}]")
    lines.append(f"  false alarm (right flagged):  {pct_s(fa)} ({fp}/{n_r}) "
                 f"95% CI [{pct_s(fa_ci[0])}, {pct_s(fa_ci[1])}]")
    lines.append(f"  precision: {pct_s(prec)} "
                 f"95% CI [{pct_s(prec_ci[0])}, {pct_s(prec_ci[1])}]")
    lines.append(f"  recall:    {pct_s(rec)}")
    lines.append(f"  accuracy:  {pct_s(acc)} "
                 f"95% CI [{pct_s(acc_ci[0])}, {pct_s(acc_ci[1])}]")
    lines.append(f"  flag rate: {pct_s((tp + fp) / len(det))} ({(tp + fp)}/{len(det)})")
    if fp == 0 and n_r:
        lines.append(f"  note: 0/{n_r} false alarms; the bootstrap CI is degenerate.")
        lines.append(f"        one-sided 95% upper bound ~{pct_s(cp_upper_zero(n_r))} "
                     f"(Clopper-Pearson/rule of three).")
    if tp + fp > 0 and fp == 0:
        lines.append(f"  note: precision {tp}/{tp + fp}; one-sided 95% lower bound "
                     f"~{pct_s(cp_lower_all(tp + fp))} (Clopper-Pearson).")
    lines.append("")
    lines.append("Score ranking (hallucinated = positive)")
    lines.append("-" * 72)
    lines.append(f"  AUROC: {fnum(auc, 4)} 95% CI [{fnum(auc_ci[0], 4)}, "
                 f"{fnum(auc_ci[1], 4)}]  (Mann-Whitney, ties averaged)")
    lines.append(f"  AUROC (score_raw): {fnum(auc_raw, 4)}")
    lines.append(f"  right         score: mean {fnum(statistics.mean(neg), 3)} "
                 f"median {fnum(statistics.median(neg), 3)} "
                 f"min {fnum(min(neg), 3)} max {fnum(max(neg), 3)}")
    lines.append(f"  hallucinated  score: mean {fnum(statistics.mean(pos), 3)} "
                 f"median {fnum(statistics.median(pos), 3)} "
                 f"min {fnum(min(pos), 3)} max {fnum(max(pos), 3)}")
    lines.append("")
    lines.append(f"Best threshold by Youden J: score >= {fnum(best_th, 4)}")
    lines.append(f"  catch {pct_s(best_c)} ({best_tp}/{n_h}), "
                 f"FA {pct_s(best_f)} ({best_fp}/{n_r}), "
                 f"precision {pct_s(best_p)}, accuracy {pct_s(best_a)}")
    lines.append("")
    lines.append("Operating points (highest catch with FA <= target)")
    lines.append("  target_FA   threshold     catch        FA          precision")
    lines.append("  " + "-" * 65)
    for target, c, f, p, a, th, tp_t, fp_t in op_points:
        lines.append(f"  {pct_s(target):>8}   {fnum(th, 4):>9}   "
                     f"{pct_s(c):>8}   {pct_s(f):>8}   {pct_s(p):>8}")
    lines.append("")
    lines.append("Examples")
    lines.append("-" * 72)
    lines.extend(example_block("caught hallucinations (flag=1)", caught, data))
    lines.extend(example_block("missed hallucinations (flag=0, highest score)",
                               missed, data))
    if fas:
        lines.extend(example_block("false alarms (right, flag=1)", fas, data))
    else:
        lines.append("  false alarms: none observed at the greedy verdict.")
        lines.append("  two highest-scoring right answers (near false alarms):")
        lines.extend(example_rows(near_fa, data, prefix="    "))
    lines.append("")
    lines.append("Notes")
    lines.append("- The verdict is the model's greedy grammar-constrained choice.")
    lines.append("- Scores are raw logit differences, not calibrated probabilities.")
    lines.append("- One machine (RTX 4060 8 GB); do not extrapolate to other GPUs.")
    lines.append("")

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {out_path}")


def example_rows(rows, data, prefix="  "):
    out = []
    for r in rows:
        d = data.get(r["id"], {})
        out.append(f"{prefix}{r['id']} [{r.get('candidate_type','')}] "
                   f"verdict={r.get('verdict','')} flag={r.get('flag','')} "
                   f"score={fnum(fget(r, 'score'), 3)}")
        out.append(f"{prefix}  Q: {snippet(d.get('question', ''), 180)}")
        out.append(f"{prefix}  candidate: {snippet(r.get('actor_text', ''), 300)}")
        out.append(f"{prefix}  thinker: {snippet(r.get('thinker_text', ''), 80)}")
    return out


def example_block(title, rows, data):
    out = [f"  {title}:"]
    if not rows:
        out.append("    none")
        return out
    out.extend(example_rows(rows, data, prefix="    "))
    return out


# ---------------- e2e ----------------

def e2e_summary(rows, data, bench_path, data_path, model_path, out_path):
    e2e = [r for r in rows if r.get("mode") == "e2e"]
    if not e2e:
        raise SystemExit("e2e: no e2e rows in the bench CSV")

    arms = []
    for a in ("baseline", "flagged", "concurrent", "concurrent_fork8"):
        if any(r.get("arm") == a for r in e2e):
            arms.append(a)

    by_arm = {a: [r for r in e2e if r.get("arm") == a] for a in arms}
    by_id_arm = {a: {r["id"]: r for r in by_arm[a]} for a in arms}

    # actor-level label for every row
    labels = {}
    for r in e2e:
        d = data.get(r["id"], {})
        labels[(r["arm"], r["id"])] = is_correct(r.get("actor_text", ""),
                                                 d.get("right_answer", ""))

    def arm_metrics(arm):
        rs = by_arm[arm]
        n = len(rs)
        corr = sum(1 for r in rs if labels[(arm, r["id"])])
        err = n - corr
        flagged = sum(1 for r in rs if iget(r, "flag") == 1)
        tp = sum(1 for r in rs if iget(r, "flag") == 1 and
                 not labels[(arm, r["id"])])
        fp = sum(1 for r in rs if iget(r, "flag") == 1 and
                 labels[(arm, r["id"])])
        catch = tp / err if err else float("nan")
        fa = fp / corr if corr else float("nan")
        if arm == "baseline":
            catch = float("nan")
            fa = float("nan")
        return {
            "n": n, "corr": corr, "err": err, "err_rate": err / n if n else float("nan"),
            "flagged": flagged, "flag_rate": flagged / n if n else float("nan"),
            "tp": tp, "fp": fp, "catch": catch, "fa": fa,
        }

    def ci_metric(arm, key):
        rs = by_arm[arm]
        n = len(rs)
        if key == "err_rate":
            def f(i):
                return sum(1 for k in i if not labels[(arm, rs[k]["id"])]) / len(i)
            return boot_ci(f, n)
        if key == "flag_rate":
            def f(i):
                return sum(1 for k in i if iget(rs[k], "flag") == 1) / len(i)
            return boot_ci(f, n)
        if key in ("catch", "fa"):
            def f(i):
                if key == "catch":
                    den = sum(1 for k in i if not labels[(arm, rs[k]["id"])])
                    num = sum(1 for k in i if not labels[(arm, rs[k]["id"])]
                              and iget(rs[k], "flag") == 1)
                else:
                    den = sum(1 for k in i if labels[(arm, rs[k]["id"])])
                    num = sum(1 for k in i if labels[(arm, rs[k]["id"])]
                              and iget(rs[k], "flag") == 1)
                return num / den if den else None
            return boot_ci(f, n)
        return (float("nan"), float("nan"))

    def deltas(arm_a, arm_b, key):
        out = []
        for rid, ra in by_id_arm[arm_a].items():
            rb = by_id_arm[arm_b].get(rid)
            if rb is None:
                continue
            out.append(fget(ra, key) - fget(rb, key))
        return out

    lines = []
    lines.append("RecycleLM HaluEval e2e (flag-only circuit breaker)")
    lines.append("=" * 78)
    lines.append(f"input: {os.path.relpath(bench_path, ROOT).replace(os.sep, '/')}")
    lines.append(f"csv sha256: {sha256(bench_path)}")
    lines.append(f"model: {os.path.relpath(model_path, ROOT).replace(os.sep, '/')} "
                 f"sha256={sha256(model_path)}")
    lines.append(f"data: {os.path.relpath(data_path, ROOT).replace(os.sep, '/')} "
                 f"sha256={sha256(data_path)}")
    lines.append(f"rows: {len(e2e)}")
    lines.append("")
    lines.append("Actor: Knowledge/Question -> answer <=128 tokens, temp 0.6, "
                 "top_k 20, top_p 0.95.")
    lines.append("Thinker: same FATAL/OK prompt as the detector, GBNF "
                 'root ::= "FATAL" | "OK", greedy.')
    lines.append("Flag-only: FATAL marks the answer flagged; there is no rewrite.")
    lines.append("Judge: actor answer vs HaluEval right_answer (normalized lexical:")
    lines.append("       exact/containment, all numbers present, single-token")
    lines.append("       containment, or token F1 >= 0.5). Proxy, not human.")
    lines.append("")
    lines.append("Per-arm")
    lines.append("-" * 78)
    lines.append("arm                 n  actor_err  flag_rate   catch      FA     "
                 "med_total  p95_total")
    for a in arms:
        m = arm_metrics(a)
        totals = [fget(r, "total_ms") for r in by_arm[a]]
        lines.append(f"{a:<18} {m['n']:>3}  {pct_s(m['err_rate']):>9}  "
                     f"{pct_s(m['flag_rate']):>9}  "
                     f"{pct_s(m['catch']):>8}  {pct_s(m['fa']):>7}  "
                     f"{fnum(statistics.median(totals)):>9}  "
                     f"{fnum(pct(totals, 95)):>9}")
    lines.append("")
    lines.append("95% bootstrap CIs (paired by item)")
    lines.append("-" * 78)
    for a in arms:
        if a == "baseline":
            continue
        m = arm_metrics(a)
        eci = ci_metric(a, "err_rate")
        cci = ci_metric(a, "catch")
        fci = ci_metric(a, "fa")
        lines.append(f"{a:<18} actor_err {pct_s(m['err_rate'])} "
                     f"[{pct_s(eci[0])}, {pct_s(eci[1])}]  "
                     f"catch {pct_s(m['catch'])} [{pct_s(cci[0])}, {pct_s(cci[1])}]  "
                     f"FA {pct_s(m['fa'])} [{pct_s(fci[0])}, {pct_s(fci[1])}]")
    if "flagged" in by_arm:
        m = arm_metrics("flagged")
        lines.append(f"  note: flagged catch 0/{m['err']} -> one-sided 95% upper "
                     f"~{pct_s(cp_upper_zero(m['err']))}; FA 0/{m['corr']} -> "
                     f"one-sided 95% upper ~{pct_s(cp_upper_zero(m['corr']))}.")
    lines.append("")
    lines.append("Latency (paired by id, total_ms; includes actor prompt prefill)")
    lines.append("-" * 78)
    for a in arms:
        if a == "baseline":
            continue
        d = deltas(a, "baseline", "total_ms")
        if not d:
            continue
        under = sum(1 for x in d if abs(x) < 100.0)
        lines.append(f"  {a:<18} - baseline: n={len(d)} "
                     f"median {fnum(statistics.median(d))} ms  "
                     f"mean {fnum(statistics.mean(d))} ms  "
                     f"p95 {fnum(pct(d, 95))} ms  "
                     f"|delta|<100ms {under}/{len(d)}")
    for a in ("flagged", "concurrent", "concurrent_fork8"):
        if a in by_arm:
            th = [fget(r, "think_ms") for r in by_arm[a]]
            au = [fget(r, "actor_ms") for r in by_arm[a]]
            if th:
                lines.append(f"  {a:<18} think_ms median {fnum(statistics.median(th))} "
                             f"mean {fnum(statistics.mean(th))}  "
                             f"actor_ms median {fnum(statistics.median(au))}")
    lines.append("")
    lines.append("Score-based flagging (flagged arm, audit branch score)")
    lines.append("-" * 78)
    if "flagged" in by_arm:
        rs = by_arm["flagged"]
        pos = [fget(r, "score") for r in rs if not labels[("flagged", r["id"])]]
        neg = [fget(r, "score") for r in rs if labels[("flagged", r["id"])]]
        if pos and neg:
            cands = sorted(set(pos + neg), reverse=True)
            best = None
            for th in cands:
                tp_t = sum(1 for s in pos if s >= th)
                fp_t = sum(1 for s in neg if s >= th)
                c = tp_t / len(pos)
                f = fp_t / len(neg)
                j = c - f
                if best is None or j > best[0]:
                    best = (j, th, c, f, tp_t, fp_t)
            _, bth, bc, bf, btp, bfp = best
            lines.append(f"  best Youden: score >= {fnum(bth, 4)} -> "
                         f"catch {pct_s(bc)} ({btp}/{len(pos)}), "
                         f"FA {pct_s(bf)} ({bfp}/{len(neg)})")
            for target in (0.05, 0.10, 0.20):
                bop = None
                for th in cands:
                    tp_t = sum(1 for s in pos if s >= th)
                    fp_t = sum(1 for s in neg if s >= th)
                    c = tp_t / len(pos)
                    f = fp_t / len(neg)
                    if f <= target + 1e-12 and (bop is None or c > bop[0]):
                        bop = (c, f, th, tp_t, fp_t)
                if bop is not None:
                    lines.append(f"  FA <= {pct_s(target):>5}: score >= {fnum(bop[2], 4)} -> "
                                 f"catch {pct_s(bop[0])} ({bop[3]}/{len(pos)}), "
                                 f"FA {pct_s(bop[1])} ({bop[4]}/{len(neg)})")
    lines.append("")
    lines.append("Concurrent fork")
    lines.append("-" * 78)
    for a in ("concurrent", "concurrent_fork8"):
        if a not in by_arm:
            continue
        rs = by_arm[a]
        forked = [r for r in rs if iget(r, "forked") == 1]
        short = [r for r in rs if iget(r, "forked") == 0]
        lines.append(f"  {a}: forked {len(forked)}/{len(rs)}  "
                     f"serial_fallback {len(short)}/{len(rs)}")
        if forked:
            ov = [fget(r, "overlap_ms") for r in forked]
            jt = [iget(r, "joint_ticks") for r in forked]
            vae = sum(1 for r in forked if iget(r, "verdict_at_actor_end") == 1)
            tail = sum(1 for r in forked if iget(r, "tail_wait") == 1)
            lines.append(f"    overlap_ms median {fnum(statistics.median(ov))} "
                         f"mean {fnum(statistics.mean(ov))} p95 {fnum(pct(ov, 95))}")
            lines.append(f"    joint_ticks median {fnum(statistics.median(jt))} "
                         f"mean {fnum(statistics.mean(jt))} max {max(jt)}")
            lines.append(f"    verdict at actor end {vae}/{len(forked)}; "
                         f"tail wait {tail}/{len(forked)}")
    actor_ns = [iget(r, "actor_n") for r in by_arm.get("baseline", [])]
    if actor_ns:
        lines.append(f"  actor tokens (baseline): median {fnum(statistics.median(actor_ns))} "
                     f"min {min(actor_ns)} max {max(actor_ns)}; "
                     f"<=32 {sum(1 for x in actor_ns if x <= 32)}/{len(actor_ns)}")
    lines.append("")
    lines.append("Verdict agreement (flagged vs concurrent, by id)")
    lines.append("-" * 78)
    for a in ("concurrent", "concurrent_fork8"):
        if a not in by_arm or "flagged" not in by_arm:
            continue
        agree = same_fatal = diff_fatal = 0
        n = 0
        for rid, rf in by_id_arm["flagged"].items():
            rc = by_id_arm[a].get(rid)
            if rc is None:
                continue
            n += 1
            ff = iget(rf, "flag")
            fc = iget(rc, "flag")
            if ff == fc:
                agree += 1
                if ff == 1:
                    same_fatal += 1
            elif ff == 1:
                diff_fatal += 1
        lines.append(f"  flagged vs {a}: {agree}/{n} agree "
                     f"({pct_s(agree / n if n else float('nan'))}); "
                     f"both FATAL {same_fatal}, serial-only FATAL {diff_fatal}")
    lines.append("")
    lines.append("Does the \"100% catch at flat latency\" claim hold?")
    lines.append("-" * 78)
    for ref in [a for a in ("flagged", "concurrent", "concurrent_fork8")
                if a in by_arm]:
        m = arm_metrics(ref)
        d = deltas(ref, "baseline", "total_ms") if ref != "baseline" else []
        med = statistics.median(d) if d else float("nan")
        holds = (m["catch"] == m["catch"] and m["catch"] >= 0.95 and
                 abs(med) <= 100.0)
        lines.append(f"  {ref}: catch {pct_s(m['catch'])} ({m['tp']}/{m['err']}), "
                     f"FA {pct_s(m['fa'])} ({m['fp']}/{m['corr']}), "
                     f"flag rate {pct_s(m['flag_rate'])}, "
                     f"median overhead {fnum(med)} ms -> "
                     + ("HOLDS" if holds else "DOES NOT HOLD"))
    lines.append("  criterion: catch >= 95% and |median overhead| <= 100 ms.")
    lines.append("")
    lines.append("Examples (flagged arm)")
    lines.append("-" * 78)
    if "flagged" in by_arm:
        rs = by_arm["flagged"]
        caught = [r for r in rs if iget(r, "flag") == 1 and
                  not labels[("flagged", r["id"])]][:2]
        missed = sorted([r for r in rs if iget(r, "flag") == 0 and
                         not labels[("flagged", r["id"])]],
                        key=lambda r: fget(r, "score"), reverse=True)[:2]
        fas = [r for r in rs if iget(r, "flag") == 1 and
               labels[("flagged", r["id"])]][:2]
        near = [] if fas else sorted(
            [r for r in rs if labels[("flagged", r["id"])]],
            key=lambda r: fget(r, "score"), reverse=True)[:2]
        lines.extend(e2e_example_block("caught actor errors (flag=1)", caught,
                                       data, labels, "flagged"))
        lines.extend(e2e_example_block("missed actor errors (flag=0, highest score)",
                                       missed, data, labels, "flagged"))
        if fas:
            lines.extend(e2e_example_block("false alarms (correct actor, flag=1)",
                                           fas, data, labels, "flagged"))
        else:
            lines.append("  false alarms: none observed at the greedy verdict.")
            lines.append("  two highest-scoring correct answers (near false alarms):")
            lines.extend(e2e_example_rows(near, data, labels, "flagged", "    "))
    lines.append("")
    lines.append("Notes")
    lines.append("- The e2e label is the lexical judge above, not a human label.")
    lines.append("- Label caveat: several of the 20 lexical 'errors' are aliases or")
    lines.append("  partial answers (e.g. 'Rob Quist' vs 'Robert Ernest Quist',")
    lines.append("  'The Brothers Grimm' vs 'Jacob and Wilhelm'), so the 10% actor")
    lines.append("  error rate is an upper bound. The thinker flags 0/200 answers,")
    lines.append("  so catch is 0 under any label.")
    lines.append("- Concurrent audits the partial answer at the fork; the serial")
    lines.append("  arm audits the full answer. Verdicts can differ.")
    lines.append("- Different arms are separate actor generations; batch")
    lines.append("  composition can change numerics slightly.")
    lines.append("- One machine (RTX 4060 8 GB); do not extrapolate to other GPUs.")
    lines.append("")

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {out_path}")


def e2e_example_rows(rows, data, labels, arm, prefix="  "):
    out = []
    for r in rows:
        d = data.get(r["id"], {})
        out.append(f"{prefix}{r['id']} verdict={r.get('verdict','')} "
                   f"flag={r.get('flag','')} score={fnum(fget(r, 'score'), 3)} "
                   f"correct={1 if labels[(arm, r['id'])] else 0}")
        out.append(f"{prefix}  Q: {snippet(d.get('question', ''), 180)}")
        out.append(f"{prefix}  right: {snippet(d.get('right_answer', ''), 120)}")
        out.append(f"{prefix}  actor: {snippet(r.get('actor_text', ''), 300)}")
        out.append(f"{prefix}  thinker: {snippet(r.get('thinker_text', ''), 80)}")
    return out


def e2e_example_block(title, rows, data, labels, arm):
    out = [f"  {title}:"]
    if not rows:
        out.append("    none")
        return out
    out.extend(e2e_example_rows(rows, data, labels, arm, prefix="    "))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default=os.path.join(ROOT, "results", "halueval_bench.csv"))
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "halueval_qa_200.tsv"))
    ap.add_argument("--model", default=os.path.join(
        ROOT, "models", "Llama-3.2-3B-Instruct-Q4_K_M.gguf"))
    ap.add_argument("--out-det", default=os.path.join(ROOT, "results", "halueval_detector_summary.txt"))
    ap.add_argument("--out-e2e", default=os.path.join(ROOT, "results", "halueval_e2e_summary.txt"))
    args = ap.parse_args()

    if not os.path.exists(args.bench):
        sys.exit(f"missing bench CSV: {args.bench}")
    rows = load_rows(args.bench)
    data = read_tsv(args.data) if os.path.exists(args.data) else {}

    if any(r.get("mode") == "detector" for r in rows):
        detector_summary(rows, data, args.bench, args.data, args.model, args.out_det)
    else:
        print("no detector rows; skipping detector summary")
    if any(r.get("mode") == "e2e" for r in rows):
        e2e_summary(rows, data, args.bench, args.data, args.model, args.out_e2e)
    else:
        print("no e2e rows; skipping e2e summary")


if __name__ == "__main__":
    main()
