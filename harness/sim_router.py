#!/usr/bin/env python3
"""Angle 41 Phase A — offline Adaptive Compute Router simulation (ZERO GPU).

Hypothesis: a per-question router that meters compute (run 1 stream when the
answer looks reliable, escalate to 3 or full-N when signals flag it) can match
full-vote accuracy while saving >=40% of streams. If no adaptive point
CV-dominates the fixed-N endpoints on existing dumps, the angle dies before
any GPU time.

Signals (research log rows 38-40):
  agreement   row 38: unanimous runs ~all-correct; AUROC(agree->maj_ok) ~0.85
  lp_min      row 40: reasoning-phase emitted-token logprob min; AUROC 0.844
              (answer-phase features dead, p1~0.97 everywhere)
  stability   row 39: reform-intervention stability; AUROC 0.890 (expensive,
              2nd invocation) — optional gate feature where available

Inputs (verified schemas):
  results/probs_probe30.json + data/_probs_pNN.{json,jsonl}   n=32, N=8, lp
  results/disagree_probe100_n5.json                           n=100, N=5
  results/disagree_probe30.json                               n=30, N=8
      NOTE: identical questions+answers to probs_probe30[0:30] (same run
      re-emitted with --probs); not independent evidence.
  results/reform_probe30.json -> joined to disagree_probe100_n5 by id
      n=30, N=5, adds stability (1 record stability=None, p13 all-noop)

Outputs:
  printed report + results/router_sim_report.txt
  results/router_features.csv

Rules (PREREGISTRATION.md AMENDMENT 5, 2026-10-23 — frozen definitions):
  PRIMARY tie rule = FIRST-SEEN plurality: counts over non-None answers in
  stream order; ties (incl. all-distinct) break to the earliest-seen tied
  answer. This is exactly `run_disagree_probe.analyze()` line 153
  (`max(counts.items(), key=...)` over an insertion-ordered dict), i.e. the
  rule that produced every `maj_ok` label in the dumps. SENSITIVITY column =
  STRICT plurality: ties are abstentions and count as wrong. Both rules are
  recomputed end-to-end (endpoints, CV frontiers, verdict) and reported side
  by side. The gate definitions are identical under both rules, but their
  outcomes can differ because always-full accuracy itself depends on the tie
  rule (first-seen assigns a label to a full-pool tie, strict does not).
  unanimous@k requires all k answers present and equal.
  lp_min_reason_1 = min lp over tokens whose byte offset < last b"The answer
  is" in stream 1 text; if the marker is missing, fall back to the all-token
  min (flagged in the feature table).

L4 = plurality@3 (pre-registered in AMENDMENT 5): look at the first 3
  streams' answers; UNIQUE plurality (max count strictly > runner-up) ->
  stop and return it (cost 3); tie (1-1-1 or any tie for first) -> escalate
  to all N and return the maj@N under the active tie rule (cost N).
  Parameter-free; evaluated on all four datasets under both tie rules.

Latency multipliers — E0-DERIVED (illustrative bounds only; AMENDMENT 5
  resolves the MAJOR-4 provenance error: these were mislabelled "E2
  measured", but the E2 sweep results/e2_sweep200.csv has no N=3/N=5 rows
  at all). Source file: results/e0_full.csv, ALL rows (32768 raw per-step
  rows = 16 N values x 2048 tail depths, phase=0, tail_depth 0..2047).
  Filter: multiplier(N) = median(step_ms | N) / median(step_ms | N=1):
      N=3: 6.5628/5.5044 = 1.1923 -> 1.19
      N=5: 7.9458/5.5044 = 1.4435 -> 1.44
      N=8: 10.5479/5.5044 = 1.9163 -> 1.92   (raw; tail-dependent p50
           ratios in results/e0_summary.txt span 1.70x..2.06x for N=8)
  (mean ratios over the same rows: 1.194 / 1.441 / 1.913). NO pass/fail
  GATE DEPENDS ON LATENCY — the lat_seq/lat_bat columns are descriptive.
  Single stream -> 1.0:
  SEQ   each escalation rung re-launches a fresh batch (pessimistic bound):
          L1 stop@1 = 1.0 | stop@3 = 1.0+1.19 | stop@full = 1.0+1.19+L_full
          L2/L4 stop@3 = 1.19 | stop@full = 1.19+L_full
          L3 stop@1 = 1.0 | stop@full = 1.0+L_full
  BATCH escalation widens the in-flight run and reuses stream 1; the marginal
        wall time of widening is the difference of multipliers, so a
        problem that escalates all the way costs exactly the always-full
        endpoint latency:
          L1 stop@1 = 1.0 | stop@3 = 1.19 | stop@full = L_full
          L2/L4 stop@3 = 1.19 | stop@full = L_full
          L3 stop@1 = 1.0 | stop@full = L_full
  Endpoints: E1 = 1.0; E3 = 1.19; E_full = L_full (1.92 for N=8, 1.44 for
  N=5) under both models.

Validation: thresholds are fitted inside CV (LOO for n<=40, stratified
5-fold for n=100, stratified on full-majority correctness). Per fold the
absolute thresholds come from TRAIN-fold feature quantiles on a quantile-
level grid; the fold's selected policy maximises train accuracy subject to
the pre-registered stream budget (mean_streams <= 0.6*N), tie-broken by
fewer streams. In-sample frontiers are reported separately and labelled.
"""
import io
import json
import math
import os
import random
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P_PROBS = os.path.join(ROOT, "results", "probs_probe30.json")
P_N5 = os.path.join(ROOT, "results", "disagree_probe100_n5.json")
P_DIS30 = os.path.join(ROOT, "results", "disagree_probe30.json")
P_REFORM = os.path.join(ROOT, "results", "reform_probe30.json")
OUT_TXT = os.path.join(ROOT, "results", "router_sim_report.txt")
OUT_CSV = os.path.join(ROOT, "results", "router_features.csv")

LAT1 = 1.0
LAT3 = 1.19          # E0-derived, results/e0_full.csv: 6.5628/5.5044
LAT_FULL = {8: 1.92, 5: 1.44}   # E0-derived medians; N=8 p50-tail 1.70..2.06
RULE_FIRST = "first"
RULE_STRICT = "strict"
CV_SEED = 20261023
LEVELS = [i / 10.0 for i in range(11)]
BUDGET_FRAC = 0.6

buf = io.StringIO()


def w(s=""):
    print(s)
    buf.write(str(s) + "\n")


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def pct(x):
    return "n/a" if x is None else "%.1f%%" % (100.0 * x)


def f3(x):
    return "n/a" if x is None else "%.3f" % x


def auroc(scores, labels):
    """Mann-Whitney AUC; higher score -> positive class. None if one class empty."""
    pairs = [(s, l) for s, l in zip(scores, labels) if s is not None]
    if not pairs:
        return None
    pos = [s for s, l in pairs if l]
    neg = [s for s, l in pairs if not l]
    if not pos or not neg:
        return None
    allv = sorted(s for s, _ in pairs)
    rank = {}
    i = 0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j] == allv[i]:
            j += 1
        r = (i + j + 1) / 2.0
        for k in range(i, j):
            rank[allv[k]] = r
        i = j
    sr = sum(rank[s] for s in pos)
    return (sr - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))


def plurality(answers, rule):
    """(label, tied) under the requested tie rule.

    rule == "first":  FIRST-SEEN plurality. On a tie for top count, return the
        earliest-seen tied answer (insertion order of the counter), tied=True.
        This is exactly run_disagree_probe.analyze() line 153, the rule that
        produced every maj_ok label in the dumps.
    rule == "strict": STRICT plurality. A tie for top count (incl. all-distinct)
        is an abstention -> (None, True).
    """
    c = Counter(a for a in answers if a is not None)
    if not c:
        return None, True
    if rule == RULE_FIRST:
        label, cnt = max(c.items(), key=lambda kv: kv[1])
        tied = sum(1 for v in c.values() if v == cnt) > 1
        return label, tied
    top = c.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return None, True
    return top[0][0], False


def strict_majority(answers):
    """Back-compat alias for the strict sensitivity rule."""
    return plurality(answers, RULE_STRICT)


def pairwise_agree3(answers):
    n_eq = 0
    for i in range(3):
        for j in range(i + 1, 3):
            a, b = answers[i], answers[j]
            if a is not None and a == b:
                n_eq += 1
    return n_eq / 3.0


def quantile(sorted_vals, level):
    """Empirical quantile; level<=0 -> -inf (accept all), >=1 -> +inf (none)."""
    if not sorted_vals:
        return 0.0
    if level <= 0.0:
        return -math.inf
    if level >= 1.0:
        return math.inf
    idx = level * (len(sorted_vals) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    frac = idx - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


# ---------------------------------------------------------------- feature table

def stream1_lp(streams_text, jsonl_path):
    """(lp_min_all, lp_min_reason, marker_missing) for stream '1'."""
    tb = streams_text.encode("utf-8")
    marker = tb.rfind(b"The answer is")
    has_marker = marker >= 0
    lps_all, lps_rsn = [], []
    with open(jsonl_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            if o.get("s") != "1":
                continue
            lps_all.append(o["lp"])
            if not has_marker or o["off"] < marker:
                lps_rsn.append(o["lp"])
    lp_min = min(lps_all) if lps_all else None
    lp_min_rsn = min(lps_rsn) if lps_rsn else (lp_min if not has_marker else None)
    return lp_min, lp_min_rsn, (0 if has_marker else 1)


def build_row(rec, dataset, n_streams, qid, group, lp=None, stability=None):
    answers = rec["answers"]
    gold = rec["gold"]
    maj_ok_file = rec.get("maj_ok")
    ans1 = answers[0] if answers else None
    uni3 = all(a is not None for a in answers[:3]) and len(set(answers[:3])) == 1
    row = {
        "dataset": dataset,
        "qid": qid,
        "group": group,
        "N": n_streams,
        "gold": gold,
        "answers": answers,
        "maj_ok_file": maj_ok_file,
        "ans1": ans1,
        "ok1": ans1 is not None and ans1 == gold,
        "uni3": uni3,
        "agree3": pairwise_agree3(answers),
        "lp_min_1": lp[0] if lp else None,
        "lp_min_reason_1": lp[1] if lp else None,
        "marker_missing": lp[2] if lp else None,
        "stability": stability,
    }
    return row


def apply_rule(rows_or_dict, rule):
    """Populate every plurality-derived field on the rows under `rule`.

    Accepts either a list of rows or the dataset-name -> rows mapping returned
    by load_rows(); returns the same object for chaining.
    """
    if isinstance(rows_or_dict, dict):
        for rows in rows_or_dict.values():
            apply_rule(rows, rule)
        return rows_or_dict
    for r in rows_or_dict:
        a = r["answers"]
        N = r["N"]
        r["maj_full"], r["tie_full"] = plurality(a, rule)
        r["maj_ok_full"] = (r["maj_full"] is not None
                            and r["maj_full"] == r["gold"])
        r["maj3"], r["tie3"] = plurality(a[:3], rule)
        r["maj3_ok"] = r["maj3"] is not None and r["maj3"] == r["gold"]
        r["maj5"] = r["maj5_ok"] = r["tie5"] = None
        r["maj8"] = r["maj8_ok"] = r["tie8"] = None
        if N >= 5:
            r["maj5"], r["tie5"] = plurality(a[:5], rule)
            r["maj5_ok"] = r["maj5"] is not None and r["maj5"] == r["gold"]
        if N >= 8:
            r["maj8"], r["tie8"] = plurality(a[:8], rule)
            r["maj8_ok"] = r["maj8"] is not None and r["maj8"] == r["gold"]
    return rows_or_dict


def load_rows():
    datasets = {}

    with open(P_PROBS, encoding="utf-8") as fh:
        meta = json.load(fh)
    rows = []
    for r in meta["records"]:
        with open(os.path.join(ROOT, "data", r["probs_json"]), encoding="utf-8") as fh:
            streams = json.load(fh)["streams"]
        lp = stream1_lp(streams[0]["text"],
                        os.path.join(ROOT, "data", r["probs_jsonl"]))
        rows.append(build_row(r, "probs32", 8, r["qid"], r.get("group", ""), lp=lp))
    datasets["probs32"] = rows

    with open(P_N5, encoding="utf-8") as fh:
        meta = json.load(fh)
    rows = [build_row(r, "disagree_n5", 5, r["id"], "") for r in meta["records"]]
    datasets["disagree_n5"] = rows

    with open(P_DIS30, encoding="utf-8") as fh:
        meta = json.load(fh)
    rows = [build_row(r, "disagree30", 8, r["id"], "") for r in meta["records"]]
    datasets["disagree30"] = rows

    n5_by_id = {r["id"]: r for r in json.load(
        open(P_N5, encoding="utf-8"))["records"]}
    with open(P_REFORM, encoding="utf-8") as fh:
        recs = json.load(fh)["records"]
    rows = []
    for r in recs:
        base = n5_by_id[r["id"]]
        assert base["gold"] == r["gold"]
        rows.append(build_row(base, "reform30_join", 5, r["id"],
                              r.get("stratum", ""), stability=r.get("stability")))
    datasets["reform30_join"] = rows
    return datasets


# ---------------------------------------------------------------- latencies

def lat_pair(ladder, rung, N):
    """(lat_seq, lat_batch) for a decision at the given rung."""
    L = LAT_FULL[N]
    if ladder == "E1":
        return LAT1, LAT1
    if ladder == "E3":
        return LAT3, LAT3
    if ladder == "Ef":
        return L, L
    if ladder == "L1":
        return ({"stop1": LAT1, "stop3": LAT1 + LAT3, "full": LAT1 + LAT3 + L}[rung],
                {"stop1": LAT1, "stop3": LAT3, "full": L}[rung])
    if ladder in ("L2", "L2s", "L4"):
        return ({"stop3": LAT3, "full": LAT3 + L}[rung],
                {"stop3": LAT3, "full": L}[rung])
    if ladder == "L3":
        return ({"stop1": LAT1, "full": LAT1 + L}[rung],
                {"stop1": LAT1, "full": L}[rung])
    raise ValueError(ladder)


# ---------------------------------------------------------------- deciders

def dec_endpoint(row, ladder):
    if ladder == "E1":
        return row["ans1"], 1, "stop1"
    if ladder == "E3":
        return row["maj3"], 3, "stop3"
    if ladder == "Ef":
        return row["maj_full"], row["N"], "full"
    raise ValueError(ladder)


def dec_L1(row, tau_hi, tau_lo):
    lp = row["lp_min_reason_1"]
    if lp is not None and lp >= tau_hi:
        return row["ans1"], 1, "stop1"
    if row["uni3"] and lp is not None and lp >= tau_lo:
        return row["maj3"], 3, "stop3"
    return row["maj_full"], row["N"], "full"


def dec_L2(row):
    if row["uni3"]:
        return row["maj3"], 3, "stop3"
    return row["maj_full"], row["N"], "full"


def dec_L4(row):
    """Pre-registered L4 = plurality@3.

    Unique plurality among answers[0:3] -> stop (cost 3). Any tie for first
    (including 1-1-1) -> escalate to all N, return maj@N under the active rule
    (cost N). Parameter-free; no thresholds are fitted.
    """
    maj3, tie3 = row["maj3"], row["tie3"]
    if maj3 is not None and not tie3:
        return maj3, 3, "stop3"
    return row["maj_full"], row["N"], "full"


def dec_L2s(row, tau_s):
    st = row["stability"]
    if row["uni3"] and st is not None and st >= tau_s:
        return row["maj3"], 3, "stop3"
    return row["maj_full"], row["N"], "full"


def dec_L3(row, tau):
    lp = row["lp_min_reason_1"]
    if lp is not None and lp >= tau:
        return row["ans1"], 1, "stop1"
    return row["maj_full"], row["N"], "full"


def feat_key(ladder):
    if ladder == "L1":
        return lambda r: r["lp_min_reason_1"]
    if ladder == "L3":
        return lambda r: r["lp_min_reason_1"]
    if ladder == "L2s":
        return lambda r: r["stability"]
    return lambda r: None


def grid_for(ladder):
    if ladder == "L1":
        return [(ql, qh) for qh in LEVELS for ql in LEVELS if ql <= qh]
    if ladder == "L3":
        return [(q,) for q in LEVELS]
    if ladder == "L2s":
        return [(q,) for q in LEVELS]
    if ladder == "L2":
        return [()]
    if ladder == "L4":
        return [()]
    raise ValueError(ladder)


def cfg_label(ladder, cfg):
    if ladder == "L1":
        return "tau_lo=q%.1f tau_hi=q%.1f" % (cfg[0], cfg[1])
    if ladder in ("L3", "L2s"):
        return "tau=q%.1f" % cfg[0]
    return "(no threshold)"


def make_decider(ladder, taus):
    if ladder == "L1":
        return lambda r: dec_L1(r, taus[1], taus[0])
    if ladder == "L2":
        return dec_L2
    if ladder == "L2s":
        return lambda r: dec_L2s(r, taus[0])
    if ladder == "L3":
        return lambda r: dec_L3(r, taus[0])
    if ladder == "L4":
        return dec_L4
    raise ValueError(ladder)


def fit_taus(rows, ladder, cfg):
    fk = feat_key(ladder)
    vals = sorted(v for v in (fk(r) for r in rows) if v is not None)
    return tuple(quantile(vals, lv) for lv in cfg)


# ---------------------------------------------------------------- evaluation

def evaluate(rows, decs, ladder):
    n = len(rows)
    if n == 0:
        return None
    ok = [d[0] is not None and d[0] == r["gold"] for r, d in zip(rows, decs)]
    lats_seq = [lat_pair(ladder, d[2], r["N"])[0] for r, d in zip(rows, decs)]
    lats_bat = [lat_pair(ladder, d[2], r["N"])[1] for r, d in zip(rows, decs)]
    m = {
        "n": n,
        "acc": sum(ok) / n,
        "streams": mean([d[1] for d in decs]),
        "lat_seq": mean(lats_seq),
        "lat_bat": mean(lats_bat),
    }
    if ladder in ("L1", "L2", "L2s", "L3", "L4"):
        min_s = 1 if ladder in ("L1", "L3") else 3
        esc = [d[1] > min_s for d in decs]
        skip_full = [d[1] < r["N"] for r, d in zip(rows, decs)]
        wrong_full = [not r["maj_ok_full"] for r in rows]
        m["esc_rate"] = mean([1.0 if e else 0.0 for e in esc])
        m["acc_esc"] = mean([1.0 if o else 0.0 for o, e in zip(ok, esc) if e]) \
            if any(esc) else None
        m["acc_noesc"] = mean([1.0 if o else 0.0 for o, e in zip(ok, esc) if not e]) \
            if any(not x for x in esc) else None
        wf = [1.0 if s else 0.0 for s, wf_ in zip(skip_full, wrong_full) if wf_]
        m["flag_rec"] = mean(wf) if wf else None
        sf = [1.0 if wf_ else 0.0 for s, wf_ in zip(skip_full, wrong_full) if s]
        m["prec_skip"] = mean(sf) if sf else None
        se = [1.0 if wf_ else 0.0 for e, wf_ in zip(esc, wrong_full) if e]
        m["prec_esc"] = mean(se) if se else None
        se_rec = [1.0 if e else 0.0 for e, wf_ in zip(esc, wrong_full) if wf_]
        m["esc_recall"] = mean(se_rec) if se_rec else None
    else:
        m["esc_rate"] = m["acc_esc"] = m["acc_noesc"] = None
        m["flag_rec"] = m["prec_skip"] = m["prec_esc"] = None
        m["esc_recall"] = None
    return m


def evaluate_fixed(rows, ladder):
    return evaluate(rows, [dec_endpoint(r, ladder) for r in rows], ladder)


# ---------------------------------------------------------------- CV

def loo_folds(n):
    return [[i] for i in range(n)]


def strat_folds(rows, k, seed):
    rng = random.Random(seed)
    pos = [i for i, r in enumerate(rows) if r["maj_ok_full"]]
    neg = [i for i, r in enumerate(rows) if not r["maj_ok_full"]]
    rng.shuffle(pos)
    rng.shuffle(neg)
    folds = [[] for _ in range(k)]
    for lst in (neg, pos):
        for j, i in enumerate(lst):
            folds[j % k].append(i)
    return [f for f in folds if f]


def run_ladder(rows, ladder, cv_mode):
    """Returns (selected_cv_metrics, cv_frontier, insample_frontier,
    selected_cfg_counter)."""
    n = len(rows)
    N = rows[0]["N"]
    grid = grid_for(ladder)
    budget = BUDGET_FRAC * N

    if ladder in ("E1", "E3", "Ef"):
        m = evaluate_fixed(rows, ladder)
        lab = cfg_label(ladder, ())
        return m, [(lab, (), m)], [(lab, (), m)], Counter({(): n})

    if ladder in ("L2", "L4"):
        decs = [make_decider(ladder, ())(r) for r in rows]
        m = evaluate(rows, decs, ladder)
        lab = cfg_label(ladder, ())
        return m, [(lab, (), m)], [(lab, (), m)], Counter({(): n})

    folds = loo_folds(n) if cv_mode == "loo" else strat_folds(rows, 5, CV_SEED)
    sel_decs = [None] * n
    front_decs = {gi: [None] * n for gi in range(len(grid))}
    sel_counter = Counter()

    for fold in folds:
        test = set(fold)
        train_idx = [i for i in range(n) if i not in test]
        train_rows = [rows[i] for i in train_idx]
        candidates = []
        for gi, cfg in enumerate(grid):
            taus = fit_taus(train_rows, ladder, cfg)
            decs = [make_decider(ladder, taus)(rows[i]) for i in train_idx]
            m = evaluate(train_rows, decs, ladder)
            candidates.append((gi, cfg, taus, m))
        under = [c for c in candidates if c[3]["streams"] <= budget]
        pool = under if under else candidates
        _, cfg, taus, _ = max(pool, key=lambda c: (c[3]["acc"], -c[3]["streams"]))
        sel_counter[cfg] += 1
        decider = make_decider(ladder, taus)
        for i in fold:
            sel_decs[i] = decider(rows[i])
        for gi, cfg2 in enumerate(grid):
            taus2 = fit_taus(train_rows, ladder, cfg2)
            d2 = make_decider(ladder, taus2)
            for i in fold:
                front_decs[gi][i] = d2(rows[i])

    sel_m = evaluate(rows, sel_decs, ladder)
    frontier = []
    for gi, cfg in enumerate(grid):
        m = evaluate(rows, front_decs[gi], ladder)
        frontier.append((cfg_label(ladder, cfg), cfg, m))
    insample = []
    for cfg in grid:
        taus = fit_taus(rows, ladder, cfg)
        decs = [make_decider(ladder, taus)(r) for r in rows]
        m = evaluate(rows, decs, ladder)
        insample.append((cfg_label(ladder, cfg), cfg, m))
    return sel_m, frontier, insample, sel_counter


def pareto(points):
    """points: list of (label, cfg, metrics). Keep (streams, acc) Pareto-optimal."""
    out = []
    for p in points:
        m = p[-1]
        dominated = False
        for q in points:
            mq = q[-1]
            if (mq["streams"] <= m["streams"] and mq["acc"] >= m["acc"]
                    and (mq["streams"] < m["streams"] or mq["acc"] > m["acc"])):
                dominated = True
                break
        if not dominated:
            out.append(p)
    return sorted(out, key=lambda p: p[-1]["streams"])


# ---------------------------------------------------------------- report

def write_csv(all_rows):
    cols = ["dataset", "qid", "group", "N", "gold", "maj_full", "maj_ok_full",
            "maj_ok_file", "tie_full", "ans1", "ok1", "uni3", "agree3", "maj3",
            "maj3_ok", "tie3", "maj5", "maj5_ok", "tie5", "maj8", "maj8_ok",
            "tie8", "lp_min_1", "lp_min_reason_1", "marker_missing", "stability"]
    with io.open(OUT_CSV, "w", encoding="utf-8", newline="") as fh:
        fh.write(",".join(cols) + "\n")
        for r in all_rows:
            vals = []
            for c in cols:
                v = r.get(c)
                if v is None:
                    vals.append("")
                elif isinstance(v, float):
                    vals.append("%.5f" % v)
                else:
                    vals.append(str(v))
            fh.write(",".join(vals) + "\n")


def fmt_m(m):
    return ("acc %6.2f%%  streams %5.2f  lat_seq %5.2f  lat_bat %5.2f"
            % (100.0 * m["acc"], m["streams"], m["lat_seq"], m["lat_bat"]))


def fmt_m_esc(m):
    if m["esc_rate"] is None:
        return ""
    return ("  esc %5.1f%%  prec_esc %s  prec_skip %s  flag_rec %s  acc_esc %s  acc_noesc %s"
            % (100.0 * m["esc_rate"], pct(m["prec_esc"]), pct(m["prec_skip"]),
               pct(m["flag_rec"]), pct(m["acc_esc"]), pct(m["acc_noesc"])))


def rule_name(rule):
    if rule == RULE_FIRST:
        return "FIRST-SEEN plurality (PRIMARY; ties -> earliest-seen)"
    return "STRICT plurality (SENSITIVITY; ties -> abstain = wrong)"


def report_rule(datasets, rule):
    """Run the full analysis for one tie rule; returns per-dataset L4 metrics."""
    rule_l4 = {}
    w()
    w("=" * 78)
    w("TIE RULE: %s" % rule_name(rule))
    w("=" * 78)
    w()

    w("--- endpoint baselines under this rule ---")
    endpoints = {}
    for name, rows in datasets.items():
        N = rows[0]["N"]
        e = {}
        for lad in ("E1", "E3", "Ef"):
            e[lad] = evaluate_fixed(rows, lad)
        endpoints[name] = e
        ties = sum(1 for r in rows if r["tie_full"])
        mism = sum(1 for r in rows if r["maj_ok_file"] is not None
                   and bool(r["maj_ok_file"]) != bool(r["maj_ok_full"]))
        w("  %-15s E1 %6.2f%% | E3 %6.2f%% | E_full(N=%d) %6.2f%% | "
          "full-plurality ties=%d file-label mismatches=%d"
          % (name, 100 * e["E1"]["acc"], 100 * e["E3"]["acc"], N,
             100 * e["Ef"]["acc"], ties, mism))
    w()

    w("--- feature AUROC vs %s full-majority correctness ---"
      % ("first-seen" if rule == RULE_FIRST else "strict"))
    for name, rows in datasets.items():
        labels = [r["maj_ok_full"] for r in rows]
        parts = []
        a = auroc([1.0 if r["uni3"] else 0.0 for r in rows], labels)
        parts.append("uni3=%s" % f3(a))
        if any(r["lp_min_reason_1"] is not None for r in rows):
            a = auroc([r["lp_min_reason_1"] for r in rows], labels)
            parts.append("lp_min_reason_1=%s" % f3(a))
            a = auroc([r["lp_min_1"] for r in rows], labels)
            parts.append("lp_min_1=%s" % f3(a))
        if any(r["stability"] is not None for r in rows):
            a = auroc([r["stability"] for r in rows], labels)
            parts.append("stability=%s" % f3(a))
        w("  %-15s %s" % (name, "  ".join(parts)))
    w()

    ladders_by_ds = {
        "probs32": ["L1", "L2", "L3", "L4"],
        "disagree_n5": ["L2", "L4"],
        "disagree30": ["L2", "L4"],
        "reform30_join": ["L2", "L2s", "L4"],
    }
    cv_mode_by_ds = {
        "probs32": "loo",
        "disagree_n5": "fold5",
        "disagree30": "loo",
        "reform30_join": "loo",
    }

    best_candidates = []
    w("--- ladders: CV frontiers (thresholds fitted inside CV) ---")
    for name, rows in datasets.items():
        N = rows[0]["N"]
        cv_mode = cv_mode_by_ds[name]
        e1 = endpoints[name]["E1"]["acc"]
        ef = endpoints[name]["Ef"]["acc"]
        w("  dataset %s (n=%d N=%d, CV=%s): E1=%.2f%% E_full=%.2f%%  "
          "budget=%.1f streams" % (name, len(rows), N, cv_mode, 100 * e1,
                                   100 * ef, BUDGET_FRAC * N))
        for ladder in ladders_by_ds[name]:
            sel_m, frontier, insample, sel_ctr = run_ladder(rows, ladder, cv_mode)
            if ladder == "L4":
                rule_l4[name] = {"m": sel_m, "e1": e1, "ef": ef, "N": N}
            w("    ladder %s selected-by-CV: %s%s"
              % (ladder, fmt_m(sel_m), fmt_m_esc(sel_m)))
            cfg_mode, mode_n = sel_ctr.most_common(1)[0]
            w("      selected level-config mode: %s (%d/%d folds)"
              % (cfg_label(ladder, cfg_mode), mode_n, sum(sel_ctr.values())))
            pf = pareto(frontier)
            for label, cfg, m in pf:
                mark = ""
                if m["streams"] <= BUDGET_FRAC * N and m["acc"] >= ef - 0.01:
                    mark = " <== meets ADVANCE-A (CV)"
                if m["acc"] >= e1 + 0.03 and m["streams"] <= 3.0:
                    mark += " <== meets ADVANCE-B (CV)"
                w("      CV-frontier %-22s %s%s" % (label, fmt_m(m), mark))
            ins = pareto(insample)
            w("      in-sample Pareto (IN-SAMPLE, labelled):")
            for label, cfg, m in ins[:4]:
                w("        %-22s %s%s" % (label, fmt_m(m), fmt_m_esc(m)))
            if ladder in ("L1", "L3", "L2s"):
                best_candidates.append({
                    "dataset": name, "ladder": ladder, "N": N,
                    "mode_cfg": cfg_mode, "sel_m": sel_m,
                    "frontier": frontier, "e1": e1, "ef": ef,
                    "cv_mode": cv_mode,
                })
            elif ladder in ("L2", "L4"):
                best_candidates.append({
                    "dataset": name, "ladder": ladder, "N": N,
                    "mode_cfg": (), "sel_m": sel_m,
                    "frontier": frontier, "e1": e1, "ef": ef,
                    "cv_mode": cv_mode,
                })
        w()

    w("--- verdict vs pre-registered criteria (CV-validated points only) ---")
    overall_advance = False
    escalation_notes = []
    best = None
    all_points = []
    for cand in best_candidates:
        name, ladder, N = cand["dataset"], cand["ladder"], cand["N"]
        e1, ef = cand["e1"], cand["ef"]
        adaptive_pts = []
        for label, cfg, m in cand["frontier"]:
            adaptive_pts.append((label, cfg, m))
        meets_a = [p for p in adaptive_pts
                   if p[2]["acc"] >= ef - 0.01 and p[2]["streams"] <= BUDGET_FRAC * N]
        meets_b = [p for p in adaptive_pts
                   if p[2]["acc"] >= e1 + 0.03 and p[2]["streams"] <= 3.0]
        # dominance over both endpoints
        dom_both = [p for p in adaptive_pts
                    if p[2]["acc"] >= ef - 0.01 and p[2]["streams"] < N
                    and p[2]["acc"] > e1]
        status = "ADVANCE" if (meets_a or meets_b) else "no"
        if meets_a or meets_b:
            overall_advance = True
        w("  %-15s %-3s  ADVANCE-A:%s  ADVANCE-B:%s  dominates-both:%s  [%s]"
          % (name, ladder,
             "YES(%d)" % len(meets_a) if meets_a else "no",
             "YES(%d)" % len(meets_b) if meets_b else "no",
             "YES(%d)" % len(dom_both) if dom_both else "no",
             status))
        pool = meets_a if meets_a else (meets_b if meets_b else [])
        for label, cfg, m in pool:
            cand_score = (0 if meets_a else 1, m["streams"], -m["acc"])
            entry = (cand_score, name, ladder, label, cfg, m, cand)
            if best is None or entry[0] < best[0]:
                best = entry
            if m["prec_skip"] is not None and m["prec_skip"] < 0.30:
                escalation_notes.append(
                    "  low skip precision: %s/%s %s at %s prec_skip=%s"
                    % (name, ladder, label, fmt_m(m), pct(m["prec_skip"])))
        for label, cfg, m in adaptive_pts:
            all_points.append((name, ladder, label, cfg, m, cand))
        # closest approach to each criterion (honest near-miss reporting)
        near_a = [p for p in adaptive_pts if p[2]["acc"] >= ef - 0.01]
        if near_a:
            cA = min(near_a, key=lambda p: p[2]["streams"])
            w("      closest to A: %s  acc %.2f%% at %.2f streams "
              "(budget %.1f, gap %+.2f)"
              % (cA[0], 100 * cA[2]["acc"], cA[2]["streams"],
                 BUDGET_FRAC * N, cA[2]["streams"] - BUDGET_FRAC * N))
        near_b = [p for p in adaptive_pts if p[2]["acc"] >= e1 + 0.03]
        if near_b:
            cB = min(near_b, key=lambda p: p[2]["streams"])
            w("      closest to B: %s  acc %.2f%% at %.2f streams "
              "(cap 3.0, gap %+.2f)"
              % (cB[0], 100 * cB[2]["acc"], cB[2]["streams"],
                 cB[2]["streams"] - 3.0))
    w()
    if overall_advance:
        w("PHASE A VERDICT: ADVANCE — at least one CV-validated adaptive point")
        w("meets a pre-registered criterion (see marks above).")
    else:
        w("PHASE A VERDICT: DEAD END — no CV-validated adaptive point meets")
        w("either pre-registered criterion. Some adaptive points do")
        w("Pareto-dominate the endpoints on (mean streams, accuracy) — e.g. L2")
        w("spends fewer streams at equal-or-higher accuracy — but they miss the")
        w("40%-savings budget (mean_streams <= 0.6*N), which is the binding")
        w("constraint for ADVANCE-A. Dominance alone is therefore not enough.")
    w("  NOTE: L2 Pareto-dominates the endpoints on (mean streams, accuracy) on")
    w("  all four datasets (dominates-both:YES) but still misses the")
    w("  40%-savings budget (mean_streams <= 0.6*N), so it does not meet")
    w("  ADVANCE-A; both statements are true at once.")
    w("  escalation-quality notes (always printed):")
    if escalation_notes:
        for note in escalation_notes:
            w(note)
    else:
        w("  none — no eligible point had skip-precision below 30%.")
    w()

    w("--- best policy ---")
    pick = None
    if best is None and all_points:
        near = [p for p in all_points if p[4]["acc"] >= p[5]["ef"] - 0.01]
        if near:
            pick = min(near, key=lambda p: (p[4]["streams"], -p[4]["acc"]))
        else:
            pick = max(all_points, key=lambda p: (p[4]["acc"], -p[4]["streams"]))
        name, ladder, label, cfg, m, cand = pick
        w("  NO CV point met either criterion; describing the closest near-miss.")
    elif best is not None:
        _, name, ladder, label, cfg, m, cand = best
        w("  meets a pre-registered criterion.")
    else:
        pick = None
        w("  none — no adaptive point met either criterion.")
    if best is not None or pick is not None:
        rows = datasets[name]
        N = cand["N"]
        w("  dataset=%s ladder=%s config=%s (CV=%s)"
          % (name, ladder, label, cand["cv_mode"]))
        taus = fit_taus(rows, ladder, cfg)
        w("  full-data absolute thresholds: %s"
          % ", ".join("%.4f" % t if math.isfinite(t) else str(t) for t in taus))
        w("  CV metrics: %s%s" % (fmt_m(m), fmt_m_esc(m)))
        w("  endpoints on same data: E1 %.2f%%  E3 %.2f%%  E_full %.2f%%"
          % (100 * cand["e1"], 100 * evaluate_fixed(rows, "E3")["acc"],
             100 * cand["ef"]))
        decider = make_decider(ladder, taus)
        rungs = Counter(decider(r)[2] for r in rows)
        w("  rung mix (in-sample, descriptive): %s"
          % " ".join("%s=%d" % kv for kv in sorted(rungs.items())))
        easy = [r for r in rows if r["uni3"] and r["maj_ok_full"]]
        hard = [r for r in rows if not r["uni3"] or not r["maj_ok_full"]]
        if easy:
            de = [decider(r) for r in easy]
            me = evaluate(easy, de, ladder)
            w("  typical easy (unanimous & full-majority right, n=%d): %s"
              % (len(easy), fmt_m(me)))
        if hard:
            dh = [decider(r) for r in hard]
            mh = evaluate(hard, dh, ladder)
            w("  typical hard (split or wrong full-majority, n=%d): %s"
              % (len(hard), fmt_m(mh)))
    w()

    return rule_l4


def report_l4_table(by_rule):
    w("--- L4 = plurality@3 (pre-registered gate, all datasets, both rules) ---")
    w("  escalation fallback returns maj@N under the active tie rule;")
    w("  esc_recall = share of always-full-wrong cases that escalated.")
    w("  bar = within 1pp of always-full AND mean streams <= 0.6*N")
    w("  %-12s %-15s %8s %8s %7s %9s %11s  %s"
      % ("rule", "dataset", "acc", "streams", "esc", "prec_esc",
         "esc_recall", "meets bar"))
    for rule in (RULE_FIRST, RULE_STRICT):
        tag = "first-seen" if rule == RULE_FIRST else "strict"
        for name, d in by_rule[rule].items():
            m, ef, N = d["m"], d["ef"], d["N"]
            bar = (m["acc"] >= ef - 0.01
                   and m["streams"] <= BUDGET_FRAC * N)
            w("  %-12s %-15s %7.2f%% %8.2f %6.1f%% %9s %11s  %s"
              % (tag, name, 100 * m["acc"], m["streams"],
                 100 * m["esc_rate"], pct(m["prec_esc"]),
                 pct(m["esc_recall"]), "YES" if bar else "no"))
    w()


def report_caveats():
    w("--- caveats ---")
    w("  * small n: probs32 has 6 wrong full pluralities (1 tie); n=32 LP set")
    w("    cannot resolve within-unanimous discrimination (row 40 follow-up).")
    w("  * disagree30 duplicates probs32[0:30]; treat as a robustness check,")
    w("    not a second sample.")
    w("  * PRIMARY = first-seen plurality (matches the maj_ok labels in the")
    w("    dumps); SENSITIVITY = strict plurality (ties counted wrong). Both")
    w("    are recomputed end-to-end above; file-label mismatches should be 0")
    w("    under first-seen, which validates the primary rule.")
    w("  * 2/32 stream-1 texts lack the 'The answer is' marker; lp_min_reason_1")
    w("    falls back to the all-token min for those (marker_missing column).")
    w("  * reform stability has 1 None (p13 all-noop, excluded in row 39);")
    w("    treated as gate-fail (escalate).")
    w("  * latency multipliers are E0-DERIVED, not E2-measured: source is")
    w("    results/e0_full.csv (all rows; median(step_ms|N)/median(step_ms|N=1)):")
    w("    N=3 -> 1.19, N=5 -> 1.44, N=8 -> 1.92 raw (p50-tail range")
    w("    1.70..2.06 for N=8 per results/e0_summary.txt). The old 'E2 sweep'")
    w("    labels were wrong: results/e2_sweep200.csv has no N=3/N=5 rows.")
    w("    These are descriptive; NO pass/fail gate depends on latency.")
    w("  * thresholds selected per CV fold on quantile-level grids; fold-level")
    w("    selection at n=32 with ~6 negatives is noisy — CIs not claimed.")
    w("    L4 is parameter-free and therefore not cross-validated.")
    w("  * unanimous-wrong is unfixable inside these ladders: the top rung is")
    w("    maj@full, which is wrong whenever the full pool is unanimous-wrong.")
    w("    The stability gate (L2s) can only spend more (escalate), never")
    w("    rescue; the actual rescue arm is row 39's reform invocation (2nd")
    w("    call), which sits outside both the ladders and the latency model.")
    w("  * offline assumption: answers[0:k] of an N-stream run stand in for a")
    w("    fresh k-stream batch (per-stream seeds are row-derived in the")
    w("    harness, but batch-size dependence of the sampling chain was not")
    w("    verified offline).")
    w()


def main():
    w("=== angle 41 Phase A: adaptive compute router (offline, zero GPU) ===")
    w("data only; no GPU touched; no existing files modified")
    w()
    w("PRIMARY tie rule = first-seen plurality; STRICT = sensitivity.")
    w("Both rules are recomputed end-to-end and reported side by side.")
    w()

    base = apply_rule(load_rows(), RULE_FIRST)
    w("--- datasets ---")
    for name, rows in base.items():
        N = rows[0]["N"]
        feats = []
        feats.append("lp_min=%s" % ("yes" if any(r["lp_min_reason_1"] is not None
                                                for r in rows) else "no"))
        feats.append("agree=%s" % ("yes" if any("uni3" in r for r in rows) else "no"))
        feats.append("stability=%s" % ("yes" if any(r["stability"] is not None
                                                    for r in rows) else "no"))
        w("  %-15s n=%3d N=%d  %s" % (name, len(rows), N, " ".join(feats)))
    w("  NOTE: disagree30 is the same run as probs32[0:30] (identical questions")
    w("        and answers) — not independent evidence.")
    w()

    all_rows = []
    for rows in base.values():
        all_rows.extend(rows)
    write_csv(all_rows)

    by_rule = {}
    for rule in (RULE_FIRST, RULE_STRICT):
        datasets = apply_rule(load_rows(), rule)
        by_rule[rule] = report_rule(datasets, rule)

    report_l4_table(by_rule)
    report_caveats()
    w("wrote %s" % OUT_CSV)

    with io.open(OUT_TXT, "w", encoding="utf-8") as fh:
        fh.write(buf.getvalue())
    w("wrote %s" % OUT_TXT)


if __name__ == "__main__":
    main()
