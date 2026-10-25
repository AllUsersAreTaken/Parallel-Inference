#!/usr/bin/env python3
"""Row 41 analysis: does cross-stream last-layer representation consensus
separate correct from wrong runs better than output agreement?

Input:
  results/hidden_probe30.json        run records (same 32 runs as row 40)
  data/_hid_pNN.jsonl                hidden sidecar: k=h (per stream step),
                                     k=c (per-step consensus)
  data/_hid_pNN.json                 stream texts (phase split)
  data/_probs_pNN.jsonl              row-40 offsets, used only to locate the
                                     "The answer is" marker per token step
  results/probs_features.csv         row-40 lp_min baseline, joined by qid

The 32 hidden runs are token-for-token identical to row 40 (verified: same
answers, same per-stream texts, same generated counts), so the probs offsets
map one-to-one onto the hidden consensus steps (consensus step t == probs
step t+1 == the row after t+1 emitted tokens).

Output: printed report and results/hidden_analysis.txt
"""
import io
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "results", "hidden_probe30.json")
PROBS_FEATS = os.path.join(ROOT, "results", "probs_features.csv")
OUT_TXT = os.path.join(ROOT, "results", "hidden_analysis.txt")

buf = io.StringIO()


def w(s=""):
    print(s)
    buf.write(str(s) + "\n")


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
    n1, n0 = len(pos), len(neg)
    return (sr - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def med(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2.0


def pct_of(values, v):
    """Percent of values strictly below v (ties count half)."""
    if not values:
        return None
    lo = sum(1 for x in values if x < v)
    eq = sum(1 for x in values if x == v)
    return 100.0 * (lo + 0.5 * eq) / len(values)


def std(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def pearson(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    mx = sum(x for x, _ in pairs) / len(pairs)
    my = sum(y for _, y in pairs) / len(pairs)
    sx = math.sqrt(sum((x - mx) ** 2 for x, _ in pairs))
    sy = math.sqrt(sum((y - my) ** 2 for _, y in pairs))
    if sx == 0 or sy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in pairs) / (sx * sy)


def read_sidecar(path):
    hrows, crows = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            (hrows if o["k"] == "h" else crows).append(o)
    return hrows, crows


def answer_step(probs_rows, text):
    """First 1-based token step whose byte offset reaches the answer marker."""
    marker = text.encode("utf-8").rfind(b"The answer is")
    if marker < 0:
        return None
    for r in sorted(probs_rows, key=lambda x: x["step"]):
        if r["off"] >= marker:
            return r["step"]
    return None


def run_features(qid, rec, hrows, crows, probs_rows_by_s, streams, nstreams):
    cons = sorted(crows, key=lambda x: x["step"])
    cos = [c["cos_mean"] for c in cons]
    n = len(cos)
    f = {}
    f["cos_mean_all"] = mean(cos)
    f["cos_min_all"] = min((c["cos_min"] for c in cons), default=None)
    f["cos_max_all"] = max((c["cos_max"] for c in cons), default=None)
    f["cos_std"] = std(cos)
    half = n // 2
    if half >= 2:
        f["slope"] = mean(cos[half:]) - mean(cos[:half])
    else:
        f["slope"] = None
    if n:
        amin = min(range(n), key=lambda i: cos[i])
        f["argmin_step_rel"] = cons[amin]["step"] / float(n)
    else:
        f["argmin_step_rel"] = None

    # Phase split: a consensus step t is answer-phase when at least half of the
    # live streams (generated > t) have reached their answer marker by then.
    gen = {st["id"]: st["generated"] for st in streams}
    astep = {}
    for st in streams:
        rs = probs_rows_by_s.get(st["id"], [])
        astep[st["id"]] = answer_step(rs, st["text"])
    rsn, ans = [], []
    for c in cons:
        live = [s for s, g in gen.items() if g > c["step"]]
        if not live:
            continue
        hit = sum(1 for s in live if astep.get(s) is not None and astep[s] <= c["step"] + 1)
        (ans if hit * 2 >= len(live) else rsn).append(c["cos_mean"])
    f["cos_mean_rsn"] = mean(rsn)
    f["cos_mean_ans"] = mean(ans)
    f["n_rsn_steps"] = len(rsn)
    f["n_ans_steps"] = len(ans)

    # Composition control: steps where every stream is still live (n == 8).
    # Later steps only compare the streams that have not finished yet.
    n8 = [c for c in cons if c["n"] == nstreams]
    f["cos_mean_n8"] = mean([c["cos_mean"] for c in n8])
    f["cos_min_n8"] = min((c["cos_min"] for c in n8), default=None)
    f["n8_steps"] = len(n8)

    # Stream level: mean cosine of this stream to the other live streams.
    by_s = {}
    for r in hrows:
        by_s.setdefault(r["s"], []).append(r)
    stream_feats = {}
    for s, rs in by_s.items():
        vals = [r["cmean"] for r in rs if r["cmean"] is not None and r["cmean"] >= 0.0]
        stream_feats[s] = mean(vals)

    # Covariate: pairwise token-path agreement over the common prefix. This is
    # the confound the consensus metric may simply be tracking.
    toks = {s: [r["tok"] for r in sorted(rs, key=lambda x: x["step"])]
            for s, rs in probs_rows_by_s.items()}
    keys = sorted(toks)
    tpair = []
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            ta, tb = toks[keys[a]], toks[keys[b]]
            m = min(len(ta), len(tb))
            if m:
                tpair.append(sum(1 for k in range(m) if ta[k] == tb[k]) / m)
    f["token_agree"] = mean(tpair)
    return f, stream_feats


FEATS = ["cos_mean_all", "cos_min_all", "cos_max_all", "slope",
         "argmin_step_rel", "cos_mean_rsn", "cos_mean_ans", "cos_std",
         "cos_mean_n8", "cos_min_n8"]
MAIN = ["cos_mean_all", "cos_min_all", "slope"]


def main():
    meta = json.load(open(BASE, encoding="utf-8"))
    recs = meta["records"]

    lp_min = {}
    with open(PROBS_FEATS, encoding="utf-8") as fh:
        hdr = fh.readline().strip().split(",")
        for line in fh:
            cols = line.rstrip("\n").split(",")
            row = dict(zip(hdr, cols))
            if row.get("lp_min") not in (None, "", "None"):
                lp_min[int(row["qid"])] = float(row["lp_min"])

    runs = []
    streams_lvl = []
    for rec in recs:
        qid = rec["qid"]
        with open(os.path.join(ROOT, "data", rec["hidden_json"]), encoding="utf-8") as fh:
            streams = json.load(fh)["streams"]
        hrows, crows = read_sidecar(os.path.join(ROOT, "data", rec["hidden_jsonl"]))
        probs_rows_by_s = {}
        ppath = os.path.join(ROOT, "data", "_probs_p%02d.jsonl" % qid)
        if os.path.exists(ppath):
            with open(ppath, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        o = json.loads(line)
                        probs_rows_by_s.setdefault(o["s"], []).append(o)
        f, sfeats = run_features(qid, rec, hrows, crows, probs_rows_by_s,
                                 streams, meta.get("n_streams", 8))
        f.update({"qid": qid, "group": rec["group"], "maj_ok": rec["maj_ok"],
                  "agree": rec["pairwise_agree"], "src": rec["src"],
                  "gold": rec["gold"], "maj": rec["maj"],
                  "n_cons": len(crows), "n_h": len(hrows),
                  "lp_min": lp_min.get(qid)})
        runs.append(f)

        answers = rec["answers"]
        gold = rec["gold"]
        for st in streams:
            i = int(st["id"]) - 1
            if i >= len(answers) or sfeats.get(st["id"]) is None:
                continue
            streams_lvl.append({
                "qid": qid, "ok": answers[i] is not None and answers[i] == gold,
                "cmean": sfeats[st["id"]], "group": rec["group"],
            })

    w("=== row 41 hidden-state consensus analysis ===")
    w("runs=%d  streams=%d  hidden steps=%d  consensus steps=%d"
      % (len(runs), sum(len(r["answers"]) for r in recs),
         sum(r["n_h"] for r in runs), sum(r["n_cons"] for r in runs)))
    groups = {"UR": [], "SP": [], "UW": []}
    for r in runs:
        groups[r["group"]].append(r)
    w("groups: " + " ".join("%s=%d" % (k, len(v)) for k, v in groups.items()))
    w("capture: last-layer embedding after final norm, per emitted-token row;")
    w("consensus = mean/min/max pairwise cosine over live streams at that step.")
    w("phase split uses row-40 probs offsets (the two runs are token-identical).")
    w()

    w("--- group means (run-level features) ---")
    w("%-16s %12s %12s %12s" % ("feature", "UR (n=%d)" % len(groups["UR"]),
                                "SP (n=%d)" % len(groups["SP"]),
                                "UW (n=%d)" % len(groups["UW"])))
    for name in FEATS + ["token_agree", "n_cons", "n8_steps"]:
        vals = []
        for g in ("UR", "SP", "UW"):
            v = [r[name] for r in groups[g] if r.get(name) is not None]
            vals.append(sum(v) / len(v) if v else None)
        w("%-16s %12s %12s %12s"
          % (name, *["  n/a" if v is None else "%.4f" % v for v in vals]))
    w()

    labels = [r["maj_ok"] for r in runs]
    agree_auc = auroc([r["agree"] for r in runs], labels)
    lp_auc = auroc([r["lp_min"] for r in runs], labels)
    w("--- all %d runs: AUROC -> maj_ok (higher score -> correct) ---" % len(runs))
    w("%-16s %8s %8s  %s" % ("feature", "auc", "oriented", "direction"))
    w("%-16s %8.3f %8.3f  %s" % ("pairwise_agree", agree_auc,
                                  max(agree_auc, 1 - agree_auc),
                                  "high" if agree_auc >= 0.5 else "low"))
    w("%-16s %8.3f %8.3f  %s" % ("lp_min", lp_auc, max(lp_auc, 1 - lp_auc),
                                  "high" if lp_auc >= 0.5 else "low"))
    ta_auc = auroc([r["token_agree"] for r in runs], labels)
    w("%-16s %8.3f %8.3f  %s" % ("token_agree", ta_auc, max(ta_auc, 1 - ta_auc),
                                  "high" if ta_auc >= 0.5 else "low"))
    all_auc = {}
    for name in FEATS:
        a = auroc([r[name] for r in runs], labels)
        if a is None:
            continue
        all_auc[name] = a
        w("%-16s %8.3f %8.3f  %s" % (name, a, max(a, 1 - a),
                                      "high" if a >= 0.5 else "low"))
    w()

    sp = [r for r in runs if r["group"] == "SP"]
    spl = [r["maj_ok"] for r in sp]
    w("--- within-SP only (n=%d, agree<1) ---" % len(sp))
    w("%-16s %8s %8s" % ("feature", "auc", "oriented"))
    for name, vals in [("pairwise_agree", [r["agree"] for r in sp]),
                       ("lp_min", [r["lp_min"] for r in sp]),
                       ("token_agree", [r["token_agree"] for r in sp])] + \
                      [(n, [r[n] for r in sp]) for n in FEATS]:
        a = auroc(vals, spl)
        if a is None:
            continue
        w("%-16s %8.3f %8.3f" % (name, a, max(a, 1 - a)))
    w()

    w("--- DECISIVE: within-unanimous (agreement constant = 1.0) ---")
    w("UR=%d UW=%d (single UW run: percentile report, not a test)"
      % (len(groups["UR"]), len(groups["UW"])))
    for uw in groups["UW"]:
        w("UW qid=%d src=%s gold=%s maj=%s n_cons=%d"
          % (uw["qid"], uw["src"], uw["gold"], uw["maj"], uw["n_cons"]))
        for name in FEATS + ["token_agree"]:
            vals = [r[name] for r in groups["UR"] if r.get(name) is not None]
            v = uw.get(name)
            if v is None or not vals:
                continue
            p = pct_of(vals, v)
            edge = "BOTTOM-decile" if p <= 10 else ("TOP-decile" if p >= 90 else "")
            w("  %-16s UW=%9.4f UR median=%9.4f -> %.0fth pct %s"
              % (name, v, med(vals), p, edge))
    w()

    w("--- confound: is consensus just token-path agreement? ---")
    c_all = [r["cos_mean_all"] for r in runs]
    c_min = [r["cos_min_all"] for r in runs]
    t_all = [r["token_agree"] for r in runs]
    w("  pearson(cos_mean_all, token_agree) = %.3f over %d runs"
      % (pearson(c_all, t_all), len(runs)))
    w("  pearson(cos_min_all,  token_agree) = %.3f"
      % pearson(c_min, t_all))
    w("  token_agree -> maj_ok AUC=%.3f; within-SP AUC=%.3f"
      % (ta_auc, auroc([r["token_agree"] for r in sp], spl)))
    w("  pearson(cos_mean_all, n_cons) = %.3f (long n=2 tails inflate the mean)"
      % pearson(c_all, [r["n_cons"] for r in runs]))
    w("  among unanimous runs: UR median token_agree=%.3f, UW=%.3f"
      % (med([r["token_agree"] for r in groups["UR"]]),
         groups["UW"][0]["token_agree"] if groups["UW"] else float("nan")))
    ur = groups["UR"]
    w("  within UR only (n=%d): pearson(cos_mean_all, token_agree) = %.3f"
      % (len(ur), pearson([r["cos_mean_all"] for r in ur],
                          [r["token_agree"] for r in ur])))
    if groups["UW"]:
        uw = groups["UW"][0]
        for name in ("cos_mean_all", "cos_min_all"):
            vals = [r[name] for r in ur if r.get(name) is not None]
            sd = std(vals)
            z = ((uw[name] - mean(vals)) / sd) if sd else float("nan")
            w("  UW %s = %.4f vs UR mean %.4f (sd %.4f) -> z=%+.2f"
              % (name, uw[name], mean(vals), sd, z))
    w()

    ok = [s for s in streams_lvl if s["ok"]]
    bad = [s for s in streams_lvl if not s["ok"]]
    sa = auroc([s["cmean"] for s in streams_lvl], [s["ok"] for s in streams_lvl])
    w("--- stream level: mean cosine-to-others -> that stream's correctness ---")
    w("n=%d (pos=%d neg=%d), pooled AUC=%.3f oriented=%.3f; clustered in runs, "
      "not iid" % (len(streams_lvl), len(ok), len(bad), sa, max(sa, 1 - sa)))
    per_run = []
    for qid in sorted({s["qid"] for s in streams_lvl}):
        rs = [s for s in streams_lvl if s["qid"] == qid]
        if len({s["ok"] for s in rs}) < 2:
            continue
        a = auroc([s["cmean"] for s in rs], [s["ok"] for s in rs])
        if a is not None:
            per_run.append(a)
    w("  runs with both classes: %d, mean within-run AUC=%.3f (oriented %.3f)"
      % (len(per_run), mean(per_run) if per_run else float("nan"),
         mean([max(a, 1 - a) for a in per_run]) if per_run else float("nan")))
    w("  mean cmean correct=%.4f wrong=%.4f"
      % (mean([s["cmean"] for s in ok]), mean([s["cmean"] for s in bad])))
    w()

    # ---------------- pre-stated gate ----------------
    w("=== pre-stated success / dead-end gate ===")
    agree_or = max(agree_auc, 1 - agree_auc)
    cands = {n: all_auc.get(n) for n in MAIN}
    best = None
    for n, a in cands.items():
        if a is None:
            continue
        o = max(a, 1 - a)
        if best is None or o > best[1]:
            best = (n, o, a)
    if best:
        w("best representation feature: %s auc=%.3f oriented=%.3f "
          "(agreement oriented=%.3f, margin=%+.3f)"
          % (best[0], best[2], best[1], agree_or, best[1] - agree_or))
    # UW decile count
    uw_edges = 0
    uw_edge_names = []
    for uw in groups["UW"]:
        for name in FEATS + ["token_agree"]:
            vals = [r[name] for r in groups["UR"] if r.get(name) is not None]
            v = uw.get(name)
            if v is None or not vals:
                continue
            p = pct_of(vals, v)
            if p <= 10 or p >= 90:
                uw_edges += 1
                uw_edge_names.append((name, p))
                w("  UW edge: %-16s at %.0fth pct" % (name, p))
    auc_adv = bool(best and best[1] >= 0.85 and (best[1] - agree_or) >= 0.05)
    uw_adv = uw_edges >= 2
    nearly_constant = (mean([r["cos_std"] for r in runs]) is not None and
                       (mean([r["cos_std"] for r in runs]) < 0.01))
    w("  criterion A (oriented AUC >= 0.85 and >= agreement+0.05): %s"
      % ("MET" if auc_adv else "NOT MET"))
    w("  criterion B (UW in bottom/top decile of UR on >= 2 features): %s"
      % ("MET (%d features)" % uw_edges if uw_adv else "NOT MET"))
    if auc_adv:
        verdict = "ADVANCE on criterion A"
    elif uw_adv:
        verdict = ("INCONCLUSIVE: criterion B fires on the single UW run only; "
                   "primary AUC criterion fails")
    elif best and best[1] <= agree_or:
        verdict = "DEAD (primary AUC <= agreement)"
    elif nearly_constant:
        verdict = "DEAD (consensus nearly constant)"
    else:
        verdict = "WEAK"
    w("verdict: %s" % verdict)
    w("  pre-stated rule: ADVANCE if (oriented AUC >= 0.85 and beats agreement")
    w("  by >= 0.05) or (UW in the bottom/top decile of UR on >= 2 features)")
    w("  primary AUC result: %s" % ("passes" if auc_adv else
      "fails (oriented %.3f <= agreement %.3f)" % (best[1], agree_or) if best else "n/a"))
    w("  mean within-run std(cos_mean) = %.5f -> %s"
      % (mean([r["cos_std"] for r in runs]),
         "nearly constant" if nearly_constant else "varies"))
    w()
    w("caveats: n=32 runs / 12 UR / 1 UW; one model, one seed (per-stream")
    w("seeds 1234..1241), one machine; embeddings are post-final-norm and")
    w("identical-text streams sit at ~0.999 (batch numerics), compressing the")
    w("top of the scale; consensus compares the same token index even when the")
    w("streams' texts have diverged; stream-level samples are clustered in runs.")
    w("the UW-decile clause rests on 1 run, and with n_UR=12 the '90th pct' is")
    w("the top 1-2 values, so it is an anecdote, not a separation.")

    with io.open(OUT_TXT, "w", encoding="utf-8") as fh:
        fh.write(buf.getvalue())
    w("wrote %s" % OUT_TXT)


if __name__ == "__main__":
    main()
