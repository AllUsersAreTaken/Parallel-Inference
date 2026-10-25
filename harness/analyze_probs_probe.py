#!/usr/bin/env python3
"""Row 40 analysis: do raw-logit probability shapes distinguish wrong answers
beyond agreement, especially within the unanimous group?

Input:  results/probs_probe30.json + data/_probs_pNN.{json,jsonl}
Output: printed report, results/probs_analysis.txt, results/probs_features.csv

Segments: tokens at byte offset >= last b"The answer is" = answer, else reasoning.
All stats are PRE-sampler full-vocab raw-logit quantities (nats).
"""
import io
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "results", "probs_probe30.json")
OUT_TXT = os.path.join(ROOT, "results", "probs_analysis.txt")
OUT_CSV = os.path.join(ROOT, "results", "probs_features.csv")

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


def med(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2.0


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def std(xs):
    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def stream_feats(recs, tb):
    """Features for one stream's step-ordered records over its byte text."""
    if not recs:
        return None, 0
    marker = tb.rfind(b"The answer is")
    has_ans = marker >= 0
    rsn = [r for r in recs if not has_ans or r["off"] < marker]
    ans = [r for r in recs if has_ans and r["off"] >= marker]
    H = [r["H"] for r in recs]
    lp = [r["lp"] for r in recs]
    f = {
        "H_all": mean(H),
        "H_p95": sorted(H)[int(0.95 * (len(H) - 1))],
        "H_max": max(H),
        "frac_unc": sum(1 for h in H if h > 1.0) / len(H),
        "lp_all": mean(lp),
        "lp_min": min(lp),
        "early_lp": mean(lp[:40]),
        "H_rsn": mean([r["H"] for r in rsn]),
        "lp_rsn": mean([r["lp"] for r in rsn]),
        "H_ans": mean([r["H"] for r in ans]) if ans else None,
        "p1_ans": mean([r["p1"] for r in ans]) if ans else None,
        "mg_ans": mean([r["mg"] for r in ans]) if ans else None,
        "lp_ans": mean([r["lp"] for r in ans]) if ans else None,
        "n_ans_tok": len(ans),
    }
    return f, 1 if has_ans else 0


def main():
    with open(BASE, encoding="utf-8") as fh:
        meta = json.load(fh)
    runs = []
    marker_hits = 0
    total_streams = 0
    identical_pairs, identical_dH = 0, []
    for r in meta["records"]:
        sid = r["qid"]
        with open(os.path.join(ROOT, "data", r["probs_json"]), encoding="utf-8") as fh:
            streams = json.load(fh)["streams"]
        recs_by_s = {}
        with open(os.path.join(ROOT, "data", r["probs_jsonl"]), encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    o = json.loads(line)
                    recs_by_s.setdefault(o["s"], []).append(o)
        sfeats, texts, lp_ans_list, H_ans_list, p1_ans_list = [], [], [], [], []
        for i, st in enumerate(streams):
            total_streams += 1
            key = str(i + 1)
            rs = sorted(recs_by_s.get(key, []), key=lambda x: x["step"])
            tb = st["text"].encode("utf-8")
            f, hit = stream_feats(rs, tb)
            marker_hits += hit
            if f:
                sfeats.append(f)
                lp_ans_list.append(f["lp_ans"])
                H_ans_list.append(f["H_ans"])
                p1_ans_list.append(f["p1_ans"])
            texts.append(st["text"])
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                if texts[i] == texts[j]:
                    identical_pairs += 1
                    a = sorted(recs_by_s[str(i + 1)], key=lambda x: x["step"])
                    b = sorted(recs_by_s[str(j + 1)], key=lambda x: x["step"])
                    n = min(len(a), len(b))
                    if n:
                        identical_dH.append(mean(
                            [abs(a[k]["H"] - b[k]["H"]) for k in range(n)]))
        run = {
            "qid": sid, "group": r["group"], "maj_ok": r["maj_ok"],
            "agree": r["pairwise_agree"], "src": r["src"],
            "gold": r["gold"], "maj": r["maj"],
        }
        for name in ("H_all", "H_p95", "frac_unc", "lp_all", "lp_min",
                     "early_lp", "H_rsn", "lp_rsn", "H_ans", "p1_ans",
                     "mg_ans", "lp_ans"):
            run[name] = mean([f[name] for f in sfeats])
        run["ans_conf_disp"] = std(lp_ans_list)
        run["H_ans_disp"] = std(H_ans_list)
        run["n_ans_missing"] = sum(1 for f in sfeats if f["H_ans"] is None)
        run["n_stream_ok"] = len(sfeats)
        runs.append(run)

    feats = ["H_all", "H_p95", "frac_unc", "lp_all", "lp_min", "early_lp",
             "H_rsn", "lp_rsn", "H_ans", "p1_ans", "mg_ans", "lp_ans",
             "ans_conf_disp", "H_ans_disp"]

    w("=== row 40 prob-shape analysis ===")
    w("runs=%d streams=%d answer-marker found in %d/%d streams"
      % (len(runs), total_streams, marker_hits, total_streams))
    w("unanimous-identical text pairs: %d (mean |dH| aligned: %s)"
      % (identical_pairs,
         "n/a" if not identical_dH else "%.4f nats" % mean(identical_dH)))
    groups = {"UR": [], "SP": [], "UW": []}
    for r in runs:
        groups[r["group"]].append(r)
    w("groups: " + " ".join("%s=%d" % (k, len(v)) for k, v in groups.items()))
    w()

    w("--- group medians (run-level features) ---")
    w("%-15s %14s %14s %14s" % ("feature", "UR (n=%d)" % len(groups["UR"]),
                                "SP (n=%d)" % len(groups["SP"]),
                                "UW (n=%d)" % len(groups["UW"])))
    for name in feats:
        vals = [med([r[name] for r in groups[g]]) for g in ("UR", "SP", "UW")]
        w("%-15s %14s %14s %14s"
          % (name, *["  n/a" if v is None else "%.4f" % v for v in vals]))
    w()

    labels = [r["maj_ok"] for r in runs]
    w("--- AUROC -> maj_ok (all %d runs; oriented = max(a,1-a)) ---" % len(runs))
    w("baseline pairwise_agree: %.3f" % (auroc([r["agree"] for r in runs], labels) or 0.0))
    best = (0.0, None)
    for name in feats:
        a = auroc([r[name] for r in runs], labels)
        if a is None:
            continue
        o = max(a, 1 - a)
        if o > best[0]:
            best = (o, name)
        w("  %-15s auc=%.3f oriented=%.3f (%s)" % (name, a, o, "high" if a >= 0.5 else "low"))
    w("  best feature: %s %.3f vs agree %.3f"
      % (best[1], best[0], auroc([r["agree"] for r in runs], labels) or 0.0))
    w()

    sp = [r for r in runs if r["group"] == "SP"]
    spl = [r["maj_ok"] for r in sp]
    w("--- within-SP only (n=%d, agree<1): features vs maj_ok ---" % len(sp))
    w("  pairwise_agree auc=%.3f"
      % (auroc([r["agree"] for r in sp], spl) if len(set(spl)) > 1 else float("nan")))
    for name in feats:
        a = auroc([r[name] for r in sp], spl)
        if a is not None and len(set(spl)) > 1:
            w("  %-15s auc=%.3f oriented=%.3f" % (name, a, max(a, 1 - a)))
    w()

    uni = groups["UR"] + groups["UW"]
    w("--- DECISIVE: within-unanimous (agreement constant = 1.0) ---")
    w("UR=%d UW=%d  (UW is 1 run -- percentile report, not a test)" % (len(groups["UR"]), len(groups["UW"])))
    for uw in groups["UW"]:
        w("UW run: qid=%s src=%s gold=%s maj=%s" % (uw["qid"], uw["src"], uw["gold"], uw["maj"]))
        for name in feats:
            ur_vals = sorted(r[name] for r in groups["UR"] if r[name] is not None)
            v = uw[name]
            if v is None or not ur_vals:
                continue
            below = sum(1 for x in ur_vals if x < v)
            pct = 100.0 * below / len(ur_vals)
            w("  %-15s UW=%.4f  UR median=%.4f  UW at %.0fth pct of UR"
              % (name, v, med(ur_vals), pct))
    w()

    sl, sc = [], []
    for r in meta["records"]:
        sid = r["qid"]
        with open(os.path.join(ROOT, "data", r["probs_json"]), encoding="utf-8") as fh:
            streams = json.load(fh)["streams"]
        recs_by_s = {}
        with open(os.path.join(ROOT, "data", r["probs_jsonl"]), encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    o = json.loads(line)
                    recs_by_s.setdefault(o["s"], []).append(o)
        for i, st in enumerate(streams):
            rs = sorted(recs_by_s.get(str(i + 1), []), key=lambda x: x["step"])
            f, _ = stream_feats(rs, st["text"].encode("utf-8"))
            if f and f["H_ans"] is not None and i < len(r["answers"]):
                sc.append(r["answers"][i] is not None and r["answers"][i] == r["gold"])
                sl.append(f)
    w("--- stream level (n=%d, clustered in runs; not iid) ---" % len(sl))
    for name in ("lp_ans", "H_ans", "p1_ans", "mg_ans", "lp_all", "frac_unc"):
        a = auroc([f[name] for f in sl], sc)
        if a is not None:
            w("  %-15s auc=%.3f oriented=%.3f" % (name, a, max(a, 1 - a)))
    w("  pos/neg: %d/%d" % (sum(sc), len(sc) - sum(sc)))
    w()

    with io.open(OUT_TXT, "w", encoding="utf-8") as fh:
        fh.write(buf.getvalue())
    cols = ["qid", "group", "maj_ok", "agree", "src"] + feats
    with io.open(OUT_CSV, "w", encoding="utf-8", newline="") as fh:
        fh.write(",".join(cols) + "\n")
        for r in runs:
            fh.write(",".join(
                str(r[c]) if not isinstance(r.get(c), float) else "%.5f" % r[c]
                for c in cols) + "\n")
    w("wrote %s" % OUT_TXT)
    w("wrote %s" % OUT_CSV)


if __name__ == "__main__":
    main()
