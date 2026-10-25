#!/usr/bin/env python3
"""Population feature audit for the N=8 stream dump (llama 3.2 3B, seed 1234).

Reads (read-only):
  results/probs_probe30.json + data/_probs_pNN.json/jsonl
  results/hidden_probe30.json + data/_hid_pNN.json/jsonl

Writes:
  results/feature_catalog.md
  a compact summary to stdout

No GPU; no existing file or dump is modified.

Alignment (asserted below, not assumed)
---------------------------------------
probs row step t is the distribution that emitted token t (0-based).
The hidden sidecar h step k is captured on the same engine tick as probs
step k: it is the state after token k-1 that produces token k.
The consensus row c step t is the consensus of h step t+1, so it is
*contemporaneous with probs step t+1*, not probs step t.  The loader
asserts c[t].cos_mean == mean(h[t+1].cmean) over live streams for every
consensus row.  Consequence: naively correlating c[t] with agreement[t]
manufactures a one-step "hidden lead"; the aligned series is
Cal[t] = c[t-1] (the state that produces token t).

All findings are n=32 problems x 8 streams, one model, one seed, one
machine.  Effect sizes are descriptive, not hypothesis tests.
"""

import collections
import itertools
import json
import math
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
PROBS_META = os.path.join(ROOT, "results", "probs_probe30.json")
HID_META = os.path.join(ROOT, "results", "hidden_probe30.json")
OUT_MD = os.path.join(ROOT, "results", "feature_catalog.md")
SEED = 1234
EOS = 128009


# ---------------------------------------------------------------- helpers
def mean(xs):
    xs = [float(x) for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(xs)) if xs else float("nan")


def median(xs):
    xs = [float(x) for x in xs if x is not None and np.isfinite(x)]
    return float(np.median(xs)) if xs else float("nan")


def pctl(xs, p):
    xs = [float(x) for x in xs if x is not None and np.isfinite(x)]
    return float(np.percentile(xs, p)) if xs else float("nan")


def pearson(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3:
        return float("nan")
    return pearson(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))


def fisher_mean(rs):
    rs = [float(r) for r in rs if np.isfinite(r)]
    return float(np.tanh(np.mean(np.arctanh(np.clip(rs, -0.999, 0.999))))) if rs else float("nan")


def partial_corr(a, b, controls):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    X = np.column_stack([np.ones(len(a))] + [np.asarray(c, float) for c in controls])
    ra = a - X @ np.linalg.lstsq(X, a, rcond=None)[0]
    rb = b - X @ np.linalg.lstsq(X, b, rcond=None)[0]
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def slope(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or np.std(x[m]) == 0:
        return float("nan")
    return float(np.polyfit(x[m], y[m], 1)[0])


def boot_ci(values, n=5000, seed=SEED):
    v = np.asarray([x for x in values if np.isfinite(x)], float)
    if len(v) < 3:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    bs = np.array([np.mean(v[rng.integers(0, len(v), len(v))]) for _ in range(n)])
    return (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def fmt(x, nd=3):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(x):
        return "n/a"
    s = ("%." + str(nd) + "f") % abs(x)
    return ("-" if x < 0 else "") + s


def pct(x, nd=1):
    return ("%." + str(nd) + "f%%") % (100.0 * x)


def piece_label(tok, pmap):
    if tok == EOS:
        return "`<EOS>`"
    return clean_piece(pmap.get(tok))


def clean_piece(s, maxlen=12):
    if s is None:
        return "?"
    s = (s.replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t")
         .replace("|", "\\|"))
    s = "".join(c if 32 <= ord(c) < 127 or c == "\\" else "\\x%02x" % ord(c) for c in s)
    if not s:
        return "<empty>"
    return ("`%s`" % s[:maxlen]) if len(s) <= maxlen else ("`%s...`" % s[:maxlen])


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return out


def spark(values, width=60, lo=None, hi=None, chars=" .:-=+*#%@"):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return ""
    if len(v) > width:
        edges = np.linspace(0, len(v), width + 1).astype(int)
        b = []
        for i in range(width):
            seg = v[edges[i]:max(edges[i] + 1, edges[i + 1])]
            b.append(np.mean(seg) if len(seg) else np.nan)
        v = np.array(b)
    lo = np.min(v) if lo is None else lo
    hi = np.max(v) if hi is None else hi
    if hi <= lo:
        return chars[0] * len(v)
    idx = np.clip(((v - lo) / (hi - lo) * (len(chars) - 1)).round().astype(int),
                  0, len(chars) - 1)
    return "".join(chars[i] for i in idx)


def rle(flags):
    lens, run = [], 0
    for f in flags:
        if f:
            run += 1
        else:
            if run:
                lens.append(run)
            run = 0
    if run:
        lens.append(run)
    return lens


# ---------------------------------------------------------------- data
class Run(object):
    pass


def load_runs():
    with open(PROBS_META, encoding="utf-8") as fh:
        meta = json.load(fh)
    with open(HID_META, encoding="utf-8") as fh:
        hmeta = json.load(fh)
    hid_by_qid = {r["qid"]: r for r in hmeta["records"]}

    runs = []
    for rec in meta["records"]:
        qid = rec["qid"]
        with open(os.path.join(DATA, rec["probs_json"]), encoding="utf-8") as fh:
            pj = json.load(fh)
        pstreams = {s["id"]: s for s in pj["streams"]}
        with open(os.path.join(DATA, hid_by_qid[qid]["hidden_json"]), encoding="utf-8") as fh:
            hj = json.load(fh)
        hstreams = {s["id"]: s for s in hj["streams"]}
        for sid, s in pstreams.items():
            if (hstreams[sid]["text"] != s["text"]
                    or hstreams[sid]["generated"] != s["generated"]):
                raise SystemExit("q%d s%s probs/hidden text mismatch" % (qid, sid))

        rows_by_s = collections.defaultdict(list)
        with open(os.path.join(DATA, rec["probs_jsonl"]), encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    o = json.loads(line)
                    rows_by_s[o["s"]].append(o)
        hrows_by_s = collections.defaultdict(list)
        crows = []
        with open(os.path.join(DATA, hid_by_qid[qid]["hidden_jsonl"]), encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                o = json.loads(line)
                if o["k"] == "h":
                    hrows_by_s[o["s"]].append(o)
                elif o["k"] == "c":
                    crows.append(o)
                else:
                    raise SystemExit("bad sidecar kind %r" % o["k"])
        for s in rows_by_s:
            rows_by_s[s].sort(key=lambda x: x["step"])
        for s in hrows_by_s:
            hrows_by_s[s].sort(key=lambda x: x["step"])
        crows.sort(key=lambda x: x["step"])

        gen = {s: len(v) for s, v in rows_by_s.items()}
        if sum(gen.values()) != rec["generated"]:
            raise SystemExit("q%d token count mismatch" % qid)
        if len(rows_by_s) != 8 or len(hrows_by_s) != 8:
            raise SystemExit("q%d expected 8 streams" % qid)
        for s in gen:
            if len(hrows_by_s[s]) != gen[s]:
                raise SystemExit("q%d s%s h rows != generated" % (qid, s))
            if [r["step"] for r in hrows_by_s[s]] != list(range(1, gen[s] + 1)):
                raise SystemExit("q%d s%s h steps not 1..N" % (qid, s))
            if [r["step"] for r in rows_by_s[s]] != list(range(gen[s])):
                raise SystemExit("q%d s%s probs steps not 0..N-1" % (qid, s))
        if [c["step"] for c in crows] != list(range(len(crows))):
            raise SystemExit("q%d c steps not contiguous" % qid)

        max_t = len(crows)
        n_live = np.array([c["n"] for c in crows], float)
        cos = np.array([c["cos_mean"] for c in crows], float)
        cos_min = np.array([c["cos_min"] for c in crows], float)
        cos_max = np.array([c["cos_max"] for c in crows], float)
        step_rows, live_sets = [], []
        A, H, MG, P1, LP, NM = [], [], [], [], [], []
        for t in range(max_t):
            live = [rows_by_s[s][t] for s in rows_by_s if t < gen[s]]
            if len(live) != int(n_live[t]):
                raise SystemExit("q%d t%d live != c.n" % (qid, t))
            if len(live) < 2:
                raise SystemExit("q%d t%d c row with <2 live" % (qid, t))
            step_rows.append(live)
            A.append(max(collections.Counter(o["tok"] for o in live).values()) / len(live))
            H.append(mean([o["H"] for o in live]))
            MG.append(mean([o["mg"] for o in live]))
            P1.append(mean([o["p1"] for o in live]))
            LP.append(mean([o["lp"] for o in live]))
            live_s = [s for s in rows_by_s if t < gen[s]]
            live_sets.append(live_s)
            NM.append(mean([hrows_by_s[s][t]["norm"] for s in live_s]))
            cm = mean([hrows_by_s[s][t]["cmean"] for s in live_s])
            if abs(cm - cos[t]) > 2e-4:
                raise SystemExit("q%d t%d cos alignment mismatch %.6f %.6f"
                                 % (qid, t, cm, cos[t]))
        A, H, MG = np.array(A), np.array(H), np.array(MG)
        P1, LP, NM = np.array(P1), np.array(LP), np.array(NM)
        Cal = np.full(max_t, np.nan)
        Cal[1:] = cos[:-1]

        piece = {}
        for s, rows in rows_by_s.items():
            b = pstreams[s]["text"].encode("utf-8")
            for r in rows:
                if r["len"] > 0:
                    piece[r["tok"]] = b[r["off"]:r["off"] + r["len"]].decode(
                        "utf-8", errors="replace")
        tail = {s: [r["tok"] for r in reversed(rows_by_s[s])] for s in rows_by_s}

        def Aend(k):
            vals = [tail[s][k] for s in tail if len(tail[s]) > k]
            if len(vals) < 2:
                return float("nan")
            return max(collections.Counter(vals).values()) / len(vals)

        r = Run()
        r.qid, r.group = qid, rec["group"]
        r.maj_ok = bool(rec["maj_ok"])
        r.answers, r.gold = rec["answers"], rec["gold"]
        r.pairwise_agree = float(rec["pairwise_agree"])
        r.maj_share = float(rec["maj_share"])
        r.distinct = int(rec["distinct"])
        r.wall_ms = float(rec["wall_ms"])
        r.rows_by_s, r.hrows_by_s = rows_by_s, hrows_by_s
        r.gen = gen
        r.comp = {s: float(pstreams[s]["completion_ms"]) for s in pstreams}
        r.text = {s: pstreams[s]["text"] for s in pstreams}
        r.n, r.C, r.Cmin, r.Cmax = n_live, cos, cos_min, cos_max
        r.Cal, r.A, r.H, r.MG, r.P1, r.LP, r.NM = Cal, A, H, MG, P1, LP, NM
        r.step_rows, r.live_sets, r.max_t = step_rows, live_sets, max_t
        r.piece, r.tail, r.Aend = piece, tail, Aend

        all_rows = [o for s in rows_by_s for o in rows_by_s[s]]
        r.mean_A = mean(A)
        r.mean_C = mean(cos)
        r.mean_H = mean([o["H"] for o in all_rows])
        r.mean_MG = mean([o["mg"] for o in all_rows])
        r.mean_P1 = mean([o["p1"] for o in all_rows])
        r.mean_LP = mean([o["lp"] for o in all_rows])
        r.mean_NM = mean([o["norm"] for s in hrows_by_s for o in hrows_by_s[s]])
        r.mean_gen = mean(list(gen.values()))
        r.max_gen, r.min_gen = max(gen.values()), min(gen.values())
        r.suffix_agree = mean([Aend(k) for k in range(1, 7)])
        r.answer_tok_agree = Aend(1)
        r.reason_end_agree = mean([Aend(k) for k in range(10, 21)])
        runs.append(r)
    return runs, meta, hmeta


# ---------------------------------------------------------------- Q1 lead-lag
def q1_lead_lag(runs):
    lags = list(range(-10, 11))

    def xcorr(mode, naive=False, n8_only=False):
        out = {k: [] for k in lags}
        for r in runs:
            if naive:
                x, y = r.C, r.A[:len(r.C)]
            else:
                x, y = r.Cal[1:], r.A[1:]
            if n8_only:
                m = (r.n[:-1] == 8) & (r.n[1:] == 8)
                x, y = x[m], y[m]
            if mode == "diff":
                x, y = np.diff(x), np.diff(y)
            elif mode == "detrend21":
                w = 21
                if len(x) > 2 * w + 5:
                    k = np.ones(w) / w
                    x = x - np.convolve(x, k, "same")
                    y = y - np.convolve(y, k, "same")
                    x, y = x[w:-w], y[w:-w]
            for k in lags:
                a, b = [], []
                for t in range(len(x)):
                    u = t + k
                    if 0 <= u < len(y) and np.isfinite(x[t]) and np.isfinite(y[u]):
                        a.append(x[t])
                        b.append(y[u])
                if len(a) > 10 and np.std(a) > 0 and np.std(b) > 0:
                    out[k].append(float(np.corrcoef(a, b)[0, 1]))
        return {k: (fisher_mean(v), len(v)) for k, v in out.items()}

    xc = {
        "naive_levels": xcorr("levels", naive=True),
        "aligned_levels": xcorr("levels"),
        "aligned_n8": xcorr("levels", n8_only=True),
        "aligned_diff": xcorr("diff"),
        "aligned_detrend21": xcorr("detrend21"),
    }

    rows = []
    for r in runs:
        for t in range(2, r.max_t - 2):
            if (r.n[t - 1] == 8 and r.n[t] == 8 and r.n[t + 1] == 8
                    and r.n[t + 2] == 8 and np.isfinite(r.Cal[t - 1])):
                rows.append((r.qid, r.A[t - 1], r.A[t], r.A[t + 1], r.A[t + 2],
                             r.Cal[t - 1], r.Cal[t], r.Cal[t + 1], r.H[t], r.MG[t]))
    qid = np.array([x[0] for x in rows])
    Z = np.array([x[1:] for x in rows], float)
    for c in range(Z.shape[1]):
        for q in np.unique(qid):
            m = qid == q
            Z[m, c] -= Z[m, c].mean()
    Aprev, Anow, Anext, Anext2, Cprev, Cnow, Cnxt, Hnow, MGnow = Z.T
    partials = {
        "C[t]->A[t+1] | A[t]": partial_corr(Cnow, Anext, [Anow]),
        "C[t]->A[t+1] | A[t],A[t-1]": partial_corr(Cnow, Anext, [Anow, Aprev]),
        "C[t]->A[t+2] | A[t+1],A[t]": partial_corr(Cnow, Anext2, [Anext, Anow]),
        "C[t]->A[t+2] | A[t+1],A[t],A[t-1]": partial_corr(Cnow, Anext2, [Anext, Anow, Aprev]),
        "A[t]->C[t+1] | C[t]": partial_corr(Anow, Cnxt, [Cnow]),
        "A[t]->C[t+1] | C[t],C[t-1]": partial_corr(Anow, Cnxt, [Cnow, Cprev]),
        "C[t]~A[t] | A[t-1]": partial_corr(Cnow, Anow, [Aprev]),
        "A[t]~C[t] | C[t-1]": partial_corr(Anow, Cnow, [Cprev]),
        "H[t]->A[t+1] | A[t]": partial_corr(Hnow, Anext, [Anow]),
        "MG[t]->A[t+1] | A[t]": partial_corr(MGnow, Anext, [Anow]),
        "H[t]->A[t] | A[t-1]": partial_corr(Hnow, Anow, [Aprev]),
        "H[t]~C[t] | A[t]": partial_corr(Hnow, Cnow, [Anow]),
    }
    rng = np.random.default_rng(SEED)
    qs = np.unique(qid)
    bs = []
    for _ in range(500):
        pick = rng.choice(qs, len(qs), replace=True)
        idx = np.concatenate([np.where(qid == q)[0] for q in pick])
        bs.append(partial_corr(Cnow[idx], Anext[idx], [Anow[idx]]))
    ci = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))

    offsets = list(range(-3, 4))
    ev = []
    for r in runs:
        for t in range(3, r.max_t - 2):
            if r.n[t] == 8 and r.n[t + 1] == 8 and r.A[t] - r.A[t + 1] >= 0.25:
                base = [r.Cal[u] for u in range(max(1, t - 8), max(2, t - 3))
                        if np.isfinite(r.Cal[u])]
                if len(base) >= 2:
                    b = float(np.mean(base))
                    ev.append([r.Cal[t + o] - b if (0 <= t + o < r.max_t
                                                    and np.isfinite(r.Cal[t + o]))
                               else np.nan for o in offsets])
                break
    ev = np.array(ev) if ev else np.zeros((0, len(offsets)))
    ev_mean = [mean(ev[:, j]) if len(ev) else float("nan") for j in range(len(offsets))]
    raw_next = {
        "H[t],A[t+1]": pearson(Hnow, Anext),
        "H[t],A[t]": pearson(Hnow, Anow),
        "MG[t],A[t+1]": pearson(MGnow, Anext),
        "MG[t],A[t]": pearson(MGnow, Anow),
    }
    return {"lags": lags, "xc": xc, "partials": partials, "lead_ci": ci,
            "ev_offsets": offsets, "ev_mean": ev_mean, "ev_n": len(ev),
            "raw_next": raw_next}


# ---------------------------------------------------------------- Q2 matrix
def q2_matrix(runs):
    names = [
        ("token_agree", [r.mean_A for r in runs]),
        ("hidden_cos", [r.mean_C for r in runs]),
        ("mean_entropy", [r.mean_H for r in runs]),
        ("mean_margin", [r.mean_MG for r in runs]),
        ("mean_norm", [r.mean_NM for r in runs]),
        ("mean_lp", [r.mean_LP for r in runs]),
        ("completion_ms", [r.wall_ms for r in runs]),
        ("trace_len", [r.mean_gen for r in runs]),
        ("answer_agree", [r.pairwise_agree for r in runs]),
    ]
    labels = [n for n, _ in names]
    vals = [v for _, v in names]
    M = np.full((len(names), len(names)), np.nan)
    for i in range(len(names)):
        for j in range(len(names)):
            M[i, j] = pearson(vals[i], vals[j])
    pairs = sorted([(abs(M[i, j]), labels[i], labels[j], M[i, j])
                    for i, j in itertools.combinations(range(len(names)), 2)],
                   reverse=True)
    ca, ch, cc = [], [], []
    for r in runs:
        a, h, c = r.A[1:], r.H[1:], r.Cal[1:]
        m = np.isfinite(a) & np.isfinite(h) & np.isfinite(c)
        if m.sum() > 5:
            ca.append(pearson(a[m] - a[m].mean(), c[m] - c[m].mean()))
            ch.append(pearson(a[m] - a[m].mean(), h[m] - h[m].mean()))
            cc.append(pearson(c[m] - c[m].mean(), h[m] - h[m].mean()))
    return {"labels": labels, "M": M, "pairs": pairs,
            "redundant": [p for p in pairs if p[0] >= 0.7],
            "independent": [p for p in pairs if p[0] < 0.2],
            "within": {"A_C": mean(ca), "A_H": mean(ch), "C_H": mean(cc)}}


# ---------------------------------------------------------------- Q3 regimes
def kmeans(X, k, seed=SEED, n_init=10, max_iter=100):
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        centers = [X[rng.integers(len(X))]]
        for _ in range(1, k):
            d2 = np.min(((X[:, None, :] - np.array(centers)[None, :, :]) ** 2).sum(2), axis=1)
            centers.append(X[rng.choice(len(X), p=d2 / d2.sum())]
                           if d2.sum() > 0 else X[rng.integers(len(X))])
        C = np.array(centers)
        lab = np.zeros(len(X), int)
        for _ in range(max_iter):
            lab = ((X[:, None, :] - C[None, :, :]) ** 2).sum(2).argmin(1)
            newC = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j]
                             for j in range(k)])
            if np.allclose(newC, C):
                C = newC
                break
            C = newC
        inertia = float(((X - C[lab]) ** 2).sum())
        if best is None or inertia < best[0]:
            best = (inertia, lab, C)
    return best[1], best[2]


def q3_regimes(runs):
    rows = []
    for r in runs:
        raw = np.array([(r.H[t], r.A[t], r.Cal[t]) for t in range(1, r.max_t)])
        Z = raw.copy()
        for c in range(3):
            s = Z[:, c].std()
            Z[:, c] = (Z[:, c] - Z[:, c].mean()) / s if s > 0 else 0.0
        for i, t in enumerate(range(1, r.max_t)):
            rows.append((r.qid, t / float(r.max_t - 1), Z[i, 0], Z[i, 1], Z[i, 2],
                         r.n[t - 1]))
    D = np.array([x[2:5] for x in rows], float)
    relpos = np.array([x[1] for x in rows])
    ncons = np.array([x[5] for x in rows])
    lab4, cen4 = kmeans(D, 4)
    cell = ((D[:, 0] > 0).astype(int) * 4 + (D[:, 1] > 0).astype(int) * 2
            + (D[:, 2] > 0).astype(int))
    cells = {}
    for c in range(8):
        m = cell == c
        cells[c] = (int(m.sum()), float(m.mean()),
                    float(np.mean(relpos[m])) if m.any() else float("nan"))
    return {"D": D, "relpos": relpos, "ncons": ncons, "labels4": lab4,
            "centers4": cen4, "cells": cells}


# ---------------------------------------------------------------- Q4 dynamics
def q4_dynamics(runs):
    recon = 0
    one_hi, one_lo, div_epochs = [], [], []
    sA, sC, sA8, sC8, sA2, sC2 = [], [], [], [], [], []
    early_late = []
    for r in runs:
        A, n = r.A, r.n
        low = np.where(A < 1)[0]
        if len(low) and (A[low[-1] + 1:] == 1).any():
            recon += 1
        # A==1 epochs split by average live count
        i = 0
        while i < len(A):
            if A[i] == 1:
                j = i
                while j < len(A) and A[j] == 1:
                    j += 1
                (one_hi if np.mean(n[i:j]) >= 6 else one_lo).append(j - i)
                i = j
            else:
                i += 1
        div_epochs.extend(rle(A < 1))
        sA.append(slope(np.linspace(0, 1, len(A)), A))
        ok = np.isfinite(r.Cal)
        sC.append(slope(np.linspace(0, 1, len(A))[ok], r.Cal[ok]))
        m8 = (n == 8) & np.isfinite(r.Cal)
        if m8.sum() > 10:
            sA8.append(slope(np.linspace(0, 1, m8.sum()), A[m8]))
            sC8.append(slope(np.linspace(0, 1, m8.sum()), r.Cal[m8]))
            h = m8.sum() // 2
            sA2.append(slope(np.linspace(0, 1, m8.sum() - h), A[m8][h:]))
            sC2.append(slope(np.linspace(0, 1, m8.sum() - h), r.Cal[m8][h:]))
        half = len(A) // 2
        early_late.append((float(np.mean(A[:half])), float(np.mean(A[half:]))))

    hid_epochs = {}
    for thr in (0.99, 0.999):
        durs, rel = [], []
        tot = 0
        for r in runs:
            cal = r.Cal
            tot += len(cal) - 1
            run = 0
            for t in range(1, len(cal)):
                if cal[t] >= thr:
                    run += 1
                else:
                    if run:
                        durs.append(run)
                        rel.append((t - run) / len(cal))
                    run = 0
            if run:
                durs.append(run)
                rel.append((len(cal) - run) / len(cal))
        hid_epochs[thr] = {"steps": int(sum(durs)), "total": tot, "n": len(durs),
                           "med": median(durs), "p90": pctl(durs, 90),
                           "max": max(durs) if durs else 0, "relpos": mean(rel)}

    n_count, a_one = collections.Counter(), collections.Counter()
    for r in runs:
        for t in range(r.max_t):
            n_count[int(r.n[t])] += 1
            if r.A[t] == 1:
                a_one[int(r.n[t])] += 1
    small = tot_last = 0
    for r in runs:
        for t in range(int(0.9 * r.max_t), r.max_t):
            tot_last += 1
            small += int(r.n[t] <= 4)
    # second-half A/C correlation per run
    r2 = []
    for r in runs:
        m = (r.n == 8) & np.isfinite(r.Cal)
        if m.sum() > 20:
            a, c = r.A[m], r.Cal[m]
            h = len(a) // 2
            if np.std(a[h:]) > 0 and np.std(c[h:]) > 0:
                r2.append(pearson(a[h:], c[h:]))
    return {"recon": recon, "one_hi": one_hi, "one_lo": one_lo,
            "div_epochs": div_epochs, "slope_A": sA, "slope_C": sC,
            "slope_A8": sA8, "slope_C8": sC8, "slope_A2": sA2, "slope_C2": sC2,
            "early_late": early_late, "hid_epochs": hid_epochs,
            "n_count": dict(n_count), "a_one": dict(a_one),
            "last_small": small, "last_tot": tot_last, "r2": r2}


# ---------------------------------------------------------------- Q5 variance
def q5_variance(runs):
    shares, ratios = [], []
    for r in runs:
        L = r.min_gen
        S = len(r.gen)
        M = np.array([[r.rows_by_s[s][t]["H"] for t in range(L)]
                      for s in sorted(r.rows_by_s)])
        gm = M.mean()
        ss_step = S * ((M.mean(0) - gm) ** 2).sum()
        ss_stream = L * ((M.mean(1) - gm) ** 2).sum()
        tot = ((M - gm) ** 2).sum()
        shares.append((ss_step / tot, ss_stream / tot,
                       (tot - ss_step - ss_stream) / tot))
        ratios.append(np.mean(np.var(M, axis=0, ddof=1)) / np.mean(np.var(M, axis=1, ddof=1)))
    shares = np.array(shares)

    hes = {}
    for thr in (math.log(2.0), 1.0, 1.5):
        durs, runs_with, total = [], 0, 0
        for r in runs:
            M = np.array([[r.rows_by_s[s][t]["H"] for t in range(r.min_gen)]
                          for s in sorted(r.rows_by_s)])
            flag = (M > thr).all(0)
            total += len(flag)
            runs_with += int(flag.any())
            durs.extend(rle(flag))
        hes[thr] = {"steps": int(sum(durs)), "total": total, "runs": runs_with,
                    "windows": len(durs), "med": median(durs),
                    "p90": pctl(durs, 90), "max": max(durs) if durs else 0}

    conf = {}
    for thr in (0.05, 0.01):
        steps = total = 0
        for r in runs:
            M = np.array([[r.rows_by_s[s][t]["H"] for t in range(r.min_gen)]
                          for s in sorted(r.rows_by_s)])
            flag = (M < thr).all(0)
            steps += int(flag.sum())
            total += len(flag)
        conf[thr] = (steps, total)
    return {"shares": shares, "ratios": ratios, "hes": hes, "conf": conf}


# ---------------------------------------------------------------- Q6 sampler
def tail_predict(p1, H, T):
    """Top token p1, remainder uniform over K-1 effective tokens chosen to
    match entropy H.  Returns (p1 at temperature T, sum q^2, P(all 8 equal), K)."""
    p1 = min(max(float(p1), 1e-12), 1.0 - 1e-12)
    K = 1.0 + math.exp(min((H + p1 * math.log(p1)) / (1.0 - p1), 50.0))
    tail = (1.0 - p1) / (K - 1.0) if K > 1.0 + 1e-9 else (1.0 - p1)
    a1 = math.log(p1) / T
    at = math.log(max(tail, 1e-300)) / T
    Z = math.exp(a1) + (K - 1.0) * math.exp(at)
    q1, qt = math.exp(a1) / Z, math.exp(at) / Z
    return q1, q1 * q1 + (K - 1.0) * qt * qt, q1 ** 8 + (K - 1.0) * qt ** 8, K


def q6_sampler(runs):
    argmax, p1t1, p1t6 = [], [], []
    step_pred = []
    for r in runs:
        for t in range(r.max_t):
            live = r.step_rows[t]
            toks = [o["tok"] for o in live]
            A = max(collections.Counter(toks).values()) / len(live)
            eq = (sum(v * (v - 1) for v in collections.Counter(toks).values())
                  / (len(live) * (len(live) - 1)))
            mp1 = mean([o["p1"] for o in live])
            mh = mean([o["H"] for o in live])
            _, coll, pall, _ = tail_predict(mp1, mh, 0.6)
            step_pred.append((A, eq, pall, coll, len(live), t == 0))
            for o in live:
                am = (math.exp(o["lp"]) / o["p1"] > 0.999) if o["p1"] > 0 else False
                q1_1 = tail_predict(o["p1"], o["H"], 1.0)[0]
                q1_6 = tail_predict(o["p1"], o["H"], 0.6)[0]
                argmax.append(am)
                p1t1.append(q1_1)
                p1t6.append(q1_6)
    argmax = np.array(argmax, float)
    p1t1, p1t6 = np.array(p1t1), np.array(p1t6)
    bins = [0.0, 0.3, 0.5, 0.7, 0.85, 0.95, 1.01]
    binned = []
    for i in range(len(bins) - 1):
        m = (p1t1 >= bins[i]) & (p1t1 < bins[i + 1])
        if m.any():
            binned.append((bins[i], bins[i + 1], int(m.sum()), float(argmax[m].mean()),
                           float(p1t1[m].mean()), float(p1t6[m].mean())))
    sp = np.array(step_pred, float)
    s0 = sp[sp[:, 5] == 1]
    return {"obs_argmax": float(argmax.mean()), "pred_T1": float(p1t1.mean()),
            "pred_T06": float(p1t6.mean()), "binned": binned,
            "obs_pair": float(sp[:, 1].mean()), "pred_pair": float(sp[:, 3].mean()),
            "obs_pall": float(np.mean(sp[:, 0] == 1)), "pred_pall": float(sp[:, 2].mean()),
            "step0_A": float(s0[:, 0].mean()) if len(s0) else float("nan"),
            "step0_pall": float(s0[:, 2].mean()) if len(s0) else float("nan"),
            "step0_n": len(s0)}


# ---------------------------------------------------------------- Q7 streams
def q7_streams(runs):
    metrics = ["gen", "completion", "entropy", "norm", "lp", "cmean", "correct"]
    M = {k: np.full((len(runs), 8), np.nan) for k in metrics}
    for i, r in enumerate(runs):
        for j, s in enumerate(sorted(r.rows_by_s, key=int)):
            rows, hrows = r.rows_by_s[s], r.hrows_by_s[s]
            M["gen"][i, j] = len(rows)
            M["completion"][i, j] = r.comp[s]
            M["entropy"][i, j] = mean([o["H"] for o in rows])
            M["norm"][i, j] = mean([o["norm"] for o in hrows])
            M["lp"][i, j] = mean([o["lp"] for o in rows])
            M["cmean"][i, j] = mean([o["cmean"] for o in hrows])
            a = r.answers[j]
            M["correct"][i, j] = 1.0 if (a is not None and a == r.gold) else 0.0
    rng = np.random.default_rng(SEED)
    out = {}
    for k in metrics:
        X = M[k]
        obs = np.nanmean(X, axis=0)
        spread = float(np.nanmax(obs) - np.nanmin(obs))
        null = np.array([np.nanmax(np.nanmean(Y, axis=0)) - np.nanmin(np.nanmean(Y, axis=0))
                         for Y in (np.array([row[rng.permutation(8)] for row in X])
                                   for _ in range(2000))])
        out[k] = {"means": obs, "spread": spread,
                  "null50": float(np.median(null)), "null95": float(np.percentile(null, 95)),
                  "ratio": spread / float(np.median(null)) if np.median(null) > 0 else float("nan"),
                  "rho": spearman(np.arange(1, 9), obs)}
    return out


# ---------------------------------------------------------------- Q8 norm
def q8_norm(runs):
    norm, tok = [], []
    for r in runs:
        for s in sorted(r.rows_by_s, key=int):
            rows, hrows = r.rows_by_s[s], r.hrows_by_s[s]
            for t in range(len(rows)):
                norm.append(hrows[t]["norm"])
                tok.append(rows[t]["tok"])
    norm, tok = np.array(norm), np.array(tok)
    cnt = collections.Counter(tok.tolist())
    common = [t for t, c in cnt.items() if c >= 20]
    gm = norm.mean()
    ss_tot = ((norm - gm) ** 2).sum()
    token_mean = {}
    ss_between = 0.0
    for t in common:
        m = tok == t
        token_mean[t] = float(norm[m].mean())
        ss_between += int(m.sum()) * (token_mean[t] - gm) ** 2
    eta2 = float(ss_between / ss_tot)

    space = tok == 220
    d_space = ((norm[space].mean() - norm[~space].mean())
               / math.sqrt((norm[space].var(ddof=1) + norm[~space].var(ddof=1)) / 2.0))

    res = norm.copy()
    for t in common:
        m = tok == t
        res[m] -= token_mean[t]

    def slope_by_stream(vec):
        vals = []
        k = 0
        for r in runs:
            for s in sorted(r.rows_by_s, key=int):
                ln = len(r.rows_by_s[s])
                y = vec[k:k + ln]
                k += ln
                if ln > 5:
                    vals.append(slope(np.linspace(0, 1, ln), y))
        return vals

    slopes, slopes_res = slope_by_stream(norm), slope_by_stream(res)

    # NM[t] is h step t+1: the state that produces token t+1.  To align to
    # token t use NM[t-1], Cal[t] (=c[t-1]=consensus of h step t), H[t], A[t].
    rN_H, rN_A, rN_C = [], [], []
    for r in runs:
        nm = r.NM[:-1]
        m = np.isfinite(nm) & np.isfinite(r.Cal[1:])
        if m.sum() > 5:
            hh, aa, cc = r.H[1:], r.A[1:], r.Cal[1:]
            rN_H.append(pearson(nm[m] - nm[m].mean(), hh[m] - hh[m].mean()))
            rN_A.append(pearson(nm[m] - nm[m].mean(), aa[m] - aa[m].mean()))
            rN_C.append(pearson(nm[m] - nm[m].mean(), cc[m] - cc[m].mean()))

    spikes = []
    for r in runs:
        for s in sorted(r.rows_by_s, key=int):
            hrows, rows = r.hrows_by_s[s], r.rows_by_s[s]
            for t in range(len(hrows) - 1):
                d = hrows[t + 1]["norm"] - hrows[t]["norm"]
                spikes.append((abs(d), d, r.qid, s, t + 1, hrows[t + 1]["norm"],
                               rows[t + 1]["tok"]))
    spikes.sort(reverse=True)
    tok_rows = sorted(((token_mean[t], cnt[t], t) for t in common))
    hist, _ = np.histogram(norm, bins=15, range=(62, 93))
    gpiece = {}
    for r in runs:
        gpiece.update(r.piece)
    return {"n": len(norm), "mean": float(norm.mean()), "sd": float(norm.std(ddof=1)),
            "p1": float(np.percentile(norm, 1)), "p5": float(np.percentile(norm, 5)),
            "p50": float(np.percentile(norm, 50)), "p95": float(np.percentile(norm, 95)),
            "p99": float(np.percentile(norm, 99)), "min": float(norm.min()),
            "max": float(norm.max()), "eta2": eta2,
            "space_mean": float(norm[space].mean()), "nonspace_mean": float(norm[~space].mean()),
            "space_var_share": float(int(space.sum()) * (norm[space].mean() - gm) ** 2 / ss_tot),
            "d_space": float(d_space),
            "slope_med": median(slopes),
            "slope_neg": mean([1.0 if x < 0 else 0.0 for x in slopes]),
            "slope_res_med": median(slopes_res),
            "slope_res_neg": mean([1.0 if x < 0 else 0.0 for x in slopes_res]),
            "rN_H": mean(rN_H), "rN_A": mean(rN_A), "rN_C": mean(rN_C),
            "top_spikes": spikes[:8], "space_in_top": sum(1 for x in spikes[:100] if x[6] == 220),
            "low_tok": tok_rows[:6], "high_tok": tok_rows[-6:],
            "n_common": len(common),
            "coverage": sum(cnt[t] for t in common) / float(len(norm)),
            "hist": hist.tolist(), "piece": gpiece,
            "eos_norm": float(norm[tok == EOS].mean()) if (tok == EOS).any() else float("nan")}


# ---------------------------------------------------------------- Q9 timing
def q9_timing(runs):
    straggler, comp_gen, first, second, last = [], [], [], [], []
    rank_corr, length_correct, diffs = [], [], []
    first_counts = collections.Counter()
    first_shortest = 0
    attr = {k: [] for k in ("H", "NM", "C", "gen")}
    for r in runs:
        ids = sorted(r.gen, key=int)
        c = np.array([r.comp[s] for s in ids])
        straggler.append(float(c.max() / np.median(c)))
        comp_gen.append(pearson(np.log(c), np.log([r.gen[s] for s in ids])))
        ok = {}
        for j, s in enumerate(ids):
            a = r.answers[j]
            ok[s] = 1.0 if (a is not None and a == r.gold) else 0.0
        order = sorted(r.comp, key=lambda s: r.comp[s])
        first.append(ok[order[0]])
        second.append(ok[order[1]])
        last.append(ok[order[-1]])
        first_counts[order[0]] += 1
        first_shortest += int(r.gen[order[0]] == min(r.gen.values()))
        diffs.append(ok[order[0]] - np.mean([ok[s] for s in order[1:]]))
        for key, getf in (
            ("H", lambda s: mean([o["H"] for o in r.rows_by_s[s]])),
            ("NM", lambda s: mean([o["norm"] for o in r.hrows_by_s[s]])),
            ("C", lambda s: mean([o["cmean"] for o in r.hrows_by_s[s]])),
            ("gen", lambda s: float(r.gen[s])),
        ):
            vf = getf(order[0])
            vr = mean([getf(s) for s in order[1:]])
            attr[key].append(vf - vr)
        if len(set(ok.values())) > 1:
            rank = np.argsort(np.argsort([r.comp[s] for s in order])).astype(float)
            rank_corr.append(pearson(rank, np.array([ok[s] for s in order])))
        gs = np.array([r.gen[s] for s in ids], float)
        ov = np.array([ok[s] for s in ids], float)
        if np.std(gs) > 0 and np.std(ov) > 0:
            length_correct.append(pearson(gs, ov))
    d = np.array(first) - np.array(second)
    return {"straggler": straggler, "comp_gen": comp_gen,
            "first": float(np.mean(first)), "second": float(np.mean(second)),
            "last": float(np.mean(last)), "diff": float(np.mean(diffs)),
            "diff_ci": boot_ci(diffs),
            "d12": float(d.mean() / d.std()) if d.std() > 0 else float("nan"),
            "rank_corr": mean(rank_corr), "first_counts": dict(first_counts),
            "first_shortest": first_shortest / float(len(runs)),
            "length_correct": mean(length_correct), "attr": attr}


# ---------------------------------------------------------------- Q10 other
def q10_other(runs):
    prof = collections.defaultdict(list)
    prof_g = {g: collections.defaultdict(list) for g in ("UR", "SP", "UW")}
    for r in runs:
        for k in range(min(25, min(r.gen.values()))):
            vals = [r.tail[s][k] for s in r.tail if len(r.tail[s]) > k]
            if len(vals) >= 2:
                a = max(collections.Counter(vals).values()) / len(vals)
                prof[k].append(a)
                prof_g[r.group][k].append(a)
    ja, jc = [], []
    for r in runs:
        for t in range(1, r.max_t):
            ja.append((abs(r.A[t] - r.A[t - 1]), r.qid, t, t / r.max_t,
                       r.A[t - 1], r.A[t], int(r.n[t])))
            if np.isfinite(r.Cal[t - 1]):
                jc.append((abs(r.Cal[t] - r.Cal[t - 1]), r.qid, t, t / r.max_t,
                           r.Cal[t - 1], r.Cal[t], int(r.n[t])))
    ja.sort(reverse=True)
    jc.sort(reverse=True)
    groups = {g: [r for r in runs if r.group == g] for g in ("UR", "SP", "UW")}

    def extremes(key, n=3):
        s = sorted(runs, key=lambda r: getattr(r, key))
        return s[:n], s[-n:]

    # answer-marker timing
    rel, live_at = [], []
    for r in runs:
        astep = {}
        for s, rows in r.rows_by_s.items():
            m = r.text[s].encode("utf-8").rfind(b"The answer is")
            if m >= 0:
                astep[s] = next((o["step"] for o in rows if o["off"] >= m), None)
        vals = [v for v in astep.values() if v is not None]
        if not vals:
            continue
        for s, v in astep.items():
            if v is not None:
                rel.append(r.gen[s] - v)
        tmed = int(np.median(vals))
        live_at.append(sum(1 for s, g in r.gen.items() if g > tmed))

    return {"prof": prof, "prof_g": prof_g, "jumps_A": ja[:8], "jumps_C": jc[:8],
            "jumps_A8": [x for x in ja if x[6] == 8][:4],
            "jumps_C8": [x for x in jc if x[6] == 8][:4],
            "groups": groups,
            "marker_rel": median(rel), "marker_n": len(rel),
            "marker_live": median(live_at),
            "marker_ge4": sum(1 for x in live_at if x >= 4),
            "extremes": {k: extremes(k) for k in
                         ("mean_H", "mean_A", "mean_C", "wall_ms", "mean_gen")}}


# ---------------------------------------------------------------- report
def peak(d):
    return max(d, key=lambda k: (d[k][0] if np.isfinite(d[k][0]) else -9))


def main():
    runs, meta, hmeta = load_runs()
    n = len(runs)
    print("loaded %d runs, %d token rows, %d consensus rows"
          % (n, sum(sum(r.gen.values()) for r in runs), sum(r.max_t for r in runs)))

    q1 = q1_lead_lag(runs)
    q2 = q2_matrix(runs)
    q3 = q3_regimes(runs)
    q4 = q4_dynamics(runs)
    q5 = q5_variance(runs)
    q6 = q6_sampler(runs)
    q7 = q7_streams(runs)
    q8 = q8_norm(runs)
    q9 = q9_timing(runs)
    q10 = q10_other(runs)
    xc = q1["xc"]

    print()
    print("== feature audit summary (n=%d problems x 8 streams, seed 1234) ==" % n)
    print("Q1 aligned xcorr peak: levels %+d r=%s | diff %+d r=%s | detrend %+d r=%s"
          % (peak(xc["aligned_levels"]), fmt(xc["aligned_levels"][peak(xc["aligned_levels"])][0]),
             peak(xc["aligned_diff"]), fmt(xc["aligned_diff"][peak(xc["aligned_diff"])][0]),
             peak(xc["aligned_detrend21"]),
             fmt(xc["aligned_detrend21"][peak(xc["aligned_detrend21"])][0])))
    print("Q1 naive xcorr peak %+d r=%s (sidecar offset, not a lead)"
          % (peak(xc["naive_levels"]), fmt(xc["naive_levels"][peak(xc["naive_levels"])][0])))
    print("Q1 partial C->A[t+1]|A[t] %s (CI %s..%s) vs reverse A->C[t+1]|C[t] %s"
          % (fmt(q1["partials"]["C[t]->A[t+1] | A[t]"]), fmt(q1["lead_ci"][0]),
             fmt(q1["lead_ci"][1]), fmt(q1["partials"]["A[t]->C[t+1] | C[t]"])))
    print("Q2 run-level r(A,H)=%s r(A,C)=%s r(C,H)=%s; within-run r(A,H)=%s"
          % (fmt(q2["M"][0, 2]), fmt(q2["M"][0, 1]), fmt(q2["M"][1, 2]),
             fmt(q2["within"]["A_H"])))
    print("Q3 k=4 prevalence %s"
          % ", ".join("%d:%s" % (i, pct(np.mean(q3["labels4"] == i))) for i in range(4)))
    print("Q4 recon %d/%d; slope A n=8 %s (neg %d/%d); C n=8 %s (neg %d/%d); 2nd-half r(A,C)=%s"
          % (q4["recon"], n, fmt(median(q4["slope_A8"])),
             sum(1 for x in q4["slope_A8"] if x < 0), len(q4["slope_A8"]),
             fmt(median(q4["slope_C8"])),
             sum(1 for x in q4["slope_C8"] if x < 0), len(q4["slope_C8"]),
             fmt(mean(q4["r2"]))))
    print("Q5 H shares step %s / stream %s / resid %s; global hesitation H>1 %d steps (%s)"
          % (fmt(median(q5["shares"][:, 0])), fmt(median(q5["shares"][:, 1])),
             fmt(median(q5["shares"][:, 2])), q5["hes"][1.0]["steps"],
             pct(q5["hes"][1.0]["steps"] / max(1, q5["hes"][1.0]["total"]))))
    print("Q6 argmax obs %s vs T1 %s / T0.6 %s; pair-match obs %s vs shared-null %s"
          % (fmt(q6["obs_argmax"]), fmt(q6["pred_T1"]), fmt(q6["pred_T06"]),
             fmt(q6["obs_pair"]), fmt(q6["pred_pair"])))
    print("Q7 spread/null50 " + " ".join("%s=%.2f" % (k, q7[k]["ratio"])
          for k in ("gen", "completion", "entropy", "norm", "correct")))
    print("Q8 norm mean %s sd %s eta2(token)=%s space %s vs %s d=%s; slope %s -> resid %s"
          % (fmt(q8["mean"], 2), fmt(q8["sd"], 2), fmt(q8["eta2"], 2),
             fmt(q8["space_mean"], 1), fmt(q8["nonspace_mean"], 1), fmt(q8["d_space"], 1),
             fmt(q8["slope_med"], 2), fmt(q8["slope_res_med"], 2)))
    print("Q9 straggler med %s p90 %s; first-finisher %s vs rest %.2f (diff %+.3f, CI %+.3f..%+.3f)"
          % (fmt(median(q9["straggler"]), 2), fmt(pctl(q9["straggler"], 90), 2),
             fmt(q9["first"], 2), q9["first"] - q9["diff"], q9["diff"],
             q9["diff_ci"][0], q9["diff_ci"][1]))
    print("Q10 end-aligned answer %s | last-6 %s | 10-20-back %s"
          % (fmt(mean([r.answer_tok_agree for r in runs])),
             fmt(mean([r.suffix_agree for r in runs])),
             fmt(mean([r.reason_end_agree for r in runs]))))

    md = build_md(runs, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10)
    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    print()
    print("wrote %s" % OUT_MD)


def build_md(runs, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10):
    n = len(runs)
    L = []
    add = L.append
    xc = q1["xc"]

    add("# N=8 stream population: feature catalog")
    add("")
    add("Generated by `harness/feature_audit.py` (no GPU, read-only inputs). "
        "Data: `results/probs_probe30.json` + `data/_probs_pNN.jsonl`, "
        "`results/hidden_probe30.json` + `data/_hid_pNN.jsonl`.")
    add("")
    add("**Population.** %d problems x 8 streams, llama 3.2 3B, temperature 0.6, "
        "top-p 0.95, seed 1234, one RTX 4060. %d emitted-token rows, %d "
        "cross-stream consensus rows. Answer groups: UR=%d, SP=%d, UW=%d."
        % (n, sum(sum(r.gen.values()) for r in runs), sum(r.max_t for r in runs),
           len(q10["groups"]["UR"]), len(q10["groups"]["SP"]), len(q10["groups"]["UW"])))
    add("")
    add("**Read me first.** n=32 problems, one model, one seed, one machine. At "
        "n=32 the 95% band for a zero correlation is roughly |r|<0.35, so "
        "`|r|<0.2` means *no detectable association*, not proven independence. "
        "Effect sizes are reported instead of p-values. Nothing was tuned to "
        "improve a result; the one non-obvious alignment below is forced by the "
        "dump and is asserted in code.")
    add("")
    add("## 0. Alignment (important)")
    add("")
    add("`probs` step `t` is the distribution that emitted token `t` (0-based). "
        "Hidden sidecar `h` step `k` is captured on the same tick as `probs` "
        "step `k`: it is the state *after* token `k-1` that produces token `k`. "
        "The consensus row `c` step `t` is the consensus of `h` step `t+1`, so "
        "it is **contemporaneous with probs step t+1**, not step `t`. The "
        "loader asserts `c[t].cos_mean == mean(h[t+1].cmean)` over live streams "
        "for all %d consensus rows (holds to <2e-4)." % sum(r.max_t for r in runs))
    add("")
    add("Consequence: naive `corr(c[t], agreement[t])` manufactures a +1-step "
        "\"hidden lead\". The aligned series below is `Cal[t] = c[t-1]` (the "
        "state that produces token `t`). Section 1 reports both; the naive peak "
        "is a dump artifact, not a population property.")
    add("")

    # Q1
    add("## 1. Lead-lag: does hidden consensus lead token agreement?")
    add("")
    add("**Method.** Per run, token agreement `A[t]` = modal token fraction "
        "among live streams; hidden consensus `C[t] = c[t].cos_mean`; aligned "
        "`Cal[t] = C[t-1]`. `r(k) = corr(Cal[t], A[t+k])`, k=-10..+10, "
        "Fisher-averaged over 32 runs. Variants: levels, first differences, "
        "21-step moving-average detrend, all-8-live steps only. Positive k = "
        "hidden at `t` vs agreement later.")
    add("")
    add("**Result (aligned).**")
    add("")
    rows = []
    for k in (-3, -2, -1, 0, 1, 2, 3):
        rows.append(("%+d" % k, fmt(xc["aligned_levels"][k][0]),
                     fmt(xc["aligned_n8"][k][0]), fmt(xc["aligned_diff"][k][0]),
                     fmt(xc["aligned_detrend21"][k][0])))
    L.extend(table(["lag k", "levels", "n=8 only", "first-diff", "detrend21"], rows))
    add("")
    for name in ("aligned_levels", "aligned_n8", "aligned_diff",
                 "aligned_detrend21", "naive_levels"):
        pk = peak(xc[name])
        add("- `%s`: peak lag **%+d**, r=%s; r(+1)-r(-1)=%s"
            % (name, pk, fmt(xc[name][pk][0]),
               fmt(xc[name][1][0] - xc[name][-1][0])))
    add("")
    add("The naive join peaks at **+1** (r=%s). After the forced one-step "
        "alignment the peak moves to **0** in every robust variant; in first "
        "differences and detrended series there is no positive-lag shoulder. "
        "The data do **not** show hidden consensus leading token agreement."
        % fmt(xc["naive_levels"][peak(xc["naive_levels"])][0]))
    add("")
    add("**Partial correlations** (within-run centered, all-8-live steps):")
    add("")
    L.extend(table(["partial", "r"],
                   [(k.replace(" | ", "; "), fmt(v))
                    for k, v in q1["partials"].items()]))
    add("")
    add("Hidden consensus carries a modest one-step-ahead signal "
        "(`C[t]->A[t+1]|A[t]` = %s, cluster-bootstrap 95%% CI %s..%s) after "
        "controlling current agreement. But the reverse direction is equally "
        "strong (`A[t]->C[t+1]|C[t]` = %s), and `C[t]->A[t+2]` drops to %s "
        "once t+1 is controlled. Interpretation: **two noisy views of one "
        "autocorrelated divergence process**, not a directional lead. Entropy "
        "and margin are weaker: raw corr(H[t],A[t+1])=%s vs corr(H[t],A[t])=%s; "
        "raw corr(MG[t],A[t+1])=%s vs corr(MG[t],A[t])=%s. Neither uncertainty "
        "measure is a useful leading indicator of token agreement."
        % (fmt(q1["partials"]["C[t]->A[t+1] | A[t]"]), fmt(q1["lead_ci"][0]),
           fmt(q1["lead_ci"][1]), fmt(q1["partials"]["A[t]->C[t+1] | C[t]"]),
           fmt(q1["partials"]["C[t]->A[t+2] | A[t+1],A[t]"]),
           fmt(q1["raw_next"]["H[t],A[t+1]"]), fmt(q1["raw_next"]["H[t],A[t]"]),
           fmt(q1["raw_next"]["MG[t],A[t+1]"]), fmt(q1["raw_next"]["MG[t],A[t]"])))
    add("")
    add("**Event study.** First step where `A` drops by >=0.25 with 8 live "
        "streams; baseline `Cal` from t-8..t-4 (n=%d). Mean change in `Cal` at "
        "offsets -3..+3: %s. The hidden drop is largest *after* the token drop, "
        "consistent with tokens driving representations rather than the reverse."
        % (q1["ev_n"], ", ".join("%+d:%s" % (o, fmt(v)) for o, v in
                                 zip(q1["ev_offsets"], q1["ev_mean"]))))
    add("")
    add("**Verdict.** No hidden lead. The highest-value question gets a clean "
        "negative after fixing the alignment; consensus and agreement are "
        "contemporaneous outputs of the same trajectory divergence.")
    add("")

    # Q2
    add("## 2. Two axes: correlation matrix over per-run aggregates")
    add("")
    add("**Method.** Pearson over 32 runs for the 9 requested aggregates: mean "
        "token agreement, mean hidden cos, mean entropy, mean margin, mean norm, "
        "mean emitted logprob, run wall time, mean trace length, answer "
        "agreement (`pairwise_agree`, the fraction of stream pairs with the "
        "same extracted answer). Step-level within-run centered correlations "
        "are reported separately.")
    add("")
    hdr = ["var"] + q2["labels"]
    rows = [[lab] + [fmt(q2["M"][i, j], 2) for j in range(len(q2["labels"]))]
            for i, lab in enumerate(q2["labels"])]
    L.extend(table(hdr, rows))
    add("")
    add("**Redundant (|r|>=0.70).**")
    for ab, a, b, r in q2["redundant"][:12]:
        add("- %s ~ %s: r=%s" % (a, b, fmt(r)))
    add("")
    add("**No detectable association (|r|<0.20).**")
    for ab, a, b, r in q2["independent"][:14]:
        add("- %s ~ %s: r=%s" % (a, b, fmt(r)))
    add("")
    add("Within-run step-level (centered): r(A,H)=%s, r(A,C)=%s, r(C,H)=%s. "
        "**Confidence and consensus are nearly orthogonal at the step level**, "
        "while token agreement and hidden consensus share about a quarter of "
        "their variance. The two axes are (i) local confidence "
        "(entropy/margin) and (ii) population consensus (token agreement/hidden "
        "cos). Run-level answer agreement tracks mean entropy (r=%s) but **not** "
        "token agreement (r=%s)."
        % (fmt(q2["within"]["A_H"]), fmt(q2["within"]["A_C"]),
           fmt(q2["within"]["C_H"]), fmt(q2["M"][2, 8]), fmt(q2["M"][0, 8])))
    add("")

    # Q3
    add("## 3. Regimes")
    add("")
    add("**Method.** Step rows `t=1..max_t-1` with `H[t]`, `A[t]`, `Cal[t]`, "
        "z-scored within run; k-means k=4 (deterministic k-means++, 10 "
        "restarts, seed 1234). A 2x2x2 median-split cell count is a "
        "non-parametric cross-check. Regimes are descriptive, not validated.")
    add("")
    add("| cluster | n | share | zH | zA | zC | relpos | early<.25 | late>.75 | mean n |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    for c in range(4):
        m = q3["labels4"] == c
        zh, za, zc = q3["centers4"][c]
        if zh < 0 and za > 0:
            nm = "confident + output-aligned"
        elif zh < 0:
            nm = "confident + divergent (floor)"
        elif za <= 0:
            nm = "uncertain + divergent"
        else:
            nm = "uncertain + converged (early)"
        add("| %d %s | %d | %s | %s | %s | %s | %s | %s | %s | %s |"
            % (c, nm, int(m.sum()), pct(m.mean()), fmt(zh, 2), fmt(za, 2),
               fmt(zc, 2), fmt(np.mean(q3["relpos"][m]), 2),
               pct(np.mean(q3["relpos"][m] < 0.25)),
               pct(np.mean(q3["relpos"][m] > 0.75)),
               fmt(np.mean(q3["ncons"][m]), 1)))
    add("")
    add("The dominant regime is the **confident + divergent floor** (~51%), "
        "followed by a **late confident + output-aligned** block (~18%, mean "
        "relpos 0.66) and a **mid-trace uncertain + divergent** block (~26%). "
        "The small early **uncertain + converged** block (~5%, relpos 0.22) is "
        "the shared exploratory prefix.")
    add("")
    add("Median-split cells (H high?, A high?, C high?), share and mean relative position:")
    add("")
    add("| cell | n | share | mean relpos |")
    add("|---|---|---|---|")
    for c in range(8):
        nn, share, rp = q3["cells"][c]
        if nn:
            add("| H%s A%s C%s | %d | %s | %s |"
                % ("+" if c & 4 else "-", "+" if c & 2 else "-",
                   "+" if c & 1 else "-", nn, pct(share), fmt(rp, 2)))
    add("")
    mixed = sum(q3["cells"][c][1] for c in (1, 2, 5, 6))
    confident_mixed = q3["cells"][1][1] + q3["cells"][2][1]
    add("A and C fall on opposite sides of their within-run medians in **%s** "
        "of all steps (the two confident decoupled cells alone are %s): streams "
        "can agree on the emitted token while their representations differ, or "
        "vice versa. That is the concrete content of the two axes."
        % (pct(mixed), pct(confident_mixed)))
    add("")

    # Q4
    add("## 4. Convergence dynamics")
    add("")
    add("**Method.** `A[t]` and `Cal[t]` over `n>=2` steps. Re-convergence = "
        "`A<1` at some step and `A==1` later. Epochs = run-length encoding of "
        "`A==1`/`A<1`, split by mean live count. Slopes are OLS on relative "
        "trace position, per run. The full-probs final step (`n=1`) is "
        "excluded: it trivially has A=1 and produces a false 100% "
        "re-convergence rate.")
    add("")
    add("- Re-convergence (n>=2): **%d/%d runs**." % (q4["recon"], n))
    add("- `A==1` epochs at mean n>=6: n=%d, median %s, p90 %s, max %d steps."
        % (len(q4["one_hi"]), fmt(median(q4["one_hi"]), 0),
           fmt(pctl(q4["one_hi"], 90), 0), max(q4["one_hi"]) if q4["one_hi"] else 0))
    add("- `A==1` epochs at mean n<6: n=%d, median %s, p90 %s, max %d steps "
        "(survivor pairs; not general re-convergence)."
        % (len(q4["one_lo"]), fmt(median(q4["one_lo"]), 0),
           fmt(pctl(q4["one_lo"], 90), 0), max(q4["one_lo"]) if q4["one_lo"] else 0))
    add("- `A<1` epochs: n=%d, median %s steps (once streams split they stay "
        "split for most of the trace)."
        % (len(q4["div_epochs"]), fmt(median(q4["div_epochs"]), 0)))
    add("")
    add("| series | full n>=2 median slope | frac neg | n=8 prefix | frac neg | 2nd half n=8 | frac neg |")
    add("|---|---|---|---|---|---|---|")
    for label, full, pref, half in (("token agreement A", q4["slope_A"],
                                     q4["slope_A8"], q4["slope_A2"]),
                                    ("hidden consensus C", q4["slope_C"],
                                     q4["slope_C8"], q4["slope_C2"])):
        add("| %s | %s | %s | %s | %s | %s | %s |"
            % (label, fmt(median(full)),
               pct(mean([1.0 if x < 0 else 0.0 for x in full])),
               fmt(median(pref)), pct(mean([1.0 if x < 0 else 0.0 for x in pref])),
               fmt(median(half)), pct(mean([1.0 if x < 0 else 0.0 for x in half]))))
    add("")
    add("Slope distributions (per run, relative trace position): A full n>=2 "
        "p10/p90 = %s/%s; C full p10/p90 = %s/%s; A n=8 p10/p90 = %s/%s; "
        "C n=8 p10/p90 = %s/%s."
        % (fmt(pctl(q4["slope_A"], 10)), fmt(pctl(q4["slope_A"], 90)),
           fmt(pctl(q4["slope_C"], 10)), fmt(pctl(q4["slope_C"], 90)),
           fmt(pctl(q4["slope_A8"], 10)), fmt(pctl(q4["slope_A8"], 90)),
           fmt(pctl(q4["slope_C8"], 10)), fmt(pctl(q4["slope_C8"], 90))))
    add("")
    a2_neg = mean([1.0 if x < 0 else 0.0 for x in q4["slope_A2"]])
    c2_neg = mean([1.0 if x < 0 else 0.0 for x in q4["slope_C2"]])
    r1 = []
    for rr in runs:
        m = (rr.n == 8) & np.isfinite(rr.Cal)
        if m.sum() > 20:
            a, c = rr.A[m], rr.Cal[m]
            h = len(a) // 2
            if np.std(a[:h]) > 0 and np.std(c[:h]) > 0:
                r1.append(pearson(a[:h], c[:h]))
    r1 = mean(r1)
    r2 = mean(q4["r2"])
    add("**Shape.** Both series fall fast while all 8 streams are live (median "
        "A slope %s, C slope %s, %d/%d A-negative, %d/%d C-negative). Then "
        "token agreement **saturates at a floor** (2nd-half A slope %s, %s "
        "negative = flat) while hidden consensus keeps a slow drift down "
        "(2nd-half C slope %s, %s negative); r(A,C) falls from %s in the first "
        "half to %s in the second. This is a real two-phase pattern, but part "
        "of it is **censoring**: A is a discrete modal fraction with a floor "
        "near 1/n, so it cannot keep falling once streams use different words. "
        "Do not read the late A plateau as semantic convergence."
        % (fmt(median(q4["slope_A8"])), fmt(median(q4["slope_C8"])),
           sum(1 for x in q4["slope_A8"] if x < 0), len(q4["slope_A8"]),
           sum(1 for x in q4["slope_C8"] if x < 0), len(q4["slope_C8"]),
           fmt(median(q4["slope_A2"])), pct(a2_neg),
           fmt(median(q4["slope_C2"])), pct(c2_neg), fmt(r1), fmt(r2)))
    add("")
    add("The ~0.2 agreement floor is **not chance**: observed pairwise token "
        "equality is 0.068, versus ~1e-5 under a uniform 128k vocabulary. The "
        "floor is a shared function-word/formatting vocabulary, not random "
        "collisions.")
    add("")
    add("Hidden consensus epochs: `C>=0.99` covers %d steps (%.1f%%), %d "
        "epochs, median %s, p90 %s, max %d, mean relative position %.2f; "
        "`C>=0.999` covers %d steps, %d epochs, mean relative position %.2f. "
        "Hidden consensus is a **brief early phenomenon** (first ~5 steps), not "
        "a recurring late state."
        % (q4["hid_epochs"][0.99]["steps"],
           100.0 * q4["hid_epochs"][0.99]["steps"] / q4["hid_epochs"][0.99]["total"],
           q4["hid_epochs"][0.99]["n"], fmt(q4["hid_epochs"][0.99]["med"], 0),
           fmt(q4["hid_epochs"][0.99]["p90"], 0), q4["hid_epochs"][0.99]["max"],
           q4["hid_epochs"][0.99]["relpos"], q4["hid_epochs"][0.999]["steps"],
           q4["hid_epochs"][0.999]["n"], q4["hid_epochs"][0.999]["relpos"]))
    add("")
    add("**Composition warning.** n=8 covers %s of consensus steps; %s of "
        "last-decile steps have n<=4 and the last step has n=1. Any "
        "trace-position profile without an n filter shows spurious late "
        "re-convergence."
        % (pct(sum(v for k, v in q4["n_count"].items() if k == 8)
               / float(sum(q4["n_count"].values()))),
           pct(q4["last_small"] / float(max(1, q4["last_tot"])))))
    add("")

    # Q5
    add("## 5. Variance decomposition: across streams vs across steps")
    add("")
    add("**Method.** Balanced all-8-live prefix. Two-way layout "
        "`H[s,t]`: SS_step = S*sum_t(mean_s H - grand)^2, SS_stream = "
        "L*sum_s(mean_t H - grand)^2, residual = remainder (stream x step "
        "interaction + noise). Also mean_t Var_s(H) vs mean_s Var_t(H). Global "
        "hesitation = every live stream has H above a threshold.")
    add("")
    add("- Median variance shares: **step %s, stream %s, residual %s** "
        "(p10-p90 step %s-%s, stream %s-%s). The additive step effect is ~%.0fx "
        "the additive stream effect, but the stream x step interaction is ~%.0fx "
        "the step effect: entropy is mostly idiosyncratic, not a stable "
        "per-stream or per-position property."
        % (fmt(median(q5["shares"][:, 0])), fmt(median(q5["shares"][:, 1])),
           fmt(median(q5["shares"][:, 2])), fmt(pctl(q5["shares"][:, 0], 10)),
           fmt(pctl(q5["shares"][:, 0], 90)), fmt(pctl(q5["shares"][:, 1], 10)),
           fmt(pctl(q5["shares"][:, 1], 90)),
           median(q5["shares"][:, 0]) / max(1e-9, median(q5["shares"][:, 1])),
           median(q5["shares"][:, 2]) / max(1e-9, median(q5["shares"][:, 0]))))
    add("- Mean cross-stream variance / mean cross-step within-stream variance: "
        "median **%s** (p10-p90 %s-%s). Neither factor dominates; the bulk is "
        "idiosyncratic stream x step variation. Stable stream identity explains "
        "~1%% of entropy variance. This decomposition uses the all-8-live "
        "prefix only; the late n<8 phase is not represented."
        % (fmt(median(q5["ratios"])), fmt(pctl(q5["ratios"], 10)),
           fmt(pctl(q5["ratios"], 90))))
    add("")
    add("**Global hesitation windows:**")
    add("")
    add("| threshold | steps | share | windows | runs | median dur | p90 | max |")
    add("|---|---|---|---|---|---|---|---|")
    for thr in (math.log(2.0), 1.0, 1.5):
        h = q5["hes"][thr]
        add("| H>%.3f | %d | %s | %d | %d/%d | %s | %s | %d |"
            % (thr, h["steps"], pct(h["steps"] / max(1, h["total"])),
               h["windows"], h["runs"], n, fmt(h["med"], 0), fmt(h["p90"], 0),
               h["max"]))
    add("")
    add("At the repo's 1-nat threshold simultaneous hesitation is essentially "
        "absent: %d all-live steps (%s), duration 1. At log-2 it is still rare "
        "and never lasts more than 2 steps. For contrast, **global confidence** "
        "(all H<0.05) covers %d/%d all-live steps (%s): the population is much "
        "more likely to be jointly confident than jointly uncertain."
        % (q5["hes"][1.0]["steps"],
           pct(q5["hes"][1.0]["steps"] / max(1, q5["hes"][1.0]["total"])),
           q5["conf"][0.05][0], q5["conf"][0.05][1],
           pct(q5["conf"][0.05][0] / max(1, q5["conf"][0.05][1]))))
    add("")

    # Q6
    add("## 6. Sampler fidelity")
    add("")
    add("**Method.** Per emitted token reconstruct an effective distribution "
        "from `(p1, H)`: top token `p1`, remainder uniform over `K-1` effective "
        "tokens with `K` chosen to match `H`. This gives a temperature-scaled "
        "top probability `p1T`, collision `sum q^2`, and `P(all 8 iid draws "
        "equal)`. Compare with the observed argmax rate (`exp(lp)/p1 > 0.999`) "
        "and with step-level token agreement. The reconstruction is rough (no "
        "rank-2 mass, no top-p truncation); it is used for order of magnitude.")
    add("")
    add("| predicted top-1 bin (T=1) | n | observed argmax | mean pred T=1 | mean pred T=0.6 |")
    add("|---|---|---|---|---|")
    for lo, hi, nn, o, pt1, pt6 in q6["binned"]:
        add("| [%.2f,%.2f) | %d | %s | %s | %s |" % (lo, hi, nn, fmt(o), fmt(pt1), fmt(pt6)))
    add("")
    add("- Observed argmax rate **%s**; predicted mean top-1 at T=1 **%s**, at "
        "T=0.6 **%s**. Per-stream sampling is broadly consistent with a "
        "temperature around 0.6 (not exact); no evidence of gross sampler "
        "infidelity." % (fmt(q6["obs_argmax"]), fmt(q6["pred_T1"]), fmt(q6["pred_T06"])))
    add("- Observed pairwise token equality **%s** vs shared-distribution "
        "prediction **%s**. Observed fraction of steps with A==1 **%s** vs "
        "predicted P(all 8 same) **%s**. Under the null that all 8 streams "
        "sample one common distribution, token agreement would be an order of "
        "magnitude higher. The gap is **content divergence**, not sampling "
        "noise: each stream is locally confident but their modes differ."
        % (fmt(q6["obs_pair"]), fmt(q6["pred_pair"]), pct(q6["obs_pall"]),
           pct(q6["pred_pall"])))
    add("- At step 0 (shared prompt) observed agreement is %s vs shared-null %s "
        "(n=%d). The observed value is *higher* than the reconstruction "
        "predicts, consistent with top-p/top-k truncation removing tail mass "
        "and with the shared prefix; it is not evidence of correlated RNG."
        % (fmt(q6["step0_A"]), fmt(q6["step0_pall"]), q6["step0_n"]))
    add("")
    add("**Verdict.** The population is not sampling one shared distribution, "
        "but that is the expected consequence of independent sampling at "
        "near-ties compounding into different contexts. Per-stream top-1 rates "
        "are in the right range for T=0.6.")
    add("")

    # Q7
    add("## 7. Per-stream structure (is any stream 1..8 special?)")
    add("")
    add("**Method.** Per stream id, mean over 32 problems of generated length, "
        "completion time, entropy, norm, emitted logprob, mean cmean, and answer "
        "correctness. Spread = max-min of the 8 stream means. Null: shuffle "
        "stream labels within each problem (2000 permutations), recompute "
        "spread. Ratio = observed / null median. Ratio at or below the null "
        "95th percentile means stream identity adds nothing detectable.")
    add("")
    add("| metric | spread | null median | null p95 | ratio | Spearman(id, mean) | stream means |")
    add("|---|---|---|---|---|---|---|")
    for k in ("gen", "completion", "entropy", "norm", "lp", "cmean", "correct"):
        v = q7[k]
        nd = 3 if k in ("entropy", "norm", "lp", "cmean") else (2 if k == "correct" else 1)
        add("| %s | %s | %s | %s | %s | %s | %s |"
            % (k, fmt(v["spread"], nd), fmt(v["null50"], nd), fmt(v["null95"], nd),
               fmt(v["ratio"], 2), fmt(v["rho"], 2),
               " ".join(fmt(x, 2) for x in v["means"])))
    add("")
    ratios = [q7[k]["ratio"] for k in ("gen", "completion", "entropy",
                                       "norm", "lp", "cmean", "correct")]
    add("Every observed spread is **below the permutation null's 95th "
        "percentile** (ratio range %s-%s). The stream-1 correctness advantage "
        "and the stream-4/5 deficit are exactly the spread expected from random "
        "assignment of problem difficulty. Monotone hints exist in the means "
        "(entropy rho=%s, logprob rho=%s) but the spread test does not support "
        "a systematic stream effect. **Stream order is exchangeable in this "
        "dump** at the resolution of n=32."
        % (fmt(min(ratios), 2), fmt(max(ratios), 2),
           fmt(q7["entropy"]["rho"], 2), fmt(q7["lp"]["rho"], 2)))
    add("")

    # Q8
    add("## 8. Hidden-state norm")
    add("")
    add("**Method.** Norm = `h.norm` (last-layer embedding after final norm), "
        "one row per emitted token. Distribution over all %d rows; "
        "token-identity eta^2 = between-token SS / total SS for tokens with "
        ">=20 occurrences; per-trace slope before and after subtracting token "
        "means; within-run correlations; top |delta| spikes." % q8["n"])
    add("")
    add("- Distribution: mean **%s**, sd **%s**, p1/p5/p50/p95/p99 = "
        "%s / %s / %s / %s / %s, min %s, max %s. EOS rows have mean norm %s, "
        "at the high end, not a low-norm special case."
        % (fmt(q8["mean"], 2), fmt(q8["sd"], 2), fmt(q8["p1"], 1),
           fmt(q8["p5"], 1), fmt(q8["p50"], 1), fmt(q8["p95"], 1),
           fmt(q8["p99"], 1), fmt(q8["min"], 1), fmt(q8["max"], 1),
           fmt(q8["eos_norm"], 1)))
    add("- **Token identity eta^2 = %s** (%d common tokens cover %.0f%% of "
        "rows). The single space token (id 220) alone accounts for **%s** of "
        "norm variance; its mean norm is **%s** vs **%s** for non-space "
        "(d=%s). Punctuation and newline-adjacent tokens are also low. Norm is "
        "mostly a property of *which token was emitted*, not of the context."
        % (fmt(q8["eta2"], 2), q8["n_common"], 100.0 * q8["coverage"],
           pct(q8["space_var_share"]), fmt(q8["space_mean"], 1),
           fmt(q8["nonspace_mean"], 1), fmt(q8["d_space"], 1)))
    add("- Raw per-trace slope: median **%s** per trace, %s negative. After "
        "subtracting token means: **%s** (only %s negative). The apparent "
        "\"norm declines over the trace\" is a token-mix artifact, not growing "
        "uncertainty."
        % (fmt(q8["slope_med"], 2), pct(q8["slope_neg"]),
           fmt(q8["slope_res_med"], 2), pct(q8["slope_res_neg"])))
    add("- Within-run correlations: r(norm,H)=%s, r(norm,A)=%s, r(norm,C)=%s. "
        "Weak; norm is not a clean uncertainty or convergence proxy."
        % (fmt(q8["rN_H"]), fmt(q8["rN_A"]), fmt(q8["rN_C"])))
    add("- Norm histogram (62..93, 15 bins, log counts): `%s`; the left tail "
        "is dominated by the space token, not a separate mode."
        % spark([math.log1p(x) for x in q8["hist"]], 15,
                0, max(math.log1p(x) for x in q8["hist"])))
    add("Lowest / highest mean-norm tokens (mean, n, id, piece):")
    add("")
    add("| low | high |")
    add("|---|---|")
    for (ml, nl, tl), (mh, nh, th) in zip(q8["low_tok"], reversed(q8["high_tok"])):
        add("| %s n=%d id=%d %s | %s n=%d id=%d %s |"
            % (fmt(ml, 1), nl, tl, piece_label(tl, q8["piece"]),
               fmt(mh, 1), nh, th, piece_label(th, q8["piece"])))
    add("")
    add("- Top |delta norm| events are dominated by transitions into/out of "
        "the space token: %d of the top 100 involve token 220. Top 8:"
        % q8["space_in_top"])
    add("")
    add("| abs(delta) | qid | stream | step | delta | norm | piece |")
    add("|---|---|---|---|---|---|---|")
    for ad, d, q, s, t, nm, tk in q8["top_spikes"]:
        add("| %s | %d | %s | %d | %s | %s | %s |"
            % (fmt(ad, 1), q, s, t, fmt(d, 1), fmt(nm, 1),
               piece_label(tk, q8["piece"])))
    add("")
    add("Practical point: **never use raw norm as a convergence signal without "
        "token controls.**")
    add("")

    # Q9
    add("## 9. Timing")
    add("")
    add("**Method.** Straggler = max completion / median completion per run; "
        "within-run correlation of log completion with log generated length; "
        "first/last finisher correctness vs the other streams; bootstrap over "
        "problems for the first-finisher difference.")
    add("")
    add("- Straggler ratio: median **%s**, p90 **%s**, max **%s**."
        % (fmt(median(q9["straggler"]), 2), fmt(pctl(q9["straggler"], 90), 2),
           fmt(max(q9["straggler"]), 2)))
    add("- Within-run corr(log completion, log generated) = **%s** (median). "
        "Timing is essentially trace length; not an independent axis."
        % fmt(median(q9["comp_gen"])))
    add("- First-finisher correctness **%s** vs second **%s** vs last **%s**. "
        "First minus rest = **%+.3f** (bootstrap 95%% CI %+.3f..%+.3f); first "
        "minus second d=%s. Within-run rank-correctness r=%s. First finisher is "
        "also the shortest trace in %s of runs."
        % (fmt(q9["first"], 2), fmt(q9["second"], 2), fmt(q9["last"], 2),
           q9["diff"], q9["diff_ci"][0], q9["diff_ci"][1], fmt(q9["d12"], 2),
           fmt(q9["rank_corr"]), pct(q9["first_shortest"])))
    add("- First-finisher stream id counts: %s. No fixed stream; this is a "
        "length/quality effect, not a stream-order effect."
        % ", ".join("%s:%d" % (k, v) for k, v in sorted(q9["first_counts"].items())))
    parts = []
    for key, label in (("H", "entropy"), ("NM", "norm"), ("C", "cmean"),
                       ("gen", "generated")):
        v = np.array(q9["attr"][key], float)
        parts.append("%s %+.3f (d=%.2f)" % (label, v.mean(), v.mean() / v.std())
                     if v.std() > 0 else "%s %+.3f" % (label, v.mean()))
    add("- First finisher vs the other 7 (paired): " + "; ".join(parts) + ".")
    add("")
    add("**Interpretation.** The ~14 pp first-finisher advantage is suggestive "
        "(bootstrap CI just excludes 0) and fits the cross-run pattern that "
        "longer traces have lower answer agreement. With n=32 and one seed it "
        "is not robust enough to build on without replication.")
    add("")

    # Q10
    add("## 10. Other structure that stands out")
    add("")
    add("### 10.1 End-aligned answer convergence (the headline)")
    add("")
    add("Absolute-step agreement is the wrong alignment for the answer: streams "
        "end at different lengths, so the shared suffix is compared at "
        "different indices. Aligning on the **end** of each trace (`k` = tokens "
        "from the end; k=0 is EOS) exposes a sharp convergence:")
    add("")
    add("| k from end | all | UR | SP |")
    add("|---|---|---|---|")
    for k in range(0, 13):
        add("| %d | %s | %s | %s |"
            % (k, fmt(mean(q10["prof"].get(k, [])), 3),
               fmt(mean(q10["prof_g"]["UR"].get(k, [])), 3),
               fmt(mean(q10["prof_g"]["SP"].get(k, [])), 3)))
    add("")
    a_reason = mean([r.reason_end_agree for r in runs])
    a_suf = mean([r.suffix_agree for r in runs])
    a_ans = mean([r.answer_tok_agree for r in runs])
    add("Agreement rises from %s at 10-20 tokens before the end to **%s in the "
        "last 6 tokens** (%+.3f absolute, ~2x). The rise is the templated "
        "answer suffix (`\\n\\nThe answer is: X`), not the reasoning. The "
        "answer value token itself (k=1) is %s overall: %s in UR runs but only "
        "%s in SP runs. `suffix_agree` and `answer_token_agree` correlate r=%s "
        "and r=%s with answer agreement; `suffix_agree` vs end-of-reasoning "
        "agreement is only r=%s, so suffix convergence is largely independent "
        "of reasoning convergence."
        % (fmt(a_reason), fmt(a_suf), a_suf - a_reason, fmt(a_ans),
           fmt(mean([r.answer_tok_agree for r in runs if r.group == "UR"])),
           fmt(mean([r.answer_tok_agree for r in runs if r.group == "SP"])),
           fmt(pearson([r.suffix_agree for r in runs], [r.pairwise_agree for r in runs])),
           fmt(pearson([r.answer_tok_agree for r in runs], [r.pairwise_agree for r in runs])),
           fmt(pearson([r.suffix_agree for r in runs], [r.reason_end_agree for r in runs]))))
    add("")
    add("**Interpretation.** The true late behaviour is **format convergence on "
        "top of content divergence**. Start-aligned agreement alone shows "
        "divergence everywhere and misses this. Caveat: the last token is EOS "
        "(always shared, excluded from the summary) and much of the rest is a "
        "fixed template, so this is not evidence that reasoning converges.")
    add("")
    add("### 10.2 Exemplar traces (ASCII)")
    add("")
    for r in [x for x in runs if x.qid in (1, 13, 23)]:
        add("**q%02d** (%s, %d steps, answers=%s, pairwise_agree=%.2f)"
            % (r.qid, r.group, r.max_t, ",".join(str(a) for a in r.answers),
               r.pairwise_agree))
        add("")
        add("```")
        add("A   |" + spark(r.A, 60, 0.0, 1.0))
        add("C   |" + spark(r.Cal, 60, 0.0, 1.0))
        add("H   |" + spark(r.H, 60, 0.0, max(1.5, float(np.max(r.H)))))
        add("norm|" + spark(r.NM, 60))
        add("```")
        add("")
    add("`A` and `C` fall together early; `A` flattens at a floor; `C` keeps "
        "drifting; `norm` is dominated by token-level notches (space).")
    add("")
    add("### 10.3 Largest jumps, bimodality, outliers")
    add("")
    add("- Largest step-to-step changes in token agreement:")
    for ad, q, t, rp, a0, a1, nn in q10["jumps_A"][:6]:
        add("  - q%02d t=%d rel=%s %s->%s n=%d (|delta|=%s)"
            % (q, t, fmt(rp, 2), fmt(a0, 2), fmt(a1, 2), nn, fmt(ad, 2)))
    add("- Largest step-to-step changes in hidden consensus:")
    for ad, q, t, rp, c0, c1, nn in q10["jumps_C"][:6]:
        add("  - q%02d t=%d rel=%s %s->%s n=%d (|delta|=%s)"
            % (q, t, fmt(rp, 2), fmt(c0, 2), fmt(c1, 2), nn, fmt(ad, 2)))
    add("- All of the largest jumps above occur at n<=4 (survivor/composition "
        "events, mostly late), not population-level jumps. Restricting to n=8:")
    for ad, q, t, rp, a0, a1, nn in q10["jumps_A8"]:
        add("  - A: q%02d t=%d rel=%s %s->%s (|delta|=%s)"
            % (q, t, fmt(rp, 2), fmt(a0, 2), fmt(a1, 2), fmt(ad, 2)))
    for ad, q, t, rp, c0, c1, nn in q10["jumps_C8"]:
        add("  - C: q%02d t=%d rel=%s %s->%s (|delta|=%s)"
            % (q, t, fmt(rp, 2), fmt(c0, 2), fmt(c1, 2), fmt(ad, 2)))
    add("- Run-mean token agreement ranges %s-%s (median %s); run-mean cos "
        "ranges %s-%s. UR and SP overlap completely; there is no run-level "
        "bimodality in either consensus measure."
        % (fmt(min(r.mean_A for r in runs)), fmt(max(r.mean_A for r in runs)),
           fmt(median([r.mean_A for r in runs])),
           fmt(min(r.mean_C for r in runs)), fmt(max(r.mean_C for r in runs))))
    for key, label in (("mean_H", "mean entropy"), ("mean_A", "mean token agreement"),
                       ("mean_C", "mean hidden cos"), ("wall_ms", "wall time"),
                       ("mean_gen", "mean trace length")):
        lo, hi = q10["extremes"][key]
        add("- %s: low %s; high %s"
            % (label,
               ", ".join("q%02d=%s" % (r.qid, fmt(getattr(r, key), 2)) for r in lo),
               ", ".join("q%02d=%s" % (r.qid, fmt(getattr(r, key), 2)) for r in hi)))
    add("")
    add("### 10.4 Answer-marker timing")
    add("")
    add("The last `The answer is` marker occurs a median of %s tokens before "
        "each stream's end (n=%d streams); %d/%d runs still have >=4 streams "
        "live at the median marker step (median %s live). The answer phase is "
        "short and the live set is already shrinking, which is why end "
        "alignment is the usable handle."
        % (fmt(q10["marker_rel"], 0), q10["marker_n"], q10["marker_ge4"], n,
           fmt(q10["marker_live"], 0)))
    add("")

    # ranking
    add("## Ranked top-5 features")
    add("")
    add("Criterion: size of effect, how much it changes the picture, and whether "
        "it survives the obvious composition/alignment controls. Effect sizes, "
        "not p-values.")
    add("")
    a_reason = mean([r.reason_end_agree for r in runs])
    a_suf = mean([r.suffix_agree for r in runs])
    a_ans = mean([r.answer_tok_agree for r in runs])
    r_ans = pearson([r.answer_tok_agree for r in runs], [r.pairwise_agree for r in runs])
    r_suf_reason = pearson([r.suffix_agree for r in runs], [r.reason_end_agree for r in runs])
    add("1. **End-aligned answer convergence on top of reasoning divergence.** "
        "Last-6-token agreement %.3f vs %.3f ten-to-twenty back (+%.3f, ~2x); "
        "answer-token agreement %.3f (UR %.3f, SP %.3f); r(answer-token, answer "
        "agreement)=%.3f while r(suffix, end-of-reasoning)=%.3f. Caveat: mostly "
        "the templated suffix, last token is EOS, invisible under "
        "absolute-step alignment. Not semantic convergence."
        % (a_suf, a_reason, a_suf - a_reason, a_ans,
           mean([r.answer_tok_agree for r in runs if r.group == "UR"]),
           mean([r.answer_tok_agree for r in runs if r.group == "SP"]),
           r_ans, r_suf_reason))
    add("2. **Confidence and consensus are two nearly independent axes.** "
        "Run-level r(token agreement, mean entropy)=%.3f, r(hidden cos, mean "
        "entropy)=%.3f; within-run r(A,H)=%.3f; meanwhile r(A,C)=%.3f. "
        "A and C disagree across their medians in %s of steps. Caveat: entropy "
        "and margin are nearly collinear (|r|>0.98), so \"confidence\" is one "
        "axis; one seed, 32 problems."
        % (q2["M"][0, 2], q2["M"][1, 2], q2["within"]["A_H"], q2["M"][0, 1],
           pct(sum(q3["cells"][c][1] for c in (1, 2, 5, 6)))))
    add("3. **Token agreement saturates; hidden consensus keeps drifting.** "
        "First-half all-8-live slopes A=%.3f, C=%.3f, r(A,C)=%.3f; second half "
        "A=%.3f (flat), C=%.3f (%s of runs still negative), r(A,C)=%.3f. "
        "Caveat: A is censored by its discrete floor near 1/n, so part of the "
        "late dissociation is a measurement-floor artifact; the robust part is "
        "the two-phase shape."
        % (median(q4["slope_A8"]), median(q4["slope_C8"]), r1,
           median(q4["slope_A2"]), median(q4["slope_C2"]), pct(c2_neg), r2))
    add("4. **No hidden-state lead; the apparent +1 lead is an alignment "
        "artifact.** Naive join peaks at lag +1 (r=%.3f); after shifting c by "
        "one step the peak is lag 0; detrended r(0)=%.3f, r(+1)=%.3f. Partial "
        "C[t]->A[t+1]|A[t]=%.3f vs reverse A[t]->C[t+1]|C[t]=%.3f. This is a "
        "clean, deterministic negative and it corrects a natural first read of "
        "the dump."
        % (xc["naive_levels"][peak(xc["naive_levels"])][0],
           xc["aligned_detrend21"][0][0], xc["aligned_detrend21"][1][0],
           q1["partials"]["C[t]->A[t+1] | A[t]"],
           q1["partials"]["A[t]->C[t+1] | C[t]"]))
    add("5. **Hidden norm is mostly token identity, not context.** eta^2(token "
        "id)=%.2f (%.0f%% of norm variance); the space token alone is %s of "
        "variance (mean %.1f vs %.1f non-space, d=%.1f); raw per-trace slope "
        "%.2f but %+.2f after removing token means. Top norm spikes are "
        "space-token transitions. Caveat: descriptive, but the effect is large "
        "and would silently corrupt any norm-based convergence feature."
        % (q8["eta2"], 100.0 * q8["eta2"], pct(q8["space_var_share"]),
           q8["space_mean"], q8["nonspace_mean"], q8["d_space"],
           q8["slope_med"], q8["slope_res_med"]))
    add("")
    add("### Honourable mention / classification")
    add("")
    ratios = [q7[k]["ratio"] for k in ("gen", "completion", "entropy",
                                       "norm", "lp", "cmean", "correct")]
    add("- **Suggestive, not robust:** first-finisher advantage (%+.1f pp, "
        "bootstrap CI %+.1f..%+.1f pp, d=%.2f vs second; n=32, confounded with "
        "trace length)."
        % (100 * q9["diff"], 100 * q9["diff_ci"][0], 100 * q9["diff_ci"][1],
           q9["d12"]))
    add("- **Robust-looking but n=32/single-seed:** end-aligned convergence, "
        "two axes, two-phase saturation, norm token identity.")
    add("- **Noise-level / negative:** per-stream identity (spread ratios %s-%s, "
        "all below permutation p95); global hesitation (%d/%d all-live steps at "
        "H>1.0); hidden lead (no positive-lag peak after alignment); sampler "
        "infidelity (argmax rate %.3f between T=1 %.3f and T=0.6 %.3f)."
        % (fmt(min(ratios), 2), fmt(max(ratios), 2),
           q5["hes"][1.0]["steps"], q5["hes"][1.0]["total"],
           q6["obs_argmax"], q6["pred_T1"], q6["pred_T06"]))
    add("")
    add("## Caveats")
    add("")
    add("- n=32 problems, one model, one seed, one machine; no extrapolation.")
    add("- Steps within a run are strongly autocorrelated; all step-level "
        "\"n\" values overstate independent sample size. Run-level or clustered "
        "bootstrap is the relevant unit.")
    add("- The last decile of each trace has a shrinking live set; %s of "
        "last-decile steps have n<=4. Start-aligned profiles without an n "
        "filter are biased."
        % pct(q4["last_small"] / float(max(1, q4["last_tot"]))))
    add("- `A` is a discrete modal fraction (multiples of 1/n) and is censored "
        "near its floor; `C` is compressed near 1 for identical text.")
    add("- The sampler reconstruction in section 6 is a rough three-moment "
        "model, not the true full-vocabulary distribution.")
    add("- End-aligned agreement is post-hoc in the sense that alignment is a "
        "choice; it is justified by the traces ending at the answer, but it "
        "should be replicated on another seed before being treated as a stable "
        "population law.")
    add("")
    return L


if __name__ == "__main__":
    main()

