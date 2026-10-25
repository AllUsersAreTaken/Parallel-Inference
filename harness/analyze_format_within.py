r"""Within-run prompt-format lever: plain cot vs wrapped cotw.

Reads ONE 16-stream run produced by harness/engine/modality.cpp:

    streams grouped by their recorded mode, 8 plain (mode=cot) + 8 wrapped
    (mode=cotw). With --swap-parity the two modes occupy opposite halves on
    consecutive problem ids, so a mode's position in the batch alternates and
    position effects cancel. The seeds are paired: stream j and j+8 share S_j.

Every arm lives in the same batched decode for a given problem, so the batch
shape is identical across arms; only the user-message formatting differs. The
wrapped arm's user text is exactly "Question: " + <question> + "\nSolution:"
(the s4 --wrapper text with no exemplar); its system prompt is the same SYS_COT
the plain arm uses.

Reported per mode group: tie-aware plurality (point / random-tie expectation /
upper / tie count), oracle@8, per-stream (per-seed-slot) accuracy, gold absence,
and within-group wrong-answer agreement. Then wrapper minus plain: exact
McNemar and a tie-aware paired bootstrap (2000 resamples, seed 1234, 95% CI),
plus first-K-agree (K=3) by done_ms per group with reached-only accuracy.

Verdict: PASS if wrapper >= plain + 3 pts with a tie-aware CI excluding 0;
DEAD if wrapper <= plain.

Usage:
    python harness/analyze_format_within.py \
      --run m6_1b,results/m6_1b_format.tsv,results/m6_1b_provenance.txt \
      --run m6_coder15b,results/m6_coder15b_format.tsv,results/m6_coder15b_provenance.txt \
      --out results/format_within_summary.txt
"""
import argparse
import csv
import os
import statistics
import sys
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import analyze_modality_vote as M  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
DEFAULT_GOLD = os.path.join(ROOT, "data", "gsm8k_dev200.tsv")
DEFAULT_OUT = os.path.join(RESULTS, "format_within_summary.txt")

HEADER = ["id", "stream", "mode", "finish", "generated", "ttft_ms", "done_ms", "text"]
FINISHES = ("eos", "answer", "fence", "cap")
PLAIN = "cot"
WRAP = "cotw"
GROUP_ORDER = (PLAIN, WRAP)
N_STREAMS = 16
HALF = 8
K_AGREE = 3
TIE_BOOTSTRAP_N = 2000
TIE_BOOTSTRAP_SEED = 1234


# ---------------------------------------------------------------- loading ----


def load_rows(path):
    """Parse a modality TSV; accepts any mode string (cotw included)."""
    rows = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        header = fh.readline().rstrip("\r\n").split("\t")
        if header != HEADER:
            raise SystemExit(f"{path}: unexpected header {header!r}; expected {HEADER!r}")
        for lineno, line in enumerate(fh, 2):
            line = line.rstrip("\r\n")
            if not line:
                continue
            parts = line.split("\t", 7)
            if len(parts) != 8:
                raise SystemExit(f"{path}:{lineno}: expected 8 tab fields, got {len(parts)}")
            try:
                row = {
                    "id": int(parts[0]),
                    "stream": int(parts[1]),
                    "mode": parts[2],
                    "finish": parts[3],
                    "generated": int(parts[4]),
                    "ttft_ms": float(parts[5]),
                    "done_ms": float(parts[6]),
                    "text": M.unescape_tsv(parts[7]),
                }
            except ValueError as exc:
                raise SystemExit(f"{path}:{lineno}: bad numeric field ({exc})")
            if row["finish"] not in FINISHES:
                raise SystemExit(f"{path}:{lineno}: bad finish {row['finish']!r}")
            rows.append(row)
    if not rows:
        raise SystemExit(f"{path}: no rows")

    by_id = {}
    for r in rows:
        seen = by_id.setdefault(r["id"], {})
        if r["stream"] in seen:
            raise SystemExit(f"{path}: duplicate id={r['id']} stream={r['stream']}")
        seen[r["stream"]] = r
    return rows, by_id


def load_provenance(path):
    if not path or not os.path.exists(path):
        return None, path or "(none)"
    d = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                k, v = line.split(":", 1)
            elif "=" in line:
                k, v = line.split("=", 1)
            else:
                continue
            d[k.strip()] = v.strip()
    return d, path


def load_run(path):
    _, by_id = load_rows(path)
    for i, streams in by_id.items():
        got = set(streams)
        if got != set(range(N_STREAMS)):
            raise SystemExit(
                f"{path}: id={i} has streams {sorted(got)}; expected 0..{N_STREAMS - 1}")
        counts = {}
        for s in range(N_STREAMS):
            counts[streams[s]["mode"]] = counts.get(streams[s]["mode"], 0) + 1
        if counts != {PLAIN: HALF, WRAP: HALF}:
            raise SystemExit(
                f"{path}: id={i} mode counts {counts}; expected {PLAIN}:{HALF} {WRAP}:{HALF}")
        for h in (0, 1):
            modes = {streams[s]["mode"] for s in range(h * HALF, (h + 1) * HALF)}
            if len(modes) != 1:
                raise SystemExit(
                    f"{path}: id={i} half {h} is not a single mode: {sorted(modes)}")
        if streams[0]["mode"] == streams[HALF]["mode"]:
            raise SystemExit(f"{path}: id={i} both halves have mode {streams[0]['mode']!r}")
    return by_id


def answer_of(row):
    return M.extract_cot(row["text"])[0]


# ------------------------------------------------------------ group helpers ----


def group_indices(by_id_i, mode):
    """Stream indices of one mode for a problem, in ascending order."""
    return [s for s in sorted(by_id_i) if by_id_i[s]["mode"] == mode]


def group_plurality(by_id, ids, mode):
    res = {}
    for i in ids:
        idxs = group_indices(by_id[i], mode)
        items = [(answer_of(by_id[i][s]), mode, s) for s in idxs]
        res[i] = M.plurality(items)
    return res


def group_oracle(by_id, ids, mode, gold):
    oracle, support = {}, {}
    for i in ids:
        g = M.canon(gold[i])
        c = 0
        for s in group_indices(by_id[i], mode):
            a = answer_of(by_id[i][s])
            if a is not None and M.canon(a) == g:
                c += 1
        support[i] = c
        oracle[i] = c > 0
    return oracle, support


def group_slot_accuracy(by_id, ids, mode, gold):
    """Accuracy by the stream's slot inside the group (0..7).

    The slot maps to a fixed seed S_slot because streams j and j+8 share it:
    on even ids the plain group is slots 0..7 (streams 0..7) and the wrapped
    group is slots 0..7 (streams 8..15), and the reverse on odd ids.
    """
    out = {}
    for i in ids:
        g = M.canon(gold[i])
        for slot, s in enumerate(group_indices(by_id[i], mode)):
            d = out.setdefault(slot, {"total": 0, "answered": 0, "correct": 0})
            d["total"] += 1
            a = answer_of(by_id[i][s])
            if a is not None:
                d["answered"] += 1
                d["correct"] += 1 if M.canon(a) == g else 0
    return out


def wrong_answer_agreement(by_id, ids, mode, gold):
    """Within-group agreement over wrong answers (both-answered pairs).

    For every problem, only streams whose answer differs from gold vote. Pairs
    of wrong answers are counted when both are non-None; ``agree`` counts equal
    canonical values. Also reports the unrestricted both-answered agreement.
    """
    pair_both = pair_agree = 0
    all_pair_both = all_pair_agree = 0
    problems_with_two_wrong = 0
    for i in ids:
        g = M.canon(gold[i])
        vals = [answer_of(by_id[i][s]) for s in group_indices(by_id[i], mode)]
        keys = [M.canon(a) for a in vals if a is not None]
        wrong = [k for k in keys if k != g]
        if len(wrong) >= 2:
            problems_with_two_wrong += 1
        for a, b in combinations(wrong, 2):
            pair_both += 1
            pair_agree += 1 if a == b else 0
        for a, b in combinations(keys, 2):
            all_pair_both += 1
            all_pair_agree += 1 if a == b else 0
    return {
        "pair_both": pair_both,
        "pair_agree": pair_agree,
        "pair_frac": M.pct(pair_agree, pair_both) if pair_both else float("nan"),
        "all_pair_both": all_pair_both,
        "all_pair_agree": all_pair_agree,
        "all_pair_frac": M.pct(all_pair_agree, all_pair_both) if all_pair_both else float("nan"),
        "problems_with_two_wrong": problems_with_two_wrong,
    }


def first_k_agree(entries, k=K_AGREE):
    """First answer k finished streams share, in done_ms order.

    entries: (answer, done_ms, stream_index). Returns (answer, reached, done_ms).
    When no answer reaches k, falls back to the plurality over all entries and
    marks reached=False with done_ms set to the last stream's done_ms. The
    fallback tie-break uses the original stream index, not the done_ms rank.
    """
    ordered = sorted(entries, key=lambda e: (e[1], e[2]))
    counts = {}
    last = ordered[-1][1] if ordered else 0.0
    for a, done_ms, _ in ordered:
        if a is None:
            continue
        key = M.canon(a)
        counts[key] = counts.get(key, 0) + 1
        if counts[key] == k:
            return a, True, done_ms
    items = [(a, "cot", idx) for a, _, idx in ordered]
    return M.plurality(items)["answer"], False, last


def group_first_k(by_id, ids, mode):
    fk = {}
    for i in ids:
        entries = [(answer_of(by_id[i][s]), by_id[i][s]["done_ms"], s)
                   for s in group_indices(by_id[i], mode)]
        fk[i] = first_k_agree(entries)
    return fk


def first_k_stats(fk, gold, ids):
    ok = {i: fk[i][0] is not None and M.canon(fk[i][0]) == M.canon(gold[i]) for i in ids}
    reached_ids = [i for i in ids if fk[i][1]]
    correct = sum(ok.values())
    correct_reached = sum(1 for i in reached_ids if ok[i])
    fk_ms = [fk[i][2] for i in ids]
    return {
        "ok": ok,
        "correct": correct,
        "acc": M.pct(correct, len(ids)),
        "reached": len(reached_ids),
        "correct_reached": correct_reached,
        "acc_reached": M.pct(correct_reached, len(reached_ids)),
        "median_fk_ms": statistics.median(fk_ms) if fk_ms else float("nan"),
    }


def generated_stats(by_id, ids, mode):
    gen = [by_id[i][s]["generated"] for i in ids for s in group_indices(by_id[i], mode)]
    return {
        "n": len(gen),
        "mean": statistics.fmean(gen) if gen else float("nan"),
        "max": max(gen) if gen else 0,
        "min": min(gen) if gen else 0,
    }


def wrapper_marker_leak(by_id, ids):
    """Generated wrapper text would mean the wrapper was not prompt-only."""
    total = leaked = 0
    for i in ids:
        for s in group_indices(by_id[i], WRAP):
            total += 1
            text = by_id[i][s]["text"].lstrip()
            if text.startswith("Question:") or text.startswith("Solution:"):
                leaked += 1
    return leaked, total


def parity_layout(by_id, ids):
    """Check that problem id parity decides which half carries which mode."""
    patterns = {}
    match = 0
    for i in ids:
        h0 = tuple(sorted({by_id[i][s]["mode"] for s in range(HALF)}))
        h1 = tuple(sorted({by_id[i][s]["mode"] for s in range(HALF, N_STREAMS)}))
        pat = (h0, h1)
        patterns[pat] = patterns.get(pat, 0) + 1
        expected = ((PLAIN,), (WRAP,)) if i % 2 == 0 else ((WRAP,), (PLAIN,))
        if pat == expected:
            match += 1
    return {
        "match": match,
        "n": len(ids),
        "patterns": patterns,
        "alternating": match == len(ids),
    }


# ---------------------------------------------------------------- statistics ----


def paired_test(wrap_pl, plain_pl, ids, gold):
    """wrapper minus plain; positive delta means the wrapper arm is better."""
    wv = [1 if wrap_pl[i]["answer"] is not None
          and M.canon(wrap_pl[i]["answer"]) == M.canon(gold[i]) else 0 for i in ids]
    pv = [1 if plain_pl[i]["answer"] is not None
          and M.canon(plain_pl[i]["answer"]) == M.canon(gold[i]) else 0 for i in ids]
    b = sum(1 for x, y in zip(wv, pv) if x and not y)
    c = sum(1 for x, y in zip(wv, pv) if not x and y)
    lo_tie, hi_tie = M.tie_aware_paired_bootstrap(
        wrap_pl, plain_pl, ids, gold, seed=TIE_BOOTSTRAP_SEED, reps=TIE_BOOTSTRAP_N)
    return {
        "b": b, "c": c, "p": M.mcnemar_exact(b, c),
        "delta": M.pct(sum(wv), len(ids)) - M.pct(sum(pv), len(ids)),
        "lo_tie": lo_tie, "hi_tie": hi_tie,
    }


# ---------------------------------------------------------------- analysis ----


def analyze_run(label, path, provenance_path, gold_raw):
    by_id = load_run(path)
    sidecar, sidecar_path = M.load_sidecar(path)
    prov, prov_path = load_provenance(provenance_path)

    ids = sorted(by_id)
    for i in ids:
        if i not in gold_raw:
            raise SystemExit(f"{label}: no gold for id={i}")
    gold = {i: M.normalize_number(gold_raw[i]) for i in ids}

    pl = {mode: group_plurality(by_id, ids, mode) for mode in GROUP_ORDER}
    met = {mode: M.tie_metrics(pl[mode], ids, gold) for mode in GROUP_ORDER}
    oracle, support = {}, {}
    for mode in GROUP_ORDER:
        oracle[mode], support[mode] = group_oracle(by_id, ids, mode, gold)
    slot_acc = {mode: group_slot_accuracy(by_id, ids, mode, gold) for mode in GROUP_ORDER}
    wrong_agree = {mode: wrong_answer_agreement(by_id, ids, mode, gold)
                   for mode in GROUP_ORDER}
    fk = {mode: group_first_k(by_id, ids, mode) for mode in GROUP_ORDER}
    fks = {mode: first_k_stats(fk[mode], gold, ids) for mode in GROUP_ORDER}
    gen = {mode: generated_stats(by_id, ids, mode) for mode in GROUP_ORDER}

    gold_absent_streams = {mode: sum(1 for i in ids if support[mode][i] == 0)
                           for mode in GROUP_ORDER}
    gold_absent_top = {
        mode: sum(1 for i in ids
                  if not any(M.canon(x) == M.canon(gold[i]) for x in pl[mode][i]["top"]))
        for mode in GROUP_ORDER
    }

    test = paired_test(pl[WRAP], pl[PLAIN], ids, gold)

    if test["delta"] >= 3.0 and test["lo_tie"] > 0:
        verdict = "PASS"
    elif test["delta"] <= 0.0:
        verdict = "DEAD"
    else:
        verdict = "PARTIAL"

    layouts = parity_layout(by_id, ids)
    leak = wrapper_marker_leak(by_id, ids)

    return {
        "label": label, "path": path, "ids": ids, "gold": gold, "by_id": by_id,
        "sidecar": sidecar, "sidecar_path": sidecar_path,
        "provenance": prov, "provenance_path": prov_path,
        "pl": pl, "met": met, "oracle": oracle, "support": support,
        "slot_acc": slot_acc, "wrong_agree": wrong_agree,
        "fk": fk, "fks": fks, "gen": gen,
        "gold_absent_streams": gold_absent_streams,
        "gold_absent_top": gold_absent_top,
        "test": test, "verdict": verdict,
        "layouts": layouts, "leak": leak,
    }


# ------------------------------------------------------------------ output ----


def write_csv(r, path):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        header = ["id", "gold", "stream0_mode", "stream8_mode", "parity_expected"]
        for mode in GROUP_ORDER:
            header += [f"{mode}_answer", f"{mode}_ok", f"{mode}_tied", f"{mode}_top",
                       f"{mode}_oracle", f"{mode}_gold_support",
                       f"{mode}_fk_answer", f"{mode}_fk_ok", f"{mode}_fk_reached",
                       f"{mode}_fk_ms", f"{mode}_generated", f"{mode}_answers"]
        w.writerow(header)
        for i in r["ids"]:
            g = M.canon(r["gold"][i])
            row = [i, r["gold"][i], r["by_id"][i][0]["mode"], r["by_id"][i][HALF]["mode"],
                   (PLAIN if i % 2 == 0 else WRAP)]
            for mode in GROUP_ORDER:
                p = r["pl"][mode][i]
                f = r["fk"][mode][i]
                row += [
                    p["answer"] if p["answer"] is not None else "",
                    int(p["answer"] is not None and M.canon(p["answer"]) == g),
                    int(p["tied"]),
                    "|".join(str(x) for x in p["top"]),
                    int(r["oracle"][mode][i]),
                    r["support"][mode][i],
                    f[0] if f[0] is not None else "",
                    int(r["fks"][mode]["ok"][i]),
                    int(f[1]),
                    round(f[2], 3),
                    "|".join(str(r["by_id"][i][s]["generated"])
                             for s in group_indices(r["by_id"][i], mode)),
                    "|".join((answer_of(r["by_id"][i][s]) or "")
                             for s in group_indices(r["by_id"][i], mode)),
                ]
            w.writerow(row)


def fmt_hist(values):
    h = {}
    for v in values:
        h[v] = h.get(v, 0) + 1
    return ", ".join(f"{k}:{v}" for k, v in sorted(h.items()))


def format_run(r):
    L = []
    ids = r["ids"]
    n = len(ids)
    sc = r["sidecar"] or {}
    prov = r.get("provenance") or {}
    L.append(f"=== {r['label']} (n={n}) ===")
    L.append(f"  run: {r['path']}")
    L.append(f"  sidecar: {r['sidecar_path']}")
    L.append(f"  n_streams={sc.get('n_streams')} max_trace={sc.get('max_trace')} "
             f"wrapper={sc.get('wrapper')} swap_parity={sc.get('swap_parity')} "
             f"n_ctx={sc.get('n_ctx')}")
    if sc.get("seeds"):
        L.append(f"  seeds: {sc.get('seeds')}")

    L.append("")
    L.append("provenance:")
    if prov:
        for k in ("model_file", "model_sha256", "source_file", "source_sha256",
                  "exe_file", "exe_sha256", "data_file", "data_sha256",
                  "max_trace", "seed", "seeds", "limit", "generated"):
            if k in prov:
                L.append(f"  {k:>14}: {prov[k]}")
        L.append(f"  {'file':>14}: {r['provenance_path']}")
    else:
        L.append(f"  {'file':>14}: {r['provenance_path']} (not found)")

    L.append("")
    L.append("groups (by recorded mode; plain=cot, wrapper=cotw):")
    for mode in GROUP_ORDER:
        L.append(f"  {mode:>7}: {len(group_indices(r['by_id'][ids[0]], mode))} streams/problem")

    L.append("")
    L.append("parity check (even id -> plain in the first half, odd id -> wrapper first):")
    lay = r["layouts"]
    L.append(f"  alternating layout on {lay['match']}/{lay['n']} problems "
             f"({'YES' if lay['alternating'] else 'NO'})")
    for pat, k in sorted(lay["patterns"].items(), key=lambda kv: -kv[1]):
        L.append(f"  half0={list(pat[0])} half1={list(pat[1])}: {k}/{lay['n']}")

    L.append("")
    L.append("tie-aware plurality (point uses the deterministic tie-break):")
    L.append(f"  {'group':>8} {'correct':>8} {'point%':>7} {'tied':>5} {'rand%':>7} {'upper%':>7}")
    for mode in GROUP_ORDER:
        m = r["met"][mode]
        L.append(f"  {mode:>8} {m['correct']:>8} {m['point']:>7.1f} {m['tied']:>5} "
                 f"{m['rand']:>7.1f} {m['upper']:>7.1f}")

    L.append("")
    L.append("oracle@8 and gold absence (no stream answers gold / gold not in plurality top):")
    for mode in GROUP_ORDER:
        o = sum(r["oracle"][mode].values())
        L.append(f"  {mode:>8}: oracle {o}/{n} ({M.pct(o, n):.1f}%), "
                 f"gold absent {r['gold_absent_streams'][mode]}/{n} "
                 f"({M.pct(r['gold_absent_streams'][mode], n):.1f}%), "
                 f"absent from plurality top {r['gold_absent_top'][mode]}")

    L.append("")
    L.append("per-stream accuracy by group slot (slot k shares seed S_k across halves):")
    seeds = (sc.get("seeds") or [])
    L.append(f"  {'group':>8} {'slot':>5} {'seed':>6} {'correct':>8} {'acc%':>7} {'answered':>9}")
    for mode in GROUP_ORDER:
        for slot in sorted(r["slot_acc"][mode]):
            s = r["slot_acc"][mode][slot]
            seed = seeds[slot] if slot < len(seeds) else slot
            L.append(f"  {mode:>8} {slot:>5} {seed:>6} {s['correct']:>8} "
                     f"{M.pct(s['correct'], s['total']):>7.1f} {s['answered']:>9}")

    L.append("")
    L.append("within-group wrong-answer agreement (both-answered pairs among wrong answers):")
    L.append(f"  {'group':>8} {'agree/both':>12} {'pair%':>7} {'probs>=2wrong':>14} "
             f"{'all pairs%':>11}")
    for mode in GROUP_ORDER:
        a = r["wrong_agree"][mode]
        L.append(f"  {mode:>8} {a['pair_agree']:>6}/{a['pair_both']:<5} "
                 f"{a['pair_frac']:>7.1f} {a['problems_with_two_wrong']:>14} "
                 f"{a['all_pair_frac']:>11.1f}")

    L.append("")
    L.append(f"first-K-agree (K={K_AGREE}, group streams in done_ms order):")
    L.append(f"  {'group':>8} {'correct':>8} {'acc%':>7} {'reached':>8} {'reach%':>7} "
             f"{'reach-only':>11} {'median fk ms':>13}")
    for mode in GROUP_ORDER:
        s = r["fks"][mode]
        L.append(f"  {mode:>8} {s['correct']:>8} {s['acc']:>7.1f} {s['reached']:>8} "
                 f"{M.pct(s['reached'], n):>7.1f} "
                 f"{s['correct_reached']:>5}/{s['reached']:<5} {s['median_fk_ms']:>13.1f}")

    L.append("")
    L.append("generated tokens per group (prompt-only wrapper check):")
    for mode in GROUP_ORDER:
        g = r["gen"][mode]
        L.append(f"  {mode:>8}: n={g['n']} mean {g['mean']:.1f}, min {g['min']}, max {g['max']}")
    leaked, total = r["leak"]
    L.append(f"  wrapped streams whose generated text starts with a wrapper marker: "
             f"{leaked}/{total}")

    t = r["test"]
    L.append("")
    L.append("paired test, wrapper minus plain (primary = tie-aware paired bootstrap, "
             f"{TIE_BOOTSTRAP_N} resamples, seed {TIE_BOOTSTRAP_SEED}, 95% CI):")
    L.append(f"  plurality point: delta {t['delta']:+.1f} pts, "
             f"tieCI [{t['lo_tie']:+.1f}, {t['hi_tie']:+.1f}], "
             f"b={t['b']} c={t['c']} p={t['p']:.4f}")

    L.append("")
    if r["verdict"] == "PASS":
        L.append(f"verdict: PASS (wrapper >= plain + 3 pts and tieCI excludes 0; "
                 f"delta {t['delta']:+.1f}, tieCI [{t['lo_tie']:+.1f}, {t['hi_tie']:+.1f}])")
    elif r["verdict"] == "DEAD":
        L.append(f"verdict: DEAD (wrapper <= plain; delta {t['delta']:+.1f}, "
                 f"tieCI [{t['lo_tie']:+.1f}, {t['hi_tie']:+.1f}])")
    else:
        L.append(f"verdict: PARTIAL (delta {t['delta']:+.1f}, "
                 f"tieCI [{t['lo_tie']:+.1f}, {t['hi_tie']:+.1f}], "
                 f"McNemar p={t['p']:.4f})")
    L.append("")
    return "\n".join(L)


# --------------------------------------------------------------------- main ----


def parse_run_arg(s):
    parts = s.split(",")
    if len(parts) < 2 or len(parts) > 3 or not parts[0] or not parts[1]:
        raise SystemExit(f"--run expects LABEL,TSV[,PROVENANCE], got {s!r}")
    return parts[0], parts[1], (parts[2] if len(parts) == 3 else "")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", default=[],
                    help="LABEL,TSV[,PROVENANCE] (repeatable)")
    ap.add_argument("--gold-file", default=DEFAULT_GOLD)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    if not args.run:
        ap.error("give at least one --run LABEL,TSV[,PROVENANCE]")

    gold_raw = M.load_gold(args.gold_file)
    out_lines = [
        "RecycleLM m6: within-run prompt-format lever (plain cot vs wrapped cotw)",
        f"gold: {os.path.abspath(args.gold_file)}",
        "source: harness/engine/modality.cpp (one 16-stream batch per problem)",
        "streams: 8 plain (mode=cot) + 8 wrapped (mode=cotw), seeds S0..S7 per half",
        "layout: --swap-parity puts wrapped first on odd problem ids",
        "",
    ]

    for spec in args.run:
        label, path, prov = parse_run_arg(spec)
        r = analyze_run(label, path, prov, gold_raw)
        csv_path = os.path.join(RESULTS, f"format_within_{label}.csv")
        write_csv(r, csv_path)
        out_lines.append(format_run(r))
        out_lines.append(f"[written] {csv_path}")
        out_lines.append("")

    text = "\n".join(out_lines).rstrip() + "\n"
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(text)
    sys.stdout.write(text)
    sys.stdout.write(f"\n[written] {args.out}\n")


if __name__ == "__main__":
    main()
