r"""Confound-free within-run cross-substrate ensembling (m5).

Reads ONE 24-stream run per model produced by harness/engine/modality.cpp:

    streams 0-7   cot   seeds S0..S7
    streams 8-15  cot   seeds S8..S15
    streams 16-23 code  seeds S8..S15

Because every arm lives in the same batched decode for a given problem, the
batch shape is identical across the compared arms; only the composition of the
second half changes between C16 and M16.

Arms:
    C8  = plurality over cot streams 0-7
    C16 = plurality over cot streams 0-15
    M16 = plurality over cot streams 0-7 + code streams 16-23

The analyzer mirrors harness/analyze_modality_vote.py: it reuses extract_cot,
the fenced-block extractor, and the isolated code executor (last number on
stdout, python-tagged fence, `python -I -S`, 8 s timeout; failures are not
votes). It reports tie-aware plurality, oracle coverage, per-substrate stream
accuracy, wrong-answer Jaccard, exact McNemar plus a tie-aware paired bootstrap
(2000 resamples, seed 1234), and first-K-agree (K=3) by done_ms.

Usage:
    python harness/analyze_mixed_within.py \
      --run m5_coder15b,results/m5_coder15b_within.tsv,results/m5_coder15b_provenance.txt \
      --run m5_1b,results/m5_1b_within.tsv,results/m5_1b_provenance.txt \
      --out results/m5_within_summary.txt
"""
import argparse
import csv
import os
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import analyze_modality_vote as M  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
DEFAULT_GOLD = os.path.join(ROOT, "data", "gsm8k_dev200.tsv")
DEFAULT_OUT = os.path.join(RESULTS, "m5_within_summary.txt")

COT_A = tuple(range(0, 8))
COT_B = tuple(range(8, 16))
COT16 = tuple(range(0, 16))
CODE = tuple(range(16, 24))
N_STREAMS = 24
K_AGREE = 3
TIE_BOOTSTRAP_N = 2000
TIE_BOOTSTRAP_SEED = 1234


# ---------------------------------------------------------------- loading ----


def load_run(path):
    rows, by_id = M.load_stream_tsv(path)
    for i, streams in by_id.items():
        got = set(streams)
        if got != set(range(N_STREAMS)):
            raise SystemExit(
                f"{path}: id={i} has streams {sorted(got)}; expected 0..{N_STREAMS - 1}")
        for s in COT16:
            if streams[s]["mode"] != "cot":
                raise SystemExit(f"{path}: id={i} stream={s} mode={streams[s]['mode']!r}, expected cot")
        for s in CODE:
            if streams[s]["mode"] != "code":
                raise SystemExit(f"{path}: id={i} stream={s} mode={streams[s]['mode']!r}, expected code")
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


# ------------------------------------------------------------ arm helpers ----


def answer_items(answers, streams):
    """(answer, substrate, stream_index) items in stream order."""
    return [(answers[s], "cot" if s in COT16 else "code", s) for s in streams]


def build_arm_votes(answers):
    """Per-problem plurality for C8 / C16 / M16 from a stream->answer map."""
    return {
        "C8": M.plurality(answer_items(answers, COT_A)),
        "C16": M.plurality(answer_items(answers, COT16)),
        "M16": M.plurality(answer_items(answers, COT_A + CODE)),
    }


def first_k_agree(entries, k=K_AGREE):
    """First answer k finished streams share, in done_ms order.

    entries: iterable of (answer, done_ms, stream_index). Returns
    (answer, reached, done_ms); when no answer reaches k, falls back to the
    plurality over all entries and marks reached=False. The fallback tie-break
    uses the original stream index, not the done_ms rank.
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


def jaccard(a, b):
    union = a | b
    return len(a & b) / len(union) if union else float("nan")


def stream_accuracy(ans_by_stream, streams, ids, gold):
    total = answered = correct = 0
    for i in ids:
        g = M.canon(gold[i])
        for s in streams:
            a = ans_by_stream[i][s]
            total += 1
            if a is not None:
                answered += 1
                correct += 1 if M.canon(a) == g else 0
    return {"total": total, "answered": answered, "correct": correct,
            "acc": M.pct(correct, total)}


def status_counts(status, streams, ids):
    c = {"ok": 0, "empty": 0, "error": 0, "timeout": 0, "nocode": 0}
    for i in ids:
        for s in streams:
            c[status[i][s]] = c.get(status[i][s], 0) + 1
    return c


def compliant_counts(okmap, streams, ids):
    yes = tot = 0
    for i in ids:
        for s in streams:
            tot += 1
            yes += 1 if okmap[i][s] else 0
    return {"compliant": yes, "total": tot}


def paired_test(arm_a, arm_b, ids, gold):
    """arm_a vs arm_b; positive delta means arm_a is better."""
    av = [1 if arm_a[i]["answer"] is not None and M.canon(arm_a[i]["answer"]) == M.canon(gold[i]) else 0
          for i in ids]
    bv = [1 if arm_b[i]["answer"] is not None and M.canon(arm_b[i]["answer"]) == M.canon(gold[i]) else 0
          for i in ids]
    b = sum(1 for x, y in zip(av, bv) if x and not y)
    c = sum(1 for x, y in zip(av, bv) if not x and y)
    lo_tie, hi_tie = M.tie_aware_paired_bootstrap(
        arm_a, arm_b, ids, gold, seed=TIE_BOOTSTRAP_SEED, reps=TIE_BOOTSTRAP_N)
    return {
        "delta": M.pct(sum(av), len(ids)) - M.pct(sum(bv), len(ids)),
        "mcnemar_b": b, "mcnemar_c": c, "p": M.mcnemar_exact(b, c),
        "lo_tie": lo_tie, "hi_tie": hi_tie,
    }


# ---------------------------------------------------------------- analysis ----


def analyze_run(label, path, provenance_path, gold_raw, exec_timeout):
    rows, by_id = load_run(path)
    sidecar, sidecar_path = M.load_sidecar(path)
    prov, prov_path = load_provenance(provenance_path)

    ids = sorted(by_id)
    for i in ids:
        if i not in gold_raw:
            raise SystemExit(f"{label}: no gold for id={i}")
    gold = {i: M.normalize_number(gold_raw[i]) for i in ids}

    cot_ans = {i: {} for i in ids}
    cot_paths = {i: {} for i in ids}
    for i in ids:
        for s in COT16:
            v, p = M.extract_cot(by_id[i][s]["text"])
            cot_ans[i][s] = v
            cot_paths[i][s] = p

    code_src = {i: {s: M.extract_code_block(by_id[i][s]["text"]) for s in CODE} for i in ids}
    unique = sorted({c for i in ids for c in code_src[i].values() if c})
    cache = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for src, res in zip(unique, pool.map(lambda s: M.run_code(s, exec_timeout), unique)):
            cache[src] = res

    code_ans = {i: {} for i in ids}
    code_status = {i: {} for i in ids}
    code_ok = {i: {} for i in ids}
    for i in ids:
        for s in CODE:
            src = code_src[i][s]
            if not src:
                a, st, comp = None, "nocode", False
            else:
                a, st, comp = cache[src]
            code_ans[i][s] = a
            code_status[i][s] = st
            code_ok[i][s] = comp

    ans_by_stream = {i: {**cot_ans[i], **code_ans[i]} for i in ids}

    arms = {}
    for i in ids:
        votes = build_arm_votes(ans_by_stream[i])
        for name, res in votes.items():
            arms.setdefault(name, {})[i] = res

    arm_metrics = {name: M.tie_metrics(arms[name], ids, gold) for name in ("C8", "C16", "M16")}

    oracle = {"cot16": {}, "code8": {}, "union": {}}
    for i in ids:
        g = M.canon(gold[i])
        oc = any(cot_ans[i][s] is not None and M.canon(cot_ans[i][s]) == g for s in COT16)
        op = any(code_ans[i][s] is not None and M.canon(code_ans[i][s]) == g for s in CODE)
        oracle["cot16"][i] = oc
        oracle["code8"][i] = op
        oracle["union"][i] = oc or op

    stream_stats = {
        "cot_0_7": stream_accuracy(ans_by_stream, COT_A, ids, gold),
        "cot_8_15": stream_accuracy(ans_by_stream, COT_B, ids, gold),
        "cot_0_15": stream_accuracy(ans_by_stream, COT16, ids, gold),
        "code_16_23": stream_accuracy(ans_by_stream, CODE, ids, gold),
    }

    codcov = status_counts(code_status, CODE, ids)
    codcomp = compliant_counts(code_ok, CODE, ids)

    n_both = n_wrong_union = 0
    jaccards = []
    for i in ids:
        g = M.canon(gold[i])
        cot_has = any(cot_ans[i][s] is not None for s in COT16)
        code_has = any(code_ans[i][s] is not None for s in CODE)
        if not (cot_has and code_has):
            continue
        n_both += 1
        cw = {M.canon(cot_ans[i][s]) for s in COT16
              if cot_ans[i][s] is not None and M.canon(cot_ans[i][s]) != g}
        pw = {M.canon(code_ans[i][s]) for s in CODE
              if code_ans[i][s] is not None and M.canon(code_ans[i][s]) != g}
        if cw or pw:
            n_wrong_union += 1
            jaccards.append(jaccard(cw, pw))
    mean_jaccard = statistics.fmean(jaccards) if jaccards else float("nan")

    fk = {}
    fk["C16"] = {i: first_k_agree(
        [(ans_by_stream[i][s], by_id[i][s]["done_ms"], s) for s in COT16]) for i in ids}
    fk["M16"] = {i: first_k_agree(
        [(ans_by_stream[i][s], by_id[i][s]["done_ms"], s) for s in COT_A + CODE]) for i in ids}
    fks = {name: first_k_stats(fk[name], gold, ids) for name in ("C16", "M16")}

    tests = {
        "M16_vs_C16": paired_test(arms["M16"], arms["C16"], ids, gold),
        "M16_vs_C8": paired_test(arms["M16"], arms["C8"], ids, gold),
    }

    step_p50 = M.percentile((sidecar or {}).get("step_ms", []), 0.50)
    step_p95 = M.percentile((sidecar or {}).get("step_ms", []), 0.95)

    return {
        "label": label, "path": path, "ids": ids, "gold": gold,
        "by_id": by_id, "sidecar": sidecar, "sidecar_path": sidecar_path,
        "provenance": prov, "provenance_path": prov_path,
        "arms": arms, "arm_metrics": arm_metrics, "oracle": oracle,
        "stream_stats": stream_stats, "code_cov": codcov, "code_compliant": codcomp,
        "n_both": n_both, "n_wrong_union": n_wrong_union,
        "mean_jaccard": mean_jaccard,
        "fk": fk, "fks": fks, "tests": tests,
        "cot_ans": cot_ans, "code_ans": code_ans, "code_status": code_status,
        "step_p50": step_p50, "step_p95": step_p95,
        "cot_paths": cot_paths,
    }


# ------------------------------------------------------------------ output ----


def write_csv(r, path):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        header = ["id", "gold"]
        for a in ("C8", "C16", "M16"):
            header += [f"{a}_answer", f"{a}_ok", f"{a}_tied", f"{a}_top"]
        header += ["oracle_cot16", "oracle_code8", "oracle_union",
                   "cot_answers_0_15", "code_answers_16_23", "code_status_16_23",
                   "C16_fk_answer", "C16_fk_ok", "C16_fk_reached", "C16_fk_ms",
                   "M16_fk_answer", "M16_fk_ok", "M16_fk_reached", "M16_fk_ms"]
        w.writerow(header)
        for i in r["ids"]:
            g = M.canon(r["gold"][i])
            row = [i, r["gold"][i]]
            for a in ("C8", "C16", "M16"):
                v = r["arms"][a][i]
                row += [v["answer"] if v["answer"] is not None else "",
                        int(v["answer"] is not None and M.canon(v["answer"]) == g),
                        int(v["tied"]), "|".join(str(x) for x in v["top"])]
            for o in ("cot16", "code8", "union"):
                row.append(int(r["oracle"][o][i]))
            row.append("|".join(r["cot_ans"][i][s] if r["cot_ans"][i][s] is not None else ""
                                for s in COT16))
            row.append("|".join(r["code_ans"][i][s] if r["code_ans"][i][s] is not None else ""
                                for s in CODE))
            row.append("|".join(r["code_status"][i][s] for s in CODE))
            for name in ("C16", "M16"):
                f = r["fk"][name][i]
                row += [f[0] if f[0] is not None else "", int(r["fks"][name]["ok"][i]),
                        int(f[1]), round(f[2], 3)]
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
    L.append(f"=== {r['label']} (n={n}) ===")
    L.append(f"  run: {r['path']}")
    sc = r["sidecar"] or {}
    L.append(f"  sidecar: {r['sidecar_path']}")
    L.append(f"  n_streams={sc.get('n_streams')} max_trace={sc.get('max_trace')} "
             f"seed={sc.get('seed')} seeds={sc.get('seeds')} n_ctx={sc.get('n_ctx')}")
    if sc.get("step_ms_fraction") is not None:
        L.append(f"  step_ms recorded {sc.get('step_ms_recorded')}/{sc.get('step_ms_total')} "
                 f"(fraction {sc.get('step_ms_fraction'):.4f})")

    L.append("")
    L.append("provenance:")
    pv = r.get("provenance") or {}
    if pv:
        for k in ("model_file", "model_sha256", "source_file", "source_sha256",
                  "exe_file", "exe_sha256", "data_file", "data_sha256",
                  "max_trace", "seed", "seeds", "limit", "generated"):
            if k in pv:
                L.append(f"  {k:>14}: {pv[k]}")
        L.append(f"  {'file':>14}: {r['provenance_path']}")
    else:
        L.append(f"  {'file':>14}: {r['provenance_path']} (not found)")

    L.append("")
    L.append("arms (tie-aware; point uses the deterministic tie-break):")
    L.append(f"  {'arm':>4} {'correct':>8} {'point%':>7} {'tied':>5} {'rand%':>7} {'upper%':>7}")
    for a in ("C8", "C16", "M16"):
        m = r["arm_metrics"][a]
        L.append(f"  {a:>4} {m['correct']:>8} {m['point']:>7.1f} {m['tied']:>5} "
                 f"{m['rand']:>7.1f} {m['upper']:>7.1f}")

    L.append("")
    L.append("oracle coverage (any stream correct):")
    for o, lbl in (("cot16", "cot 0-15"), ("code8", "code 16-23"), ("union", "union")):
        k = sum(r["oracle"][o].values())
        L.append(f"  {lbl:>11}: {k}/{n} ({M.pct(k, n):.1f}%)")

    L.append("")
    L.append("per-substrate stream accuracy:")
    for tag, s in r["stream_stats"].items():
        L.append(f"  {tag:>10}: answered {s['answered']}/{s['total']} "
                 f"({M.pct(s['answered'], s['total']):.1f}%), "
                 f"correct {s['correct']}/{s['total']} ({s['acc']:.1f}%)")

    L.append("")
    L.append("code exec coverage (streams 16-23):")
    c = r["code_cov"]
    tot = sum(c.values())
    L.append(f"  ok {c['ok']} ({M.pct(c['ok'], tot):.1f}%), empty {c['empty']}, "
             f"error {c['error']}, timeout {c['timeout']}, nocode {c['nocode']}")
    k = r["code_compliant"]
    L.append(f"  compliant (single-number output): {k['compliant']}/{k['total']} "
             f"({M.pct(k['compliant'], k['total']):.1f}%)")

    L.append("")
    L.append(f"wrong-answer Jaccard (both substrates answered): {r['mean_jaccard']:.3f} "
             f"over {r['n_wrong_union']} problems with a non-empty wrong union "
             f"({r['n_both']}/{n} both answered)")

    L.append("")
    L.append(f"first-K-agree (K={K_AGREE}, streams in done_ms order; 16-slot arms):")
    L.append(f"  {'arm':>4} {'correct':>8} {'acc%':>7} {'reached':>8} {'reach%':>7} "
             f"{'reach-only':>11} {'median fk ms':>13}")
    for a in ("C16", "M16"):
        s = r["fks"][a]
        L.append(f"  {a:>4} {s['correct']:>8} {s['acc']:>7.1f} {s['reached']:>8} "
                 f"{M.pct(s['reached'], n):>7.1f} "
                 f"{s['correct_reached']:>5}/{s['reached']:<5} {s['median_fk_ms']:>13.1f}")

    L.append("")
    L.append(f"paired tests (positive delta = first arm better; tie-aware paired "
             f"bootstrap {TIE_BOOTSTRAP_N} resamples, seed {TIE_BOOTSTRAP_SEED}, 95% CI):")
    for name, t in r["tests"].items():
        L.append(f"  {name:>10}: delta={t['delta']:+.1f} pts "
                 f"tieCI [{t['lo_tie']:+.1f}, {t['hi_tie']:+.1f}] "
                 f"b={t['mcnemar_b']} c={t['mcnemar_c']} p={t['p']:.4f}")

    L.append("")
    L.append("latency / batch width:")
    L.append(f"  step p50={r['step_p50']:.2f} ms, p95={r['step_p95']:.2f} ms "
             f"at width 24 (streams in the batch)")
    if sc.get("step_ms_total"):
        L.append(f"  recorded prefix {sc.get('step_ms_recorded')}/{sc.get('step_ms_total')} steps")

    L.append("")
    L.append("verdict (positive sign requires delta > 0 and a tie-aware 95% CI excluding 0):")
    for name in ("M16_vs_C16", "M16_vs_C8"):
        t = r["tests"][name]
        ok = t["delta"] > 0 and t["lo_tie"] > 0
        L.append(f"  {name:>10}: {'YES' if ok else 'NO'} "
                 f"(delta {t['delta']:+.1f}, tieCI [{t['lo_tie']:+.1f}, {t['hi_tie']:+.1f}], "
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
    ap.add_argument("--exec-timeout", type=float, default=M.EXEC_TIMEOUT)
    args = ap.parse_args()

    if not args.run:
        ap.error("give at least one --run LABEL,TSV[,PROVENANCE]")

    gold_raw = M.load_gold(args.gold_file)
    out_lines = [
        "RecycleLM m5: confound-free within-run cross-substrate ensembling",
        f"gold: {os.path.abspath(args.gold_file)}",
        "source: harness/engine/modality.cpp (one 24-stream batch per problem)",
        "streams: 0-7 cot (S0-S7), 8-15 cot (S8-S15), 16-23 code (S8-S15)",
        "arms: C8=cot0-7, C16=cot0-15, M16=cot0-7+code16-23",
        "",
    ]

    for spec in args.run:
        label, path, prov = parse_run_arg(spec)
        r = analyze_run(label, path, prov, gold_raw, args.exec_timeout)
        csv_path = os.path.join(RESULTS, f"m5_within_{label}.csv")
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
