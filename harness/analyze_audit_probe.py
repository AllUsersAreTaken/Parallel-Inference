r"""Audit / verifier-as-selector probe over existing sampled traces (zero-GPU).

Question: over the CoT sample pools we already have, is there enough headroom
for an "auditor/verifier as selector" to beat plain plurality voting? If a
perfect selector of the top-2 (or top-3) distinct answers barely beats C8, a GPU
experiment that trains/runs such a selector is not worth it.

Method (per problem, per source):
  * extract a CoT answer per stream with the PROVEN extractor from
    harness/analyze_modality_vote.py (balanced \boxed{} -> "the answer is" line
    -> last number); we import it, we do not re-derive a weaker parser.
  * rank the distinct answers by support (count), tie-broken by summed stream
    index then canonical key; C8 = cot streams 0-7, C16 = all 16 cot streams.
  * tie-aware C8 (point / random expectation / upper bound, tie count).
  * oracle_select_top2 / top3 / all (perfect selector coverage), lift = @2 - C8.
  * runner-up correctness among C8-wrong problems (the key falsifier).
  * support gap histogram + near-tie fraction; gold support when C8 is wrong.
  * uniform random-selector-over-top-2 baseline.
  * stream-order dependence check (reverse the stream order, re-rank).
  * answer abstention reported separately, never counted as a coverage miss.

Usage:
    python harness/analyze_audit_probe.py
"""
import argparse
import csv
import math
import os
import sys
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from analyze_modality_vote import (  # noqa: E402  (proven extractor, reused)
    extract_cot,
    canon,
    normalize_number,
    load_gold,
    unescape_tsv,
)

ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
DEFAULT_GOLD = os.path.join(ROOT, "data", "gsm8k_dev200.tsv")
DEFAULT_OUT = os.path.join(RESULTS, "audit_probe_summary.txt")

HEADER = ["id", "stream", "mode", "finish", "generated", "ttft_ms", "done_ms", "text"]

SOURCES = [
    dict(name="dev200_s16", path=os.path.join(RESULTS, "dev200_s16.tsv"),
         kind="legacy", note="provenance unconfirmed"),
    dict(name="m3_1b_cot16", path=os.path.join(RESULTS, "m3_1b_cot16.tsv"),
         kind="header", mode="cot",
         prov="results/m3_1b_provenance.txt -> models/Llama-3.2-1B-Instruct-Q4_K_M.gguf, seed 1234, limit 40"),
    dict(name="m3_1b_mixed", path=os.path.join(RESULTS, "m3_1b_mixed.tsv"),
         kind="header", mode="cot",
         prov="results/m3_1b_provenance.txt -> models/Llama-3.2-1B-Instruct-Q4_K_M.gguf, seed 1234, limit 40"),
    dict(name="m3_coder15b_cot16", path=os.path.join(RESULTS, "m3_coder15b_cot16.tsv"),
         kind="header", mode="cot",
         prov="results/m3_coder15b_provenance.txt -> models/Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf, seed 1234, limit 100"),
    dict(name="m3_coder15b_mixed", path=os.path.join(RESULTS, "m3_coder15b_mixed.tsv"),
         kind="header", mode="cot",
         prov="results/m3_coder15b_provenance.txt -> models/Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf, seed 1234, limit 100"),
]


# ------------------------------------------------------------------- loading -


def load_legacy(path):
    """dev200_s16.tsv: id, stream, start_pos, end_pos, text (no header, 5 fields)."""
    by_id = OrderedDict()
    with open(path, encoding="utf-8", errors="replace") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.rstrip("\r\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 5:
                raise SystemExit(f"{path}:{lineno}: expected 5 fields, got {len(parts)}")
            pid, stream, text = int(parts[0]), int(parts[1]), parts[4]
            by_id.setdefault(pid, {})[stream] = text
    return by_id


def load_header(path, mode="cot"):
    """m3_*.tsv: id, stream, mode, finish, generated, ttft_ms, done_ms, text."""
    by_id = OrderedDict()
    with open(path, encoding="utf-8", errors="replace") as fh:
        header = fh.readline().rstrip("\r\n").split("\t")
        if header != HEADER:
            raise SystemExit(f"{path}: unexpected header {header!r}")
        for lineno, line in enumerate(fh, 2):
            line = line.rstrip("\r\n")
            if not line:
                continue
            parts = line.split("\t", 7)
            if len(parts) != 8:
                raise SystemExit(f"{path}:{lineno}: expected 8 fields, got {len(parts)}")
            if parts[2] != mode:
                continue
            by_id.setdefault(int(parts[0]), {})[int(parts[1])] = unescape_tsv(parts[7])
    return by_id


# ------------------------------------------------------------------- ranking -


def rank_groups(items):
    """items: ordered list of (canonical-or-None) answers. Returns (groups, ranked).

    Support ranking: count desc, then summed position asc, then canonical key.
    """
    groups = OrderedDict()
    for idx, a in enumerate(items):
        if a is None:
            continue
        k = canon(a)
        g = groups.setdefault(k, {"count": 0, "idx": 0})
        g["count"] += 1
        g["idx"] += idx
    ranked = sorted(groups.items(), key=lambda kv: (-kv[1]["count"], kv[1]["idx"], kv[0]))
    return groups, ranked


def arm_record(votes, gold):
    """All per-problem audit diagnostics for one arm (pool of votes)."""
    groups, ranked = rank_groups(votes)
    _, rev_ranked = rank_groups(list(reversed(votes)))

    keys = [k for k, _ in ranked]
    counts = [g["count"] for _, g in ranked]
    n_distinct = len(keys)
    top1 = keys[0] if keys else None
    sup1 = counts[0] if counts else 0
    top2 = keys[1] if len(keys) > 1 else None
    sup2 = counts[1] if len(counts) > 1 else None
    top3 = keys[2] if len(keys) > 2 else None

    tied_top = sum(1 for c in counts if c == sup1) if counts else 0
    tied_second = (sum(1 for c in counts if c == sup2)
                   if len(counts) > 1 and sup2 is not None else 0)

    ok = top1 is not None and top1 == gold
    if tied_top > 1:
        rand_exp = sum(1 for k in keys[:tied_top] if k == gold) / tied_top
        upper = 1 if gold in set(keys[:tied_top]) else 0
    else:
        rand_exp = 1.0 if ok else 0.0
        upper = 1 if ok else 0

    oracle2 = 1 if gold in set(keys[:2]) else 0
    oracle3 = 1 if gold in set(keys[:3]) else 0
    oracleall = 1 if gold in groups else 0
    runner_up_correct = bool(top2 is not None and top2 == gold)
    gap = (sup1 - sup2) if sup2 is not None else None
    near_tie = bool(gap is not None and gap <= 1)
    gold_support = groups[gold]["count"] if gold in groups else 0
    gold_rank = (keys.index(gold) + 1) if gold in keys else None
    rand_sel2 = (sum(1 for k in keys[:2] if k == gold) / len(keys[:2])) if keys else 0.0

    second_unique = len(counts) > 1 and tied_second == 1
    second_tied = len(counts) > 1 and tied_second > 1
    # order-independent: is the gold answer sitting at the 2nd support tier but
    # below the top tier? (a rank-2 runner-up that is randomly broken by ties)
    gold_in_second_tier = 1 if (sup2 is not None and gold_support == sup2
                                and gold not in set(keys[:tied_top])) else 0

    rev_top1 = rev_ranked[0][0] if rev_ranked else None
    rev_top2 = rev_ranked[1][0] if len(rev_ranked) > 1 else None
    rev_set2 = {k for k, _ in rev_ranked[:2]}
    cur_set2 = set(keys[:2])

    return {
        "answers": [v for v in votes if v is not None],
        "keys": keys, "counts": counts, "n_distinct": n_distinct,
        "top1": top1, "sup1": sup1, "top2": top2, "sup2": sup2, "top3": top3,
        "ok": 1 if ok else 0, "rand_exp": rand_exp, "upper": upper,
        "tied_top": tied_top, "tied_second": tied_second,
        "oracle2": oracle2, "oracle3": oracle3, "oracleall": oracleall,
        "runner_up_correct": 1 if runner_up_correct else 0,
        "gap": gap, "near_tie": 1 if near_tie else 0,
        "gold_support": gold_support, "gold_rank": gold_rank,
        "rand_sel2": rand_sel2, "abstain": 1 if n_distinct == 0 else 0,
        "second_unique": 1 if second_unique else 0,
        "second_tied": 1 if second_tied else 0,
        "gold_in_second_tier": gold_in_second_tier,
        "order_dep_top1": 1 if tied_top > 1 else 0,
        "order_dep_top2": 1 if tied_second > 1 else 0,
        "rev_top1_diff": 1 if rev_top1 != top1 else 0,
        "rev_top2_diff": 1 if rev_top2 != top2 else 0,
        "rev_set2_diff": 1 if rev_set2 != cur_set2 else 0,
    }


# --------------------------------------------------------------- aggregation -


def hist_supports(values):
    """values: list of supports; buckets 0,1,2,3+."""
    b = {"0": 0, "1": 0, "2": 0, "3+": 0}
    for v in values:
        if v <= 0:
            b["0"] += 1
        elif v == 1:
            b["1"] += 1
        elif v == 2:
            b["2"] += 1
        else:
            b["3+"] += 1
    return b


def hist_gaps(gaps):
    """gaps: list of non-negative ints; buckets 0,1,2,3+."""
    return hist_supports(gaps)


def pct(num, den):
    return 100.0 * num / den if den else float("nan")


def aggregate(records, ids):
    n = len(ids)
    recs = [records[i] for i in ids]
    answered = [r for r in recs if not r["abstain"]]
    na = len(answered)
    c8_wrong = [r for r in answered if not r["ok"]]
    c8_wrong_with_ru = [r for r in c8_wrong if r["top2"] is not None]
    c8_wrong_unique2 = [r for r in c8_wrong if r["second_unique"]]
    c8_wrong_tied2 = [r for r in c8_wrong if r["second_tied"]]
    gaps = [r["gap"] for r in answered if r["gap"] is not None]

    return {
        "n": n,
        "abstain": sum(r["abstain"] for r in recs),
        "n_answered": na,
        "c8_correct_all": sum(r["ok"] for r in recs),
        "c8_point_all": pct(sum(r["ok"] for r in recs), n),
        "c8_point_ans": pct(sum(r["ok"] for r in answered), na),
        "c8_rand_all": pct(sum(r["rand_exp"] for r in recs), n),
        "c8_rand_ans": pct(sum(r["rand_exp"] for r in answered), na),
        "c8_upper_all": pct(sum(r["upper"] for r in recs), n),
        "c8_upper_ans": pct(sum(r["upper"] for r in answered), na),
        "tied_top": sum(r["tied_top"] > 1 for r in recs),
        "oracle2_ans": pct(sum(r["oracle2"] for r in answered), na),
        "oracle3_ans": pct(sum(r["oracle3"] for r in answered), na),
        "oracleall_ans": pct(sum(r["oracleall"] for r in answered), na),
        "oracle2_all": pct(sum(r["oracle2"] for r in recs), n),
        "oracle3_all": pct(sum(r["oracle3"] for r in recs), n),
        "oracleall_all": pct(sum(r["oracleall"] for r in recs), n),
        "lift2_ans": pct(sum(r["oracle2"] for r in answered), na) - pct(sum(r["ok"] for r in answered), na),
        "lift2_all": pct(sum(r["oracle2"] for r in recs), n) - pct(sum(r["ok"] for r in recs), n),
        "c8_wrong": len(c8_wrong),
        "c8_wrong_with_ru": len(c8_wrong_with_ru),
        "runner_up_correct": sum(r["runner_up_correct"] for r in c8_wrong),
        "runner_up_rate": pct(sum(r["runner_up_correct"] for r in c8_wrong), len(c8_wrong)),
        "runner_up_rate_avail": pct(sum(r["runner_up_correct"] for r in c8_wrong_with_ru), len(c8_wrong_with_ru)),
        "ru_unique_n": len(c8_wrong_unique2),
        "ru_unique_correct": sum(r["runner_up_correct"] for r in c8_wrong_unique2),
        "ru_unique_rate": pct(sum(r["runner_up_correct"] for r in c8_wrong_unique2), len(c8_wrong_unique2)),
        "ru_tied_n": len(c8_wrong_tied2),
        "ru_tied_correct": sum(r["runner_up_correct"] for r in c8_wrong_tied2),
        "ru_tied_rate": pct(sum(r["runner_up_correct"] for r in c8_wrong_tied2), len(c8_wrong_tied2)),
        "gold_second_tier": sum(r["gold_in_second_tier"] for r in c8_wrong),
        "gold_second_tier_rate": pct(sum(r["gold_in_second_tier"] for r in c8_wrong), len(c8_wrong)),
        "gap_hist": hist_gaps(gaps),
        "near_tie": sum(r["near_tie"] for r in answered if r["gap"] is not None),
        "near_tie_denom": len(gaps),
        "near_tie_frac": pct(sum(r["near_tie"] for r in answered if r["gap"] is not None), len(gaps)),
        "gold_support_wrong_hist": hist_supports([r["gold_support"] for r in c8_wrong]),
        "rand_sel2_ans": pct(sum(r["rand_sel2"] for r in answered), na),
        "order_dep_top1": sum(r["order_dep_top1"] for r in recs),
        "order_dep_top2": sum(r["order_dep_top2"] for r in recs),
        "rev_top1_diff": sum(r["rev_top1_diff"] for r in recs),
        "rev_top2_diff": sum(r["rev_top2_diff"] for r in recs),
        "rev_set2_diff": sum(r["rev_set2_diff"] for r in recs),
    }


# ------------------------------------------------------------------ fixtures -


def build_source(spec, gold_raw):
    if spec["kind"] == "legacy":
        by_id = load_legacy(spec["path"])
    else:
        by_id = load_header(spec["path"], spec.get("mode", "cot"))

    ids = sorted(by_id)
    missing = [i for i in ids if i not in gold_raw]
    if missing:
        raise SystemExit(f"{spec['name']}: no gold for ids {missing[:10]}")

    cot_streams = {i: sorted(by_id[i]) for i in ids}
    n_streams = sorted({len(v) for v in cot_streams.values()})

    extracted = {i: [extract_cot(by_id[i][s]) for s in cot_streams[i]] for i in ids}
    paths = {}
    for i in ids:
        for _, p in extracted[i]:
            paths[p] = paths.get(p, 0) + 1

    gold = {i: canon(normalize_number(gold_raw[i])) for i in ids}

    def votes_for(i, stream_ids):
        return [extracted[i][pos][0] for pos in range(len(cot_streams[i]))
                if cot_streams[i][pos] in stream_ids]

    c8 = {i: votes_for(i, set(cot_streams[i][:8])) for i in ids}
    has16 = all(len(cot_streams[i]) >= 16 for i in ids)
    c16 = {i: votes_for(i, set(cot_streams[i][:16])) for i in ids} if has16 else None

    return {
        "spec": spec, "ids": ids, "gold": gold, "paths": paths,
        "n_streams": n_streams, "has16": has16,
        "records_c8": {i: arm_record(c8[i], gold[i]) for i in ids},
        "records_c16": ({i: arm_record(c16[i], gold[i]) for i in ids} if has16 else None),
    }


# --------------------------------------------------------------------- gates -


def gate_verdict(a):
    """G0b / dead-end verdict for one arm's aggregate."""
    ora2 = a["oracle2_ans"]
    c8 = a["c8_point_ans"]
    ru = a["runner_up_rate"]
    lift = ora2 - c8
    if ora2 >= c8 + 10 and ru >= 30:
        return "PASS", lift
    if ora2 <= c8 + 3 or ru < 15:
        return "DEAD-END", lift
    return "BORDERLINE", lift


# --------------------------------------------------------------------- output -


def fmt_hist(h):
    return " ".join(f"{k}:{v}" for k, v in h.items())


def fmt_arm_line(arm, a):
    return (
        f"  {arm:>5} {a['n']:>4} {a['abstain']:>5} "
        f"{a['c8_point_all']:>9.1f} {a['c8_point_ans']:>9.1f} "
        f"{a['c8_rand_ans']:>7.1f} {a['c8_upper_ans']:>7.1f} {a['tied_top']:>4} "
        f"{a['oracle2_ans']:>6.1f} {a['oracle3_ans']:>6.1f} {a['oracleall_ans']:>6.1f} "
        f"{a['lift2_ans']:>+6.1f} "
        f"{a['runner_up_correct']:>6}/{a['c8_wrong']:<3} {a['runner_up_rate']:>6.1f} "
        f"{a['near_tie']:>4}/{a['near_tie_denom']:<3} {a['near_tie_frac']:>6.1f} "
        f"{a['rand_sel2_ans']:>8.1f}"
    )


def format_source(src):
    L = []
    spec = src["spec"]
    L.append(f"=== source {spec['name']} (n={len(src['ids'])}) ===")
    L.append(f"  path    : {os.path.relpath(spec['path'], ROOT)}")
    L.append(f"  layout  : {spec['kind']} loader; cot streams per problem = {src['n_streams']}")
    L.append(f"  note    : {spec.get('note', spec.get('prov', 'clean harness dump'))}")
    L.append(f"  gold    : {os.path.relpath(DEFAULT_GOLD, ROOT)}")
    L.append("  extraction path tally (all cot streams): " +
             ", ".join(f"{k}={v}" for k, v in sorted(src["paths"].items())))

    L.append("")
    L.append("  arm     n  abst  C8pt(all) C8pt(ans) C8rand C8up tied  ora@2  ora@3  ora@8   lift"
             "   run-up   ru%  near-tie  nt%  rndsel2")
    for arm, recs in (("C8", src["records_c8"]), ("C16", src["records_c16"])):
        if recs is None:
            L.append(f"  {arm:>5}   (n/a: fewer than 16 cot streams)")
            continue
        L.append(fmt_arm_line(arm, aggregate(recs, src["ids"])))

    a8 = aggregate(src["records_c8"], src["ids"])
    L.append("")
    L.append("  C8 detail (answered problems only for coverage/lift):")
    L.append(f"    answered {a8['n_answered']}/{a8['n']}, abstention {a8['abstain']}/{a8['n']} "
             f"({pct(a8['abstain'], a8['n']):.1f}%); abstentions are excluded from coverage, not counted wrong")
    L.append(f"    C8 point  all-n {a8['c8_point_all']:.1f}% | answered {a8['c8_point_ans']:.1f}%; "
             f"rand {a8['c8_rand_ans']:.1f}%; upper {a8['c8_upper_ans']:.1f}%; tied {a8['tied_top']}")
    L.append(f"    oracle @2 {a8['oracle2_ans']:.1f}% (all-n {a8['oracle2_all']:.1f}%), "
             f"@3 {a8['oracle3_ans']:.1f}%, @8 {a8['oracleall_ans']:.1f}%")
    L.append(f"    top2 lift over C8 = {a8['lift2_ans']:+.1f} pts (answered), {a8['lift2_all']:+.1f} pts (all-n)")
    L.append(f"    runner-up correct among C8-wrong: {a8['runner_up_correct']}/{a8['c8_wrong']} "
             f"({a8['runner_up_rate']:.1f}%); with a runner-up present {a8['c8_wrong_with_ru']} "
             f"({a8['runner_up_rate_avail']:.1f}%)")
    L.append(f"    runner-up tie sensitivity: unique 2nd place {a8['ru_unique_correct']}/{a8['ru_unique_n']} "
             f"({a8['ru_unique_rate']:.1f}%), tied 2nd place {a8['ru_tied_correct']}/{a8['ru_tied_n']} "
             f"({a8['ru_tied_rate']:.1f}%)")
    L.append(f"    gold at 2nd support tier among C8-wrong (order-independent): "
             f"{a8['gold_second_tier']}/{a8['c8_wrong']} ({a8['gold_second_tier_rate']:.1f}%)")
    L.append(f"    support gap (plurality - runner-up) histogram: {fmt_hist(a8['gap_hist'])}; "
             f"near-tie (gap<=1) {a8['near_tie']}/{a8['near_tie_denom']} ({a8['near_tie_frac']:.1f}%)")
    L.append(f"    gold support when C8 wrong (0/1/2/3+): {fmt_hist(a8['gold_support_wrong_hist'])}")
    L.append(f"    uniform random-selector over top-2: {a8['rand_sel2_ans']:.1f}% vs C8 {a8['c8_point_ans']:.1f}%")
    L.append(f"    order diagnostics: top1 tied {a8['order_dep_top1']}, 2nd tied {a8['order_dep_top2']}, "
             f"reversed-order top1 changed {a8['rev_top1_diff']}, top2 changed {a8['rev_top2_diff']}, "
             f"top2 set changed {a8['rev_set2_diff']}")
    L.append("")
    return L, a8


def write_csv(src, path):
    ids = src["ids"]
    gold = src["gold"]
    c8 = src["records_c8"]
    c16 = src["records_c16"]
    has16 = c16 is not None
    header = ["id", "gold"]
    for arm in ("c8", "c16"):
        header += [
            f"{arm}_answer", f"{arm}_correct",
            f"{arm}_ranked", f"{arm}_support1", f"{arm}_support2", f"{arm}_support3",
            f"{arm}_top1", f"{arm}_top2", f"{arm}_top3",
            f"{arm}_runner_up", f"{arm}_runner_up_correct",
            f"{arm}_oracle_top2", f"{arm}_oracle_top3", f"{arm}_oracle_all",
            f"{arm}_gold_rank", f"{arm}_gold_support",
            f"{arm}_gap", f"{arm}_near_tie", f"{arm}_abstain",
        ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for i in ids:
            row = [i, gold[i]]
            for arm, recs in (("c8", c8), ("c16", c16)):
                if recs is None:
                    row += [""] * 19
                    continue
                r = recs[i]
                ranked = "|".join(f"{k}={c}" for k, c in zip(r["keys"], r["counts"]))
                row += [
                    r["top1"] if r["top1"] is not None else "", r["ok"],
                    ranked,
                    r["sup1"], "" if r["sup2"] is None else r["sup2"],
                    "" if r["top3"] is None else r["counts"][2],
                    r["top1"] if r["top1"] is not None else "",
                    r["top2"] if r["top2"] is not None else "",
                    r["top3"] if r["top3"] is not None else "",
                    r["top2"] if r["top2"] is not None else "", r["runner_up_correct"],
                    r["oracle2"], r["oracle3"], r["oracleall"],
                    "" if r["gold_rank"] is None else r["gold_rank"], r["gold_support"],
                    "" if r["gap"] is None else r["gap"], r["near_tie"], r["abstain"],
                ]
            w.writerow(row)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gold-file", default=DEFAULT_GOLD)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--csv-dir", default=RESULTS)
    ap.add_argument("--source", action="append", default=[],
                    help="only this source name (repeatable); default all")
    args = ap.parse_args()

    gold_raw = load_gold(args.gold_file)

    specs = [s for s in SOURCES if not args.source or s["name"] in args.source]
    if not specs:
        raise SystemExit(f"no matching sources among {[s['name'] for s in SOURCES]}")

    L = [
        "RecycleLM audit-select probe (zero-GPU; verifier/auditor as selector over existing traces)",
        f"gold: {os.path.abspath(args.gold_file)}",
        "answer extractor: harness/analyze_modality_vote.py extract_cot "
        "(balanced \\boxed{} -> 'the answer is' line -> last number)",
        "C8 = cot streams 0-7; C16 = all 16 cot streams; mixed runs have only 8 cot streams.",
        "coverage/oracle/lift/runner-up use answered problems only; abstention reported separately.",
        "ranking: support desc, tie-break by summed stream index then canonical key.",
        "",
        "Provenance:",
    ]
    for s in specs:
        if s["kind"] == "legacy":
            L.append(f"  {s['name']}: dev200_s16 provenance unconfirmed "
                     f"(no provenance sidecar; columns id,stream,start_pos,end_pos,text, no header; "
                     f"text newline-flattened to spaces; \\boxed present in 2616/3200 streams)")
        else:
            L.append(f"  {s['name']}: {s.get('prov', 'n/a')}")
    L.append("")

    gate_rows = []
    for spec in specs:
        src = build_source(spec, gold_raw)
        section, a8 = format_source(src)
        L += section
        csv_path = os.path.join(args.csv_dir, f"audit_probe_{spec['name']}.csv")
        write_csv(src, csv_path)
        L.append(f"  [written] {os.path.relpath(csv_path, ROOT)}")
        L.append("")
        gate_rows.append((spec["name"], "C8", a8))
        if src["records_c16"] is not None:
            gate_rows.append((spec["name"], "C16", aggregate(src["records_c16"], src["ids"])))

    L.append("=== GATES (G0b pass: oracle@2 >= C8+10 AND runner-up >= 30%; "
             "dead-end: oracle@2 <= C8+3 OR runner-up < 15%) ===")
    L.append("  source                 arm   oracle@2   C8    lift   runner-up%  verdict")
    verdicts = []
    for name, arm, a in gate_rows:
        verdict, lift = gate_verdict(a)
        verdicts.append((name, arm, verdict, lift, a))
        L.append(f"  {name:<22} {arm:>3}  {a['oracle2_ans']:>8.1f} {a['c8_point_ans']:>6.1f} "
                 f"{lift:>+6.1f}   {a['runner_up_rate']:>8.1f}   {verdict}")
    L.append("")

    passes = [v for v in verdicts if v[2] == "PASS"]
    primary = next((v for v in verdicts if v[0] == "dev200_s16" and v[1] == "C8"), None)
    L.append("Note: oracle@2 - C8 (in points) and the runner-up recovery count are the same "
             "quantity (when C8 is wrong, gold in top-2 <=> gold is the rank-2 answer).")
    L.append("Recommendation (one line):")
    if primary is not None:
        pn, pa, pv, plift, pagg = primary
        if pv == "PASS":
            L.append(f"  audit-select IS worth a GPU probe: primary {pn}/{pa} clears G0b "
                     f"(oracle@2 {pagg['oracle2_ans']:.1f}%, C8 {pagg['c8_point_ans']:.1f}%, "
                     f"lift {plift:+.1f} pts, runner-up {pagg['runner_up_rate']:.1f}%).")
        elif pv == "DEAD-END":
            L.append(f"  audit-select is a DEAD END on the primary {pn}/{pa} set "
                     f"(lift {plift:+.1f} pts, runner-up {pagg['runner_up_rate']:.1f}%); go to a "
                     f"different angle (no GPU spend on a selector).")
        else:
            extra = [f"{n}/{a}" for n, a, v, _, _ in verdicts if v == "PASS"]
            tail = (f"only {', '.join(extra)} clear G0b, and that is a smaller/secondary pool. "
                    if extra else "no source clears G0b. ")
            L.append(f"  Not worth a GPU probe on this evidence: primary {pn}/{pa} is BORDERLINE "
                     f"(lift {plift:+.1f} pts, below the +10 bar; runner-up {pagg['runner_up_rate']:.1f}%); "
                     f"{tail}Pivot to a different angle or a cheaper zero-GPU selector proxy first.")
    else:
        L.append("  " + "; ".join(f"{n}/{a}={v}" for n, a, v, _, _ in verdicts))
    L.append("")
    L.append("Suspicious-data / caveats:")
    L.append("  - dev200_s16 provenance unconfirmed (no provenance sidecar). Its text is "
             "newline-flattened and 1201/3200 streams (37.5%) fall back to last_number "
             "extraction, so its per-stream answers are noisier than the clean m3 dumps.")
    L.append("  - the runner-up is ambiguous whenever the 2nd support tier is tied; the "
             "deterministic tie-break is reported alongside an order-independent "
             "'gold at 2nd support tier' count.")
    L.append("  - m3_*_mixed contain only 8 cot streams, so C16 is undefined there.")
    L.append("  - m3 pools are smaller (40 / 100 problems) than dev200_s16 (200); "
             "per-source gate verdicts inherit that uncertainty.")
    L.append("")

    text = "\n".join(L).rstrip() + "\n"
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(text)
    sys.stdout.write(text)
    sys.stdout.write(f"\n[written] {args.out}\n")


if __name__ == "__main__":
    main()
