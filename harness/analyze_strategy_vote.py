r"""Strategy-diversity probe: HOM8 (8x identical cot) vs DIV8 (8 strategies),
plus N-arm in-context exemplar arms (ZS8 / FS8 / FD8).

Reads the clean, escaped stream dumps written by harness/engine/modality.cpp:

    results/s3_<tag>_hom8.tsv   cot,cot,cot,cot,cot,cot,cot,cot
    results/s3_<tag>_div8.tsv   cot,fwd,bwd,table,subq,restate,units,check
    results/s4_1b_zs8.tsv       cot x8, wrapper only, no exemplar
    results/s4_1b_fs8.tsv       cot x8, same exemplar x8
    results/s4_1b_fd8.tsv       cot x8, 8 distinct exemplars

TSV columns: id, stream, mode, finish, generated, ttft_ms, done_ms, text
(text is escaped: \\ -> \, \\n -> newline, \\t -> tab, \\r -> CR)

For each arm and problem it reports tie-aware plurality (point / random-tie
expectation / upper / tie count), oracle@8, gold absence, distinct-answer count,
gold support, per-stream (per-exemplar) accuracy, within-arm pairwise answer
agreement (the manipulation check), per-mode answer + correctness, and pairwise
mode answer agreement. First-K-agree (K=3) uses the TSV done_ms ordering:
streams are considered in finish order and the vote is the first answer that 3
finished streams share, together with the done_ms at which it was reached.
Reached-only accuracy conditions on problems where K agreement was reached.

Every pair of arms in a multi-arm group is compared with exact McNemar and a
tie-aware paired bootstrap (2000 resamples, seed 1234, 95% CI). The verdict for
a ZS8/FS8/FD8 group is PASS when FD8 >= FS8 + 3 pts with a tie-CI excluding 0
AND FD8 >= ZS8 + 5 pts; DEAD when FD8 <= FS8.

Usage (multi-arm, labels name the arms):
    python harness/analyze_strategy_vote.py --tag s4_1b \
      --label ZS8 --tsv results/s4_1b_zs8.tsv \
      --label FS8 --tsv results/s4_1b_fs8.tsv \
      --label FD8 --tsv results/s4_1b_fd8.tsv \
      --out results/strategy_probe_s4_summary.txt

Usage (legacy 2-run mode, unchanged):
    python harness/analyze_strategy_vote.py \
      --run s3_1b,results/s3_1b_hom8.tsv,results/s3_1b_div8.tsv

A single legacy pair can also be given with --tag/--hom/--div.
"""
import argparse
import csv
import os
import random
import statistics
import sys
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import analyze_modality_vote as M  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
DEFAULT_GOLD = os.path.join(ROOT, "data", "gsm8k_dev200.tsv")
DEFAULT_OUT = os.path.join(RESULTS, "strategy_probe_summary.txt")

HEADER = ["id", "stream", "mode", "finish", "generated", "ttft_ms", "done_ms", "text"]
FINISHES = ("eos", "answer", "fence", "cap")
STRATEGIES = ["cot", "fwd", "bwd", "table", "subq", "restate", "units", "check"]
K_AGREE = 3
TIE_BOOTSTRAP_N = 2000
TIE_BOOTSTRAP_SEED = 1234


# ---------------------------------------------------------------- loading ----


def load_rows(path):
    """Parse a modality TSV; accepts any mode string (strategy modes included)."""
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


def streams(by_id, i):
    return [by_id[i][s] for s in sorted(by_id[i])]


def answer_of(row):
    return M.extract_cot(row["text"])[0]


# ------------------------------------------------------------------ voting ----


def plurality_arm(by_id, ids):
    """Per-problem plurality over all streams. Substrate = the mode name, so the
    tie-break prefers agreement across strategies before stream index."""
    res = {}
    for i in ids:
        items = [(answer_of(r), r["mode"], idx) for idx, r in enumerate(streams(by_id, i))]
        res[i] = M.plurality(items)
    return res


def arm_answers(by_id, ids):
    return {i: [answer_of(r) for r in streams(by_id, i)] for i in ids}


def first_k_agree(by_id, i, k=K_AGREE):
    """First answer 3 finished streams share, in done_ms order.

    Returns (answer, reached, done_ms). When no answer reaches k, falls back to
    the plurality over all streams and marks reached=False with done_ms set to
    the last stream's done_ms. The fallback tie-break uses each row's original
    ``stream`` index, matching plurality over the unsorted arm; done_ms ordering
    is only used for the first-K walk.
    """
    rows = sorted(streams(by_id, i), key=lambda r: (r["done_ms"], r["stream"]))
    counts = {}
    last = rows[-1]["done_ms"] if rows else 0.0
    for r in rows:
        a = answer_of(r)
        if a is None:
            continue
        key = M.canon(a)
        counts[key] = counts.get(key, 0) + 1
        if counts[key] == k:
            return a, True, r["done_ms"]
    items = [(answer_of(r), r["mode"], r["stream"]) for r in rows]
    return M.plurality(items)["answer"], False, last


def all_done_ms(by_id, i):
    return max((r["done_ms"] for r in streams(by_id, i)), default=0.0)


# ------------------------------------------------------------- diagnostics ----


def per_mode_map(by_id, ids):
    """mode -> {id: {answer, ok}} for every mode present (one stream per mode)."""
    out = {}
    for i in ids:
        for r in streams(by_id, i):
            out.setdefault(r["mode"], {})[i] = r
    return out


def stream_accuracy_by_mode(by_id, ids, gold):
    """Mode -> accuracy over every stream carrying that mode.

    DIV8 has exactly one stream per mode; HOM8 aggregates its 8 cot streams.
    """
    modes = sorted({r["mode"] for i in ids for r in streams(by_id, i)})
    acc = {}
    for m in modes:
        total = correct = answered = 0
        for i in ids:
            for row in streams(by_id, i):
                if row["mode"] != m:
                    continue
                total += 1
                a = answer_of(row)
                if a is not None:
                    answered += 1
                    correct += 1 if M.canon(a) == M.canon(gold[i]) else 0
        acc[m] = {"total": total, "answered": answered, "correct": correct,
                  "acc": M.pct(correct, total)}
    return acc


def stream_accuracy_by_index(by_id, ids, gold):
    """Accuracy by stream index (useful for HOM8, where all streams are cot)."""
    out = {}
    for i in ids:
        for idx, r in enumerate(streams(by_id, i)):
            d = out.setdefault(idx, [0, 0])
            d[0] += 1
            a = answer_of(r)
            d[1] += 1 if a is not None and M.canon(a) == M.canon(gold[i]) else 0
    return {k: M.pct(v[1], v[0]) for k, v in out.items()}


def finish_stats(by_id, ids):
    """Mode -> finish-reason counts and generated-length summary."""
    out = {}
    for i in ids:
        for r in streams(by_id, i):
            d = out.setdefault(r["mode"], {"finish": {}, "gen": []})
            d["finish"][r["finish"]] = d["finish"].get(r["finish"], 0) + 1
            d["gen"].append(r["generated"])
    return {
        m: {"n": len(v["gen"]), "finish": dict(sorted(v["finish"].items())),
            "gen_mean": statistics.fmean(v["gen"]), "gen_max": max(v["gen"])}
        for m, v in out.items()
    }


def batch_overlap(hom, div, ids):
    """Streams that share both index and mode across arms (here: stream 0 cot).

    Same prompt+seed should give identical text; any difference is the
    batch-composition/numerics confound, since only the other streams' prompts
    changed.
    """
    same_text = same_ans = total = 0
    for i in ids:
        hm = {r["stream"]: r for r in streams(hom, i)}
        dm = {r["stream"]: r for r in streams(div, i)}
        for s in sorted(set(hm) & set(dm)):
            if hm[s]["mode"] != dm[s]["mode"]:
                continue
            total += 1
            same_text += 1 if hm[s]["text"] == dm[s]["text"] else 0
            aa, bb = answer_of(hm[s]), answer_of(dm[s])
            same_ans += 1 if (aa is not None and bb is not None
                              and M.canon(aa) == M.canon(bb)) else 0
    return same_text, same_ans, total


def pairwise_mode_agreement(by_id, ids):
    """For every mode pair: both-answered count, equal-answer fraction, and the
    Jaccard index over (problem_id, canonical_answer) sets."""
    modes = sorted({r["mode"] for i in ids for r in streams(by_id, i)})
    ans = per_mode_map(by_id, ids)
    out = {}
    for a in range(len(modes)):
        for b in range(a + 1, len(modes)):
            ma, mb = modes[a], modes[b]
            both = agree = text_same = 0
            set_a, set_b = set(), set()
            for i in ids:
                ra, rb = ans[ma].get(i), ans[mb].get(i)
                if ra is None or rb is None:
                    continue
                va, vb = answer_of(ra), answer_of(rb)
                if va is not None:
                    set_a.add((i, M.canon(va)))
                if vb is not None:
                    set_b.add((i, M.canon(vb)))
                if va is not None and vb is not None:
                    both += 1
                    agree += 1 if M.canon(va) == M.canon(vb) else 0
                    text_same += 1 if ra["text"] == rb["text"] else 0
            union = len(set_a | set_b)
            out[(ma, mb)] = {
                "both": both, "agree": agree, "text_same": text_same,
                "agree_frac": (agree / both) if both else float("nan"),
                "jaccard": (len(set_a & set_b) / union) if union else float("nan"),
            }
    return out


def distinct_counts(by_id, ids):
    """Per-problem count of distinct non-None canonical answers and abstentions."""
    distinct, abstained = {}, {}
    for i in ids:
        vals = [a for a in (answer_of(r) for r in streams(by_id, i)) if a is not None]
        distinct[i] = len({M.canon(v) for v in vals})
        abstained[i] = sum(1 for a in (answer_of(r) for r in streams(by_id, i)) if a is None)
    return distinct, abstained


def gold_support(by_id, ids, gold):
    sup = {}
    for i in ids:
        g = M.canon(gold[i])
        sup[i] = sum(1 for r in streams(by_id, i)
                     if answer_of(r) is not None and M.canon(answer_of(r)) == g)
    return sup


def oracle(by_id, ids, gold):
    out = {}
    for i in ids:
        g = M.canon(gold[i])
        out[i] = any(answer_of(r) is not None and M.canon(answer_of(r)) == g
                     for r in streams(by_id, i))
    return out


# ------------------------------------------------------------------- stats ----


def mcnemar(a_ok, b_ok, ids):
    b = sum(1 for i in ids if a_ok[i] and not b_ok[i])
    c = sum(1 for i in ids if not a_ok[i] and b_ok[i])
    return b, c, M.mcnemar_exact(b, c)


def paired_point_bootstrap(a_ok, b_ok, ids, seed=TIE_BOOTSTRAP_SEED, reps=TIE_BOOTSTRAP_N):
    rng = random.Random(seed)
    n = len(ids)
    if n == 0:
        return 0.0, 0.0
    diffs = []
    for _ in range(reps):
        idx = [ids[rng.randrange(n)] for _ in range(n)]
        d = sum((1 if a_ok[i] else 0) - (1 if b_ok[i] else 0) for i in idx)
        diffs.append(100.0 * d / n)
    diffs.sort()
    return diffs[int(0.025 * reps)], diffs[min(reps - 1, int(0.975 * reps))]


def hist(values):
    h = {}
    for v in values:
        h[v] = h.get(v, 0) + 1
    return dict(sorted(h.items()))


# ---------------------------------------------------------------- analysis ----


def analyze_pair(tag, hom_path, div_path, gold_raw):
    _, hom = load_rows(hom_path)
    _, div = load_rows(div_path)
    hom_ids = set(hom)
    div_ids = set(div)
    if hom_ids != div_ids:
        raise SystemExit(f"{tag}: hom/div id mismatch {sorted(hom_ids ^ div_ids)}")
    ids = sorted(hom_ids)
    for i in ids:
        if i not in gold_raw:
            raise SystemExit(f"{tag}: no gold for id={i}")
    gold = {i: M.normalize_number(gold_raw[i]) for i in ids}

    hom_pl = plurality_arm(hom, ids)
    div_pl = plurality_arm(div, ids)
    hom_met = M.tie_metrics(hom_pl, ids, gold)
    div_met = M.tie_metrics(div_pl, ids, gold)

    hom_or = oracle(hom, ids, gold)
    div_or = oracle(div, ids, gold)

    hom_distinct, hom_abstain = distinct_counts(hom, ids)
    div_distinct, div_abstain = distinct_counts(div, ids)
    hom_support = gold_support(hom, ids, gold)
    div_support = gold_support(div, ids, gold)

    div_modes = sorted({r["mode"] for i in ids for r in streams(div, i)})
    hom_modes = sorted({r["mode"] for i in ids for r in streams(hom, i)})
    div_mode_acc = stream_accuracy_by_mode(div, ids, gold)
    hom_mode_acc = stream_accuracy_by_mode(hom, ids, gold)
    hom_index_acc = stream_accuracy_by_index(hom, ids, gold)
    agreement = pairwise_mode_agreement(div, ids)
    overlap = batch_overlap(hom, div, ids)
    hom_finish = finish_stats(hom, ids)
    div_finish = finish_stats(div, ids)

    hom_fk, div_fk = {}, {}
    for i in ids:
        hom_fk[i] = first_k_agree(hom, i)
        div_fk[i] = first_k_agree(div, i)
    hom_allfin = {i: all_done_ms(hom, i) for i in ids}
    div_allfin = {i: all_done_ms(div, i) for i in ids}

    def fk_stats(fk, allfin):
        ok = {i: fk[i][0] is not None and M.canon(fk[i][0]) == M.canon(gold[i]) for i in ids}
        reached = sum(1 for i in ids if fk[i][1])
        fk_ms = [fk[i][2] for i in ids]
        af_ms = [allfin[i] for i in ids]
        return {
            "correct": sum(ok.values()),
            "acc": M.pct(sum(ok.values()), len(ids)),
            "ok": ok,
            "reached": reached,
            "median_fk_ms": statistics.median(fk_ms) if fk_ms else float("nan"),
            "median_all_ms": statistics.median(af_ms) if af_ms else float("nan"),
        }

    hom_fks = fk_stats(hom_fk, hom_allfin)
    div_fks = fk_stats(div_fk, div_allfin)

    hom_ok = {i: hom_pl[i]["answer"] is not None and M.canon(hom_pl[i]["answer"]) == M.canon(gold[i])
              for i in ids}
    div_ok = {i: div_pl[i]["answer"] is not None and M.canon(div_pl[i]["answer"]) == M.canon(gold[i])
              for i in ids}

    b, c, p = mcnemar(div_ok, hom_ok, ids)
    lo_tie, hi_tie = M.tie_aware_paired_bootstrap(
        div_pl, hom_pl, ids, gold, seed=TIE_BOOTSTRAP_SEED, reps=TIE_BOOTSTRAP_N)
    delta = div_met["point"] - hom_met["point"]
    lo_p, hi_p = paired_point_bootstrap(div_ok, hom_ok, ids)

    fb, fc, fp = mcnemar(div_fks["ok"], hom_fks["ok"], ids)
    flo, fhi = paired_point_bootstrap(div_fks["ok"], hom_fks["ok"], ids)
    fdelta = div_fks["acc"] - hom_fks["acc"]

    if delta >= 5.0 and lo_tie > 0:
        verdict = "PASS"
    elif delta <= 1.0:
        verdict = "DEAD"
    else:
        verdict = "PARTIAL"

    return {
        "tag": tag, "hom_path": hom_path, "div_path": div_path,
        "ids": ids, "gold": gold,
        "hom_pl": hom_pl, "div_pl": div_pl,
        "hom_met": hom_met, "div_met": div_met,
        "hom_oracle": hom_or, "div_oracle": div_or,
        "hom_distinct": hom_distinct, "div_distinct": div_distinct,
        "hom_abstain": hom_abstain, "div_abstain": div_abstain,
        "hom_support": hom_support, "div_support": div_support,
        "div_modes": div_modes, "hom_modes": hom_modes,
        "div_mode_acc": div_mode_acc, "hom_mode_acc": hom_mode_acc,
        "hom_index_acc": hom_index_acc,
        "overlap": overlap,
        "hom_finish": hom_finish, "div_finish": div_finish,
        "agreement": agreement,
        "hom_fk": hom_fk, "div_fk": div_fk,
        "hom_fks": hom_fks, "div_fks": div_fks,
        "hom_allfin": hom_allfin, "div_allfin": div_allfin,
        "div_by_mode": per_mode_map(div, ids),
        "hom_ok": hom_ok, "div_ok": div_ok,
        "mcnemar": {"b": b, "c": c, "p": p, "delta": delta,
                    "lo_tie": lo_tie, "hi_tie": hi_tie, "lo": lo_p, "hi": hi_p},
        "fk_test": {"b": fb, "c": fc, "p": fp, "delta": fdelta, "lo": flo, "hi": fhi},
        "verdict": verdict,
    }


# ------------------------------------------------------------- multi-arm ----
#
# A multi-arm group is N TSVs that share the same problem ids and gold. Every
# arm gets the same per-arm diagnostics; every pair of arms gets a McNemar +
# tie-aware paired bootstrap comparison. The old analyze_pair path is untouched
# so the two-run report stays byte-compatible for existing callers.


def stream_index_stats(by_id, ids, gold):
    """Per stream index (per exemplar in the exemplar arms): total/answered/correct."""
    out = {}
    for i in ids:
        g = M.canon(gold[i])
        for idx, r in enumerate(streams(by_id, i)):
            d = out.setdefault(idx, {"total": 0, "answered": 0, "correct": 0})
            d["total"] += 1
            a = answer_of(r)
            if a is not None:
                d["answered"] += 1
                d["correct"] += 1 if M.canon(a) == g else 0
    return out


def within_arm_agreement(by_id, ids):
    """Pairwise answer agreement across the streams of one arm.

    For every problem, every unordered stream pair is counted only when both
    streams produced an answer; ``agree`` counts equal canonical answers. This
    is the manipulation check: distinct exemplars should lower agreement.
    """
    pair_both = pair_agree = 0
    all_same = answered_problems = 0
    distinct_total = 0
    for i in ids:
        vals = [answer_of(r) for r in streams(by_id, i)]
        keys = [M.canon(a) for a in vals if a is not None]
        if keys:
            answered_problems += 1
        distinct_total += len(set(keys))
        if len(keys) == len(vals) and len(set(keys)) == 1:
            all_same += 1
        for a, b in combinations(vals, 2):
            if a is None or b is None:
                continue
            pair_both += 1
            pair_agree += 1 if M.canon(a) == M.canon(b) else 0
    n = len(ids)
    return {
        "pair_both": pair_both,
        "pair_agree": pair_agree,
        "pair_frac": M.pct(pair_agree, pair_both) if pair_both else float("nan"),
        "all_same": all_same,
        "answered_problems": answered_problems,
        "mean_distinct": (distinct_total / n) if n else float("nan"),
    }


def first_k_stats(by_id, ids, gold):
    """First-K-agree overall and reached-only accuracy."""
    fk = {i: first_k_agree(by_id, i) for i in ids}
    allfin = {i: all_done_ms(by_id, i) for i in ids}
    ok = {i: fk[i][0] is not None and M.canon(fk[i][0]) == M.canon(gold[i]) for i in ids}
    reached_ids = [i for i in ids if fk[i][1]]
    ok_reached = {i: ok[i] for i in reached_ids}
    correct = sum(ok.values())
    correct_reached = sum(ok_reached.values())
    fk_ms = [fk[i][2] for i in ids]
    af_ms = [allfin[i] for i in ids]
    return {
        "fk": fk, "allfin": allfin, "ok": ok, "ok_reached": ok_reached,
        "correct": correct, "acc": M.pct(correct, len(ids)),
        "reached": len(reached_ids),
        "correct_reached": correct_reached,
        "acc_reached": M.pct(correct_reached, len(reached_ids)),
        "median_fk_ms": statistics.median(fk_ms) if fk_ms else float("nan"),
        "median_all_ms": statistics.median(af_ms) if af_ms else float("nan"),
    }


def analyze_arm(label, path, by_id, ids, gold):
    pl = plurality_arm(by_id, ids)
    met = M.tie_metrics(pl, ids, gold)
    orc = oracle(by_id, ids, gold)
    distinct, abstain = distinct_counts(by_id, ids)
    support = gold_support(by_id, ids, gold)
    modes = sorted({r["mode"] for i in ids for r in streams(by_id, i)})
    mode_acc = stream_accuracy_by_mode(by_id, ids, gold)
    index_stats = stream_index_stats(by_id, ids, gold)
    agreement = within_arm_agreement(by_id, ids)
    finish = finish_stats(by_id, ids)
    fks = first_k_stats(by_id, ids, gold)
    ok = {i: pl[i]["answer"] is not None and M.canon(pl[i]["answer"]) == M.canon(gold[i])
          for i in ids}
    gold_absent_streams = sum(1 for i in ids if support[i] == 0)
    gold_absent_top = sum(1 for i in ids
                          if not any(M.canon(x) == M.canon(gold[i]) for x in pl[i]["top"]))
    sidecar, sidecar_path = M.load_sidecar(path)
    return {
        "label": label, "path": path, "by_id": by_id,
        "modes": modes, "pl": pl, "met": met, "oracle": orc,
        "distinct": distinct, "abstain": abstain, "support": support,
        "mode_acc": mode_acc, "index_stats": index_stats,
        "agreement": agreement, "finish": finish, "fks": fks, "ok": ok,
        "gold_absent_streams": gold_absent_streams,
        "gold_absent_top": gold_absent_top,
        "sidecar": sidecar, "sidecar_path": sidecar_path,
    }


def pair_test(ra, rb, ids, gold):
    """ra vs rb: positive delta means ra is better. b = ra-only-correct."""
    b, c, p = mcnemar(ra["ok"], rb["ok"], ids)
    lo, hi = paired_point_bootstrap(ra["ok"], rb["ok"], ids)
    lo_tie, hi_tie = M.tie_aware_paired_bootstrap(
        ra["pl"], rb["pl"], ids, gold, seed=TIE_BOOTSTRAP_SEED, reps=TIE_BOOTSTRAP_N)
    fb, fc, fp = mcnemar(ra["fks"]["ok"], rb["fks"]["ok"], ids)
    flo, fhi = paired_point_bootstrap(ra["fks"]["ok"], rb["fks"]["ok"], ids)
    return {
        "a": ra["label"], "b": rb["label"],
        "delta": ra["met"]["point"] - rb["met"]["point"],
        "lo_tie": lo_tie, "hi_tie": hi_tie, "lo": lo, "hi": hi,
        "b_only_a": b, "b_only_b": c, "p": p,
        "fk_delta": ra["fks"]["acc"] - rb["fks"]["acc"],
        "fk_lo": flo, "fk_hi": fhi,
        "fk_only_a": fb, "fk_only_b": fc, "fk_p": fp,
    }


def lookup_pair(pairs, a, b):
    """Return the comparison with a positive ``delta`` meaning a better than b."""
    for t in pairs:
        if t["a"] == a and t["b"] == b:
            return t
        if t["a"] == b and t["b"] == a:
            return {
                "a": a, "b": b,
                "delta": -t["delta"],
                "lo_tie": -t["hi_tie"], "hi_tie": -t["lo_tie"],
                "lo": -t["hi"], "hi": -t["lo"],
                "b_only_a": t["b_only_b"], "b_only_b": t["b_only_a"], "p": t["p"],
                "fk_delta": -t["fk_delta"],
                "fk_lo": -t["fk_hi"], "fk_hi": -t["fk_lo"],
                "fk_only_a": t["fk_only_b"], "fk_only_b": t["fk_only_a"], "fk_p": t["fk_p"],
            }
    return None


def multi_verdict(arms, pairs):
    by_label = {a["label"]: a for a in arms}
    if {"FD8", "FS8", "ZS8"} <= set(by_label):
        dfs = lookup_pair(pairs, "FD8", "FS8")
        dzs = lookup_pair(pairs, "FD8", "ZS8")
        if dfs and dzs:
            if dfs["delta"] >= 3.0 and dfs["lo_tie"] > 0 and dzs["delta"] >= 5.0:
                return "PASS", dfs, dzs
            if dfs["delta"] <= 0.0:
                return "DEAD", dfs, dzs
            return "PARTIAL", dfs, dzs
    if len(arms) == 2:
        t = pairs[0]
        if t["delta"] >= 5.0 and t["lo_tie"] > 0:
            return "PASS", t, None
        if t["delta"] <= 1.0:
            return "DEAD", t, None
        return "PARTIAL", t, None
    return "n/a", None, None


def analyze_multi(tag, arm_specs, gold_raw):
    """arm_specs: list of (label, path) for one group."""
    if len(arm_specs) < 2:
        raise SystemExit(f"{tag}: need at least 2 arms, got {len(arm_specs)}")
    labels = [l for l, _ in arm_specs]
    if len(set(labels)) != len(labels):
        raise SystemExit(f"{tag}: duplicate arm labels {labels}")

    loaded = []
    for label, path in arm_specs:
        _, by_id = load_rows(path)
        loaded.append((label, path, by_id))
    ids = sorted(loaded[0][2])
    for label, _, by_id in loaded[1:]:
        if set(by_id) != set(ids):
            raise SystemExit(f"{tag}/{label}: id mismatch {sorted(set(by_id) ^ set(ids))}")
    for i in ids:
        if i not in gold_raw:
            raise SystemExit(f"{tag}: no gold for id={i}")
    gold = {i: M.normalize_number(gold_raw[i]) for i in ids}

    arms = [analyze_arm(label, path, by_id, ids, gold) for label, path, by_id in loaded]

    pairs = []
    for ra, rb in combinations(arms, 2):
        pairs.append(pair_test(ra, rb, ids, gold))

    verdict, dfs, dzs = multi_verdict(arms, pairs)

    manip = None
    by_label = {a["label"]: a for a in arms}
    if "FS8" in by_label and "FD8" in by_label:
        fs = by_label["FS8"]["agreement"]["pair_frac"]
        fd = by_label["FD8"]["agreement"]["pair_frac"]
        manip = {"fs": fs, "fd": fd, "delta": fs - fd}

    return {
        "tag": tag, "ids": ids, "gold": gold, "arms": arms, "pairs": pairs,
        "verdict": verdict, "verdict_fs": dfs, "verdict_zs": dzs,
        "manipulation": manip,
    }


def write_multi_csv(r, path):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        header = ["id", "gold"]
        for a in r["arms"]:
            lbl = a["label"]
            header += [f"{lbl}_answer", f"{lbl}_ok", f"{lbl}_tied", f"{lbl}_top",
                       f"{lbl}_oracle", f"{lbl}_gold_support", f"{lbl}_ndistinct",
                       f"{lbl}_fk_answer", f"{lbl}_fk_ok", f"{lbl}_fk_reached"]
        w.writerow(header)
        for i in r["ids"]:
            row = [i, r["gold"][i]]
            for a in r["arms"]:
                pl = a["pl"][i]
                fk = a["fks"]["fk"][i]
                row += [
                    pl["answer"] if pl["answer"] is not None else "",
                    int(a["ok"][i]),
                    int(pl["tied"]),
                    "|".join(str(x) for x in pl["top"]),
                    int(a["oracle"][i]),
                    a["support"][i],
                    a["distinct"][i],
                    fk[0] if fk[0] is not None else "",
                    int(a["fks"]["ok"][i]),
                    int(fk[1]),
                ]
            w.writerow(row)


def format_multi(r):
    L = []
    ids = r["ids"]
    n = len(ids)
    arms = r["arms"]
    L.append(f"=== {r['tag']} (n={n}) ===")
    L.append("arms:")
    for a in arms:
        sc = a.get("sidecar") or {}
        src = a["sidecar_path"] or "(no sidecar)"
        L.append(f"  {a['label']:>6}: {a['path']}  modes={a['modes']}  "
                 f"wrapper={sc.get('wrapper', 'n/a')} exemplars={sc.get('exemplars', 'n/a')} "
                 f"max_trace={sc.get('max_trace', 'n/a')} seed={sc.get('seed', 'n/a')} "
                 f"n_ctx={sc.get('n_ctx', 'n/a')}")
        L.append(f"          sidecar {src}")

    L.append("")
    L.append("arm plurality (tie-aware; point uses the deterministic tie-break):")
    L.append(f"  {'arm':>6} {'correct':>8} {'point%':>7} {'rand%':>7} {'upper%':>7} {'tied':>5}")
    for a in arms:
        m = a["met"]
        L.append(f"  {a['label']:>6} {m['correct']:>8} {m['point']:>7.1f} {m['rand']:>7.1f} "
                 f"{m['upper']:>7.1f} {m['tied']:>5}")

    L.append("")
    L.append("oracle@8 and gold absence (no stream answers gold / gold not in plurality top):")
    for a in arms:
        L.append(f"  {a['label']:>6}: oracle {sum(a['oracle'].values())}/{n} "
                 f"({M.pct(sum(a['oracle'].values()), n):.1f}%), "
                 f"gold absent {a['gold_absent_streams']}/{n} "
                 f"({M.pct(a['gold_absent_streams'], n):.1f}%), "
                 f"absent from plurality top {a['gold_absent_top']}")

    L.append("")
    L.append("distinct answers per problem (non-None):")
    for a in arms:
        L.append(f"  {a['label']:>6}: mean {statistics.fmean(a['distinct'].values()):.2f}, "
                 f"hist {fmt_hist(hist(a['distinct'].values()))}, "
                 f"abstentions {fmt_hist(hist(a['abstain'].values()))}")

    L.append("")
    L.append("gold support (streams whose answer == gold):")
    for a in arms:
        L.append(f"  {a['label']:>6}: hist {fmt_hist(hist(a['support'].values()))}")

    L.append("")
    L.append("per-stream (per-exemplar) accuracy (stream j = exemplar j in FD8):")
    L.append(f"  {'arm':>6} {'stream':>6} {'correct':>8} {'acc%':>7} {'answered':>9}")
    for a in arms:
        for idx in sorted(a["index_stats"]):
            s = a["index_stats"][idx]
            L.append(f"  {a['label']:>6} {idx:>6} {s['correct']:>8} "
                     f"{M.pct(s['correct'], s['total']):>7.1f} {s['answered']:>9}")

    L.append("")
    L.append("within-arm pairwise answer agreement (manipulation check; both-answered pairs):")
    L.append(f"  {'arm':>6} {'agree/both':>12} {'pair%':>7} {'all-8-same':>11} {'mean distinct':>14}")
    for a in arms:
        g = a["agreement"]
        L.append(f"  {a['label']:>6} {g['pair_agree']:>6}/{g['pair_both']:<5} "
                 f"{g['pair_frac']:>7.1f} {g['all_same']:>11} {g['mean_distinct']:>14.2f}")
    if r["manipulation"]:
        m = r["manipulation"]
        L.append(f"  manipulation check FS8 - FD8: {m['fs']:.1f} - {m['fd']:.1f} "
                 f"= {m['delta']:+.1f} pts (positive = FS8 streams agree more)")

    L.append("")
    L.append(f"first-K-agree (K={K_AGREE}, streams in done_ms order):")
    L.append(f"  {'arm':>6} {'correct':>8} {'acc%':>7} {'reached':>8} {'reach%':>7} "
             f"{'reach-only':>11} {'median fk ms':>13} {'median all ms':>14}")
    for a in arms:
        s = a["fks"]
        L.append(f"  {a['label']:>6} {s['correct']:>8} {s['acc']:>7.1f} {s['reached']:>8} "
                 f"{M.pct(s['reached'], n):>7.1f} "
                 f"{s['correct_reached']:>5}/{s['reached']:<5} "
                 f"{s['median_fk_ms']:>13.1f} {s['median_all_ms']:>14.1f}")

    L.append("")
    L.append("finish reasons and length per mode (cap = truncated at max_trace):")
    for a in arms:
        for m in sorted(a["finish"]):
            v = a["finish"][m]
            L.append(f"  {a['label']:>6} {m:>7}: n={v['n']} finish={v['finish']} "
                     f"gen mean {v['gen_mean']:.1f}, max {v['gen_max']}")

    L.append("")
    L.append(f"all pairwise comparisons (tie-aware paired bootstrap, {TIE_BOOTSTRAP_N} "
             f"resamples, seed {TIE_BOOTSTRAP_SEED}, 95% CI; positive delta = first arm better):")
    L.append(f"  {'pair':>13} {'pl delta':>9} {'tieCI':>21} {'pointCI':>19} "
             f"{'b/c':>9} {'p':>8} {'fk delta':>9} {'fkCI':>18} {'fk p':>8}")
    for t in r["pairs"]:
        L.append(f"  {t['a'] + ' vs ' + t['b']:>13} {t['delta']:>+9.1f} "
                 f"[{t['lo_tie']:>+6.1f},{t['hi_tie']:>+6.1f}] "
                 f"[{t['lo']:>+5.1f},{t['hi']:>+5.1f}] "
                 f"{str(t['b_only_a']) + '/' + str(t['b_only_b']):>9} {t['p']:>8.4f} "
                 f"{t['fk_delta']:>+9.1f} [{t['fk_lo']:>+6.1f},{t['fk_hi']:>+6.1f}] "
                 f"{t['fk_p']:>8.4f}")

    L.append("")
    v = r["verdict"]
    if v == "PASS":
        L.append(f"verdict: PASS (FD8 >= FS8 + 3 with tieCI excluding 0, and "
                 f"FD8 >= ZS8 + 5): FS8 delta {r['verdict_fs']['delta']:+.1f} "
                 f"tieCI [{r['verdict_fs']['lo_tie']:+.1f}, {r['verdict_fs']['hi_tie']:+.1f}], "
                 f"ZS8 delta {r['verdict_zs']['delta']:+.1f}")
    elif v == "DEAD":
        L.append(f"verdict: DEAD (FD8 <= FS8): FS8 delta "
                 f"{r['verdict_fs']['delta']:+.1f} tieCI "
                 f"[{r['verdict_fs']['lo_tie']:+.1f}, {r['verdict_fs']['hi_tie']:+.1f}]")
    elif v == "PARTIAL":
        fs = f" FS8 delta {r['verdict_fs']['delta']:+.1f} tieCI " \
             f"[{r['verdict_fs']['lo_tie']:+.1f}, {r['verdict_fs']['hi_tie']:+.1f}]" \
             if r["verdict_fs"] else ""
        zs = f", ZS8 delta {r['verdict_zs']['delta']:+.1f}" if r["verdict_zs"] else ""
        L.append(f"verdict: PARTIAL (neither PASS nor DEAD){fs}{zs}")
    else:
        L.append("verdict: n/a (no ZS8/FS8/FD8 labels and not a 2-arm group)")
    L.append("")
    return "\n".join(L)


# ------------------------------------------------------------------- output ----


def write_csv(r, path):
    modes = STRATEGIES
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        header = ["id", "gold",
                  "hom_answer", "hom_ok", "hom_tied", "hom_top",
                  "div_answer", "div_ok", "div_tied", "div_top",
                  "hom_oracle", "div_oracle",
                  "hom_ndistinct", "div_ndistinct",
                  "hom_gold_support", "div_gold_support",
                  "hom_fk_answer", "hom_fk_ok", "hom_fk_reached", "hom_fk_done_ms",
                  "div_fk_answer", "div_fk_ok", "div_fk_reached", "div_fk_done_ms",
                  "hom_all_done_ms", "div_all_done_ms"]
        header += [f"div_{m}" for m in modes]
        header += [f"div_{m}_ok" for m in modes]
        w.writerow(header)
        for i in r["ids"]:
            g = M.canon(r["gold"][i])
            hp, dp = r["hom_pl"][i], r["div_pl"][i]
            row = [
                i, r["gold"][i],
                hp["answer"] if hp["answer"] is not None else "", int(r["hom_ok"][i]),
                int(hp["tied"]), "|".join(str(x) for x in hp["top"]),
                dp["answer"] if dp["answer"] is not None else "", int(r["div_ok"][i]),
                int(dp["tied"]), "|".join(str(x) for x in dp["top"]),
                int(r["hom_oracle"][i]), int(r["div_oracle"][i]),
                r["hom_distinct"][i], r["div_distinct"][i],
                r["hom_support"][i], r["div_support"][i],
                r["hom_fk"][i][0] if r["hom_fk"][i][0] is not None else "",
                int(r["hom_fks"]["ok"][i]), int(r["hom_fk"][i][1]), round(r["hom_fk"][i][2], 3),
                r["div_fk"][i][0] if r["div_fk"][i][0] is not None else "",
                int(r["div_fks"]["ok"][i]), int(r["div_fk"][i][1]), round(r["div_fk"][i][2], 3),
                round(r["hom_allfin"][i], 3), round(r["div_allfin"][i], 3),
            ]
            for m in modes:
                cell = r["div_by_mode"].get(m, {}).get(i)
                row.append(answer_of(cell) if cell is not None else "")
            for m in modes:
                cell = r["div_by_mode"].get(m, {}).get(i)
                if cell is None:
                    row.append("")
                else:
                    a = answer_of(cell)
                    row.append(int(a is not None and M.canon(a) == g))
            w.writerow(row)


def fmt_hist(h):
    return ", ".join(f"{k}:{v}" for k, v in h.items())


def format_pair(r):
    L = []
    ids = r["ids"]
    n = len(ids)
    L.append(f"=== {r['tag']} (n={n}) ===")
    L.append(f"  hom8: {r['hom_path']}  modes={r['hom_modes']}")
    L.append(f"  div8: {r['div_path']}  modes={r['div_modes']}")

    L.append("")
    L.append("arm plurality (tie-aware; point uses the deterministic tie-break):")
    L.append(f"  {'arm':>6} {'correct':>8} {'point%':>7} {'tied':>5} {'rand%':>7} {'upper%':>7}")
    for name, m in (("HOM8", r["hom_met"]), ("DIV8", r["div_met"])):
        L.append(f"  {name:>6} {m['correct']:>8} {m['point']:>7.1f} {m['tied']:>5} "
                 f"{m['rand']:>7.1f} {m['upper']:>7.1f}")

    L.append("")
    L.append("oracle@8 (any stream correct):")
    L.append(f"  HOM8: {sum(r['hom_oracle'].values())}/{n} "
             f"({M.pct(sum(r['hom_oracle'].values()), n):.1f}%)")
    L.append(f"  DIV8: {sum(r['div_oracle'].values())}/{n} "
             f"({M.pct(sum(r['div_oracle'].values()), n):.1f}%)")

    L.append("")
    L.append("distinct answers per problem (non-None):")
    L.append(f"  HOM8: mean {statistics.fmean(r['hom_distinct'].values()):.2f}, "
             f"hist {fmt_hist(hist(r['hom_distinct'].values()))}")
    L.append(f"  DIV8: mean {statistics.fmean(r['div_distinct'].values()):.2f}, "
             f"hist {fmt_hist(hist(r['div_distinct'].values()))}")
    L.append("stream abstentions per problem:")
    L.append(f"  HOM8: hist {fmt_hist(hist(r['hom_abstain'].values()))}")
    L.append(f"  DIV8: hist {fmt_hist(hist(r['div_abstain'].values()))}")

    L.append("")
    L.append("gold support (streams whose answer == gold):")
    L.append(f"  HOM8: hist {fmt_hist(hist(r['hom_support'].values()))}")
    L.append(f"  DIV8: hist {fmt_hist(hist(r['div_support'].values()))}")

    L.append("")
    L.append("per-mode stream accuracy (DIV8 has one stream per mode; HOM8 cot aggregates its 8 streams):")
    L.append(f"  {'mode':>8} {'HOM8 acc%':>10} {'HOM8 n':>7} {'DIV8 acc%':>10} {'DIV8 n':>7}")
    for m in STRATEGIES:
        hm = r["hom_mode_acc"].get(m)
        dm = r["div_mode_acc"].get(m)
        hs = f"{hm['acc']:.1f}" if hm else "-"
        hn = f"{hm['correct']}/{hm['total']}" if hm else "-"
        ds = f"{dm['acc']:.1f}" if dm else "-"
        dn = f"{dm['correct']}/{dm['total']}" if dm else "-"
        L.append(f"  {m:>8} {hs:>10} {hn:>7} {ds:>10} {dn:>7}")
    L.append("HOM8 per-stream-index accuracy: " +
             ", ".join(f"{k}:{v:.0f}" for k, v in sorted(r["hom_index_acc"].items())))

    L.append("")
    L.append("finish reasons and length per mode (cap = truncated at max_trace):")
    for name, fs in (("HOM8", r["hom_finish"]), ("DIV8", r["div_finish"])):
        for m in sorted(fs):
            v = fs[m]
            L.append(f"  {name} {m:>7}: n={v['n']} finish={v['finish']} "
                     f"gen mean {v['gen_mean']:.1f}, max {v['gen_max']}")

    L.append("")
    L.append("pairwise DIV8 answer agreement (equal answers among both-answered problems,")
    L.append("Jaccard of (id,answer) sets, and byte-identical traces):")
    for (ma, mb), v in sorted(r["agreement"].items()):
        L.append(f"  {ma:>7} vs {mb:<7}: equal {v['agree']}/{v['both']} "
                 f"({100 * v['agree_frac']:.1f}%), Jaccard {v['jaccard']:.3f}, "
                 f"same trace {v['text_same']}/{v['both']}")

    L.append("")
    L.append(f"first-K-agree (K={K_AGREE}, streams in done_ms order):")
    L.append(f"  {'arm':>6} {'correct':>8} {'acc%':>7} {'reached':>8} "
             f"{'median fk ms':>13} {'median all ms':>14}")
    for name, s in (("HOM8", r["hom_fks"]), ("DIV8", r["div_fks"])):
        L.append(f"  {name:>6} {s['correct']:>8} {s['acc']:>7.1f} {s['reached']:>8} "
                 f"{s['median_fk_ms']:>13.1f} {s['median_all_ms']:>14.1f}")

    L.append("")
    L.append("batch-composition check (streams sharing both index and mode, i.e. stream 0 cot;")
    L.append("same prompt+seed, different co-resident streams):")
    st, sa, tot = r["overlap"]
    if tot:
        L.append(f"  identical sampled text {st}/{tot} ({M.pct(st, tot):.1f}%), "
                 f"identical answer {sa}/{tot} ({M.pct(sa, tot):.1f}%)")
    else:
        L.append("  n/a (no shared index+mode streams)")

    m = r["mcnemar"]
    L.append("")
    L.append("paired tests, DIV8 vs HOM8 (primary = tie-aware paired bootstrap, "
             f"{TIE_BOOTSTRAP_N} resamples, seed {TIE_BOOTSTRAP_SEED}, 95% CI):")
    L.append(f"  plurality point: delta {m['delta']:+.1f} pts, "
             f"tieCI [{m['lo_tie']:+.1f}, {m['hi_tie']:+.1f}], "
             f"pointCI [{m['lo']:+.1f}, {m['hi']:+.1f}], b={m['b']} c={m['c']} p={m['p']:.4f}")
    f = r["fk_test"]
    L.append(f"  first-K-agree  : delta {f['delta']:+.1f} pts, "
             f"pointCI [{f['lo']:+.1f}, {f['hi']:+.1f}], b={f['b']} c={f['c']} p={f['p']:.4f}")

    L.append("")
    L.append(f"verdict: {r['verdict']} (PASS: DIV8 >= HOM8 + 5 pts and tieCI excludes 0; "
             f"DEAD: DIV8 <= HOM8 + 1)")
    L.append("")
    return "\n".join(L)


# --------------------------------------------------------------------- main ----


def parse_run_arg(s):
    parts = s.split(",")
    if len(parts) != 3 or not all(parts):
        raise SystemExit(f"--run expects TAG,hom8_tsv,div8_tsv, got {s!r}")
    return parts[0], parts[1], parts[2]


def build_groups(args):
    """Return (legacy_pairs, multi_groups).

    multi_groups maps tag -> list of (label, path). Accepted forms:
      * --label L --tsv P (repeatable; --tag names the group)
      * --run L,P (repeatable; --tag names the group)
      * --run TAG,path1,path2[,path3...] (labels from --label when they match,
        else ARM1, ARM2, ...)
    legacy_pairs is the original TAG,hom8_tsv,div8_tsv form.
    """
    legacy_pairs = []
    multi = {}
    if args.tsv:
        if len(args.label) != len(args.tsv):
            raise SystemExit("--tsv and --label must be given in equal numbers")
        tag = args.tag or "run"
        multi[tag] = list(zip(args.label, args.tsv))
    for s in args.run:
        parts = s.split(",")
        if len(parts) == 3 and all(parts):
            legacy_pairs.append((parts[0], parts[1], parts[2]))
        elif len(parts) == 2 and all(parts):
            tag = args.tag or "run"
            multi.setdefault(tag, []).append((parts[0], parts[1]))
        elif len(parts) >= 3 and all(parts):
            tag = parts[0]
            paths = parts[1:]
            if len(args.label) == len(paths):
                labels = list(args.label)
            else:
                labels = [f"ARM{i + 1}" for i in range(len(paths))]
            multi[tag] = list(zip(labels, paths))
        else:
            raise SystemExit(f"--run expects TAG,HOM,DIV or LABEL,PATH or "
                             f"TAG,path1,path2,...; got {s!r}")
    if args.hom and args.div:
        legacy_pairs.append((args.tag or "run", args.hom, args.div))
    return legacy_pairs, multi


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", default=[],
                    help="TAG,hom8_tsv,div8_tsv | LABEL,tsv | TAG,tsv1,tsv2,... (repeatable)")
    ap.add_argument("--label", action="append", default=[],
                    help="arm label, paired with --tsv (repeatable)")
    ap.add_argument("--tsv", action="append", default=[],
                    help="arm TSV path, paired with --label (repeatable)")
    ap.add_argument("--tag", default="run", help="group tag for --label/--tsv or --hom/--div")
    ap.add_argument("--hom", default="")
    ap.add_argument("--div", default="")
    ap.add_argument("--gold-file", default=DEFAULT_GOLD)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    pairs, multi_groups = build_groups(args)
    if not pairs and not multi_groups:
        ap.error("give --label/--tsv, --run, or --hom/--div")

    gold_raw = M.load_gold(args.gold_file)
    out_lines = [
        "RecycleLM strategy probe (legacy HOM8 vs DIV8 and N-arm exemplar groups)",
        f"gold: {os.path.abspath(args.gold_file)}",
        "source: harness/engine/modality.cpp; analyzer: harness/analyze_strategy_vote.py",
        f"modes: HOM8 = 8x cot; DIV8 = {'/'.join(STRATEGIES)}",
        "",
    ]

    for tag, hom, div in pairs:
        r = analyze_pair(tag, hom, div, gold_raw)
        csv_path = os.path.join(RESULTS, f"strategy_probe_{tag}.csv")
        write_csv(r, csv_path)
        out_lines.append(format_pair(r))
        out_lines.append(f"[written] {csv_path}")
        out_lines.append("")

    for tag, specs in multi_groups.items():
        r = analyze_multi(tag, specs, gold_raw)
        csv_path = os.path.join(RESULTS, f"strategy_probe_{tag}.csv")
        write_multi_csv(r, csv_path)
        out_lines.append(format_multi(r))
        out_lines.append(f"[written] {csv_path}")
        out_lines.append("")

    text = "\n".join(out_lines).rstrip() + "\n"
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(text)
    sys.stdout.write(text)
    sys.stdout.write(f"\n[written] {args.out}\n")


if __name__ == "__main__":
    main()
