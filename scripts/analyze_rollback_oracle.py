r"""Oracle-span KV rollback benchmark summary (stdlib only).

Reads results/rollback_oracle_bench.csv (one row per case x arm) and reports:
  * per-arm detection-free span placement, rollback rate, outcome rates
  * latency (total_ms, rm_us) and seq_rm shape checks
  * two verbatim contradiction examples
  * paired oracle_span vs oracle_norollback on partial-span contradictions
  * McNemar exact test on explicit / bad_cont

Usage:
    python scripts/analyze_rollback_oracle.py
    python scripts/analyze_rollback_oracle.py --csv results/rollback_oracle_bench.csv
"""
import argparse
import csv
import math
import os
import statistics
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
DEFAULT_CSV = os.path.join(RESULTS, "rollback_oracle_bench.csv")
DEFAULT_OUT = os.path.join(RESULTS, "rollback_oracle_summary.txt")

ARM_ORDER = ["control", "oracle_norollback", "oracle_span"]
INT_FIELDS = ["oracle_found", "char_start", "keep_token", "keep_pos", "partial",
              "rolled", "rm_ok", "pos_max_before", "pos_max_after", "bad_cont",
              "explicit", "both_dates", "invented_years"]
FLOAT_FIELDS = ["rm_us", "actor_ms", "cont_ms", "total_ms"]


def load_rows(path):
    rows = []
    with open(path, encoding="utf-8", errors="replace", newline="") as fh:
        for raw in csv.DictReader(fh):
            r = dict(raw)
            for k in INT_FIELDS:
                try:
                    r[k] = int(r.get(k, "-1") or -1)
                except ValueError:
                    r[k] = -1
            for k in FLOAT_FIELDS:
                try:
                    r[k] = float(r.get(k, "0") or 0.0)
                except ValueError:
                    r[k] = 0.0
            rows.append(r)
    return rows


def pct(num, den):
    return 100.0 * num / den if den else float("nan")


def p95(values):
    if not values:
        return float("nan")
    v = sorted(values)
    if len(v) == 1:
        return v[0]
    k = int(math.ceil(0.95 * (len(v) - 1)))
    return v[min(k, len(v) - 1)]


def med(values):
    return statistics.median(values) if values else float("nan")


def short(text, n=420):
    text = text.replace("\\n", " ").replace("\n", " ")
    return text if len(text) <= n else text[:n] + "..."


def sentence_at(text, start):
    """Sentence beginning at start, split on '. ', '! ', '? ' and newline."""
    if start < 0 or start >= len(text):
        return ""
    i = start
    while i < len(text):
        if text[i] in ".!?" and i + 1 < len(text) and text[i + 1] == " ":
            return text[start:i + 1]
        if text[i] == "\n":
            return text[start:i]
        i += 1
    return text[start:]


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def is_skipped(r):
    return (r["cont_text"] == "" and r["cont_ms"] == 0.0 and
            r["rolled"] == 0 and r["rm_ok"] == -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        raise SystemExit(f"missing input: {args.csv}")
    rows = load_rows(args.csv)
    usable = [r for r in rows if not is_skipped(r)]
    skipped = [r for r in rows if is_skipped(r)]
    if not usable:
        raise SystemExit(f"no usable rows in {args.csv}")

    arms = [a for a in ARM_ORDER if any(r["arm"] == a for r in usable)]
    cases = OrderedDict()
    for r in usable:
        cases.setdefault(r["id"], r["type"])

    lines = []

    def out(s=""):
        lines.append(s)
        print(s)

    out("RecycleLM oracle-span KV rollback benchmark")
    out("=" * 64)
    out(f"input: {os.path.relpath(args.csv, ROOT)}")
    meta = args.csv + ".meta.txt"
    if os.path.exists(meta):
        with open(meta, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out(f"meta:  {line}")
    out(f"rows: {len(rows)}  usable: {len(usable)}  skipped_short_actor: {len(skipped)}")
    out("")

    n_c = sum(1 for t in cases.values() if t == "contradiction")
    n_k = sum(1 for t in cases.values() if t == "clean")
    out(f"cases: {len(cases)}  contradictions: {n_c}  clean: {n_k}")
    out("")

    hdr = (f"{'arm':<18} {'n':>4} {'conN':>5} {'found%':>7} {'part%':>6} "
           f"{'rolled%':>8} {'explC%':>7} {'badC%':>6} {'bothC%':>7} "
           f"{'inventC%':>9} {'med_ms':>8} {'p95_ms':>8} {'med_rm_us':>9}")
    out("Per-arm metrics (contradiction rates over contradiction rows)")
    out(hdr)
    out("-" * len(hdr))

    per_arm = {}
    for arm in arms:
        rs = [r for r in usable if r["arm"] == arm]
        rc = [r for r in rs if r["type"] == "contradiction"]
        found = [r for r in rc if r["oracle_found"] == 1]
        partial = [r for r in found if r["partial"] == 1]
        rolled = [r for r in rc if r["rolled"] == 1]
        expl = [r for r in rc if r["explicit"] == 1]
        bad = [r for r in rc if r["bad_cont"] == 1]
        both = [r for r in rc if r["both_dates"] == 1]
        inv = [r for r in rc if r["invented_years"] > 0]
        totals = [r["total_ms"] for r in rs if r["total_ms"] > 0]
        rms = [r["rm_us"] for r in rolled if r["rm_us"] > 0]
        per_arm[arm] = dict(rows=rs, rc=rc, found=found, partial=partial,
                            rolled=rolled, expl=expl, bad=bad, both=both, inv=inv)
        out(f"{arm:<18} {len(rs):>4} {len(rc):>5} "
            f"{pct(len(found), len(rc)):>7.1f} {pct(len(partial), len(found)):>6.1f} "
            f"{pct(len(rolled), len(rc)):>8.1f} {pct(len(expl), len(rc)):>7.1f} "
            f"{pct(len(bad), len(rc)):>6.1f} {pct(len(both), len(rc)):>7.1f} "
            f"{pct(len(inv), len(rc)):>9.1f} {med(totals):>8.1f} "
            f"{p95(totals):>8.1f} {med(rms):>9.1f}")
    out("")
    out("found%  = contradiction rows with a deterministic oracle span")
    out("part%   = keep_token > 0, of found contradictions (shared by all arms)")
    out("rolled% = contradiction rows where seq_rm ran (oracle_span only)")
    out("explC%  = continuation contains an explicit correction marker")
    out("badC%   = first 40 chars of the dropped sentence in the continuation")
    out("bothC%  = y1 and y2 both present in final_text")
    out("inventC%= final_text has a 4-digit year outside {y1,y2}")
    out("med_ms  = median total_ms over all usable rows in the arm")
    out("med_rm  = median rm_us over rolled rows")
    out("")

    # ------------------------------------------------------------ shape
    span = per_arm.get("oracle_span")
    if span:
        rolled = span["rolled"]
        ok = [r for r in rolled if r["rm_ok"] == 1]
        pmok = [r for r in rolled if r["pos_max_after"] == r["keep_pos"] - 1]
        clean_rolled = [r for r in span["rows"] if r["type"] == "clean" and r["rolled"] == 1]
        out("Oracle surgery shape")
        out(f"  contradictions rolled: {len(rolled)}/{len(span['rc'])}  "
            f"seq_rm true: {len(ok)}/{len(rolled)}  "
            f"pos_max_after == keep_pos-1: {len(pmok)}/{len(rolled)}")
        out(f"  partial spans (keep_token>0): {len(span['partial'])}/{len(span['found'])} "
            f"found contradictions "
            f"({pct(len(span['partial']), len(span['found'])):.1f}%)")
        out(f"  full erasures (keep_token==0): "
            f"{len([r for r in span['found'] if r['partial'] == 0])}")
        out(f"  clean cases rolled: {len(clean_rolled)}/{len([r for r in span['rows'] if r['type'] == 'clean'])}")
        dropped_chars = [len(r["dropped_text"]) for r in rolled if r["dropped_text"]]
        out(f"  median dropped span chars: {med(dropped_chars):.0f}")
        out("")

    # ---------------------------------------------------------- paired
    span = per_arm.get("oracle_span")
    no = per_arm.get("oracle_norollback")
    if span and no:
        by_no = {r["id"]: r for r in no["rc"]}
        paired = [(r, by_no[r["id"]]) for r in span["rc"]
                  if r["rolled"] == 1 and r["partial"] == 1 and r["id"] in by_no]
        out("Paired oracle_span vs oracle_norollback")
        out(f"  contradiction cases rolled with partial=1: {len(paired)}")
        if paired:
            def pair_counts(key):
                b = sum(1 for s, n in paired if s[key] == 1 and n[key] == 0)
                c = sum(1 for s, n in paired if s[key] == 0 and n[key] == 1)
                both = sum(1 for s, n in paired if s[key] == 1 and n[key] == 1)
                neither = sum(1 for s, n in paired if s[key] == 0 and n[key] == 0)
                return b, c, both, neither

            for key, label in (("explicit", "explicit"), ("bad_cont", "bad_cont")):
                b, c, both, neither = pair_counts(key)
                s_rate = pct(both + b, len(paired))
                n_rate = pct(both + c, len(paired))
                p = mcnemar_exact(b, c)
                out(f"  {label}: span {both + b}/{len(paired)} ({s_rate:.1f}%) "
                    f"vs norollback {both + c}/{len(paired)} ({n_rate:.1f}%); "
                    f"discordant span-only={b} norollback-only={c}; "
                    f"McNemar exact p={p:.4f}")

            erased = [(s, n) for s, n in paired
                      if s["bad_cont"] == 0 and n["bad_cont"] == 1]
            retained = [(s, n) for s, n in paired if s["bad_cont"] == 1]
            appeared = [(s, n) for s, n in paired
                        if s["bad_cont"] == 1 and n["bad_cont"] == 0]
            span_final_has = sum(
                1 for s, _ in paired
                if sentence_at(s["actor_text"], s["char_start"]) and
                sentence_at(s["actor_text"], s["char_start"]) in s["final_text"])
            no_final_has = sum(
                1 for _, n in paired
                if sentence_at(n["actor_text"], n["char_start"]) and
                sentence_at(n["actor_text"], n["char_start"]) in n["final_text"])
            out(f"  offending sentence gone from span continuation but present in "
                f"norollback: {len(erased)}/{len(paired)}")
            out(f"  offending sentence still in span continuation: "
                f"{len(retained)}/{len(paired)}")
            out(f"  offending sentence only in span continuation: {len(appeared)}")
            out(f"  offending sentence in final_text: span {span_final_has}/{len(paired)} "
                f"vs norollback {no_final_has}/{len(paired)}")
            out("")
        else:
            out("  no partial-span contradictions rolled")
            out("")

    # ---------------------------------------------------------- examples
    def show(r, title):
        out(title)
        out(f"  id={r['id']} type={r['type']} template={r['template']} arm={r['arm']}")
        out(f"  actor: {short(r['actor_text'])}")
        sent = sentence_at(r["actor_text"], r["char_start"]) if r["char_start"] >= 0 else ""
        out(f"  dropped sentence: {short(sent)}")
        out(f"  oracle_found={r['oracle_found']} char_start={r['char_start']} "
            f"keep_token={r['keep_token']} keep_pos={r['keep_pos']} "
            f"partial={r['partial']} rolled={r['rolled']}")
        out(f"  dropped span: {short(r['dropped_text'])}")
        out(f"  final: {short(r['final_text'], 640)}")
        out(f"  explicit={r['explicit']} bad_cont={r['bad_cont']} "
            f"both_dates={r['both_dates']} invented_years={r['invented_years']} "
            f"total_ms={r['total_ms']:.1f} rm_us={r['rm_us']:.1f}")
        out("")

    out("Examples")
    out("-" * 64)
    if span:
        cand = [r for r in span["rc"] if r["rolled"] == 1 and r["partial"] == 1]
        good = [r for r in cand if r["bad_cont"] == 0 and r["explicit"] == 1]
        bad = [r for r in cand if r["bad_cont"] == 1]
        if cand:
            show(good[0] if good else cand[0],
                 "Example 1: partial span, explicit correction, offending "
                 "sentence absent from continuation")
        elif span["rc"]:
            show(span["rc"][0], "Example 1: contradiction, no partial span rolled")
        rest = [r for r in span["rc"]
                if r is not (good[0] if good else (cand[0] if cand else None))]
        if bad:
            show(bad[0], "Example 2: partial span, offending sentence still in "
                         "continuation")
        elif len(cand) > 1:
            show(cand[1], "Example 2: partial span, offending sentence absent")
        elif rest:
            show(rest[0], "Example 2: contradiction")

    # --------------------------------------------------------- question
    out("Primary question")
    out("-" * 64)
    if span and no:
        by_no = {x["id"]: x for x in no["rc"]}
        paired = [(r, by_no[r["id"]]) for r in span["rc"]
                  if r["rolled"] == 1 and r["partial"] == 1 and r["id"] in by_no]
        n_partial = len(span["partial"])
        out(f"  partial-span contradictions: {n_partial}/{len(span['found'])} found "
            f"({pct(n_partial, len(span['found'])):.1f}%)")
        if paired:
            erased = sum(1 for s, n in paired if s["bad_cont"] == 0 and n["bad_cont"] == 1)
            still = sum(1 for s, n in paired if s["bad_cont"] == 1)
            absent_both = sum(1 for s, n in paired
                              if s["bad_cont"] == 0 and n["bad_cont"] == 0)
            b, c, both, _ = (sum(1 for s, n in paired if s["explicit"] == 1 and n["explicit"] == 0),
                             sum(1 for s, n in paired if s["explicit"] == 0 and n["explicit"] == 1),
                             sum(1 for s, n in paired if s["explicit"] == 1 and n["explicit"] == 1),
                             0)
            p = mcnemar_exact(b, c)
            out(f"  paired partial-span cases: {len(paired)}")
            span_final_has = sum(
                1 for s, _ in paired
                if sentence_at(s["actor_text"], s["char_start"]) and
                sentence_at(s["actor_text"], s["char_start"]) in s["final_text"])
            no_final_has = sum(
                1 for _, n in paired
                if sentence_at(n["actor_text"], n["char_start"]) and
                sentence_at(n["actor_text"], n["char_start"]) in n["final_text"])
            out(f"  offending sentence erased vs norollback: {erased}/{len(paired)} "
                f"({pct(erased, len(paired)):.1f}%)")
            out(f"  still present after span: {still}/{len(paired)} "
                f"({pct(still, len(paired)):.1f}%)")
            out(f"  absent in both arms: {absent_both}/{len(paired)}")
            out(f"  offending sentence in final_text: span {span_final_has}/"
                f"{len(paired)} vs norollback {no_final_has}/{len(paired)}")
            out(f"  explicit correction: span {both + b}/{len(paired)} "
                f"({pct(both + b, len(paired)):.1f}%) vs norollback "
                f"{both + c}/{len(paired)} ({pct(both + c, len(paired)):.1f}%), "
                f"McNemar p={p:.4f}")
            if still == 0 and erased > 0:
                verdict = (f"the span rollback erased the offending sentence in "
                           f"{erased}/{len(paired)} paired cases where append-only "
                           f"reproduced it, and retained it in none")
            elif still > 0:
                verdict = (f"the span rollback left the offending sentence in "
                           f"{still}/{len(paired)} continuations")
            elif span_final_has < no_final_has:
                verdict = (f"the span rollback removed the offending sentence from "
                           f"final_text in {no_final_has - span_final_has}/"
                           f"{len(paired)} cases (span {span_final_has}/"
                           f"{len(paired)} vs norollback {no_final_has}/"
                           f"{len(paired)}), but neither continuation reproduced "
                           f"it and the explicit rate was unchanged at "
                           f"{pct(both + b, len(paired)):.1f}%")
            else:
                verdict = (f"neither arm reproduced the offending sentence in any "
                           f"of the {len(paired)} paired continuations, so this "
                           f"run shows no erasure advantage for the span rollback")
            out(f"  verdict from this run: {verdict}.")
        else:
            out("  no paired partial-span cases; primary question not testable here")
        out("")

    out("Notes")
    out("  - The 3B thinker is not run in oracle mode; span placement is")
    out("    deterministic (sentence containing y2, else event-2 keyword).")
    out("  - The correction instruction quotes neither the offending sentence nor")
    out("    the dates, so bad_cont measures reproduction, not instruction leakage.")
    out("  - explicit, bad_cont scan cont_text only; both_dates and invented_years")
    out("    scan final_text.")
    out("  - One RTX 4060, 8 GB. Numbers are hardware-specific.")

    text = "\n".join(lines) + "\n"
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    print(f"wrote {os.path.relpath(args.out, ROOT)}")


if __name__ == "__main__":
    main()
