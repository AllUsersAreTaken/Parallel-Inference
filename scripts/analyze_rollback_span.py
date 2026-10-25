r"""Span-rollback benchmark summary (stdlib only).

Reads results/rollback_span_bench.csv (one row per case x arm) and reports:
  * per-arm detection (fatal), quote mapping, rollback rate, and outcome rates
  * FP on clean cases (thinker fatal rate)
  * latency (total_ms, rm_us)
  * paired norollback vs spanrollback comparison on contradictions
  * two verbatim examples

Usage:
    python scripts/analyze_rollback_span.py
    python scripts/analyze_rollback_span.py --csv results/rollback_span_bench.csv
"""
import argparse
import csv
import math
import os
import statistics
import sys
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
DEFAULT_CSV = os.path.join(RESULTS, "rollback_span_bench.csv")
DEFAULT_OUT = os.path.join(RESULTS, "rollback_span_summary.txt")

ARM_ORDER = ["control", "norollback", "tailrollback", "spanrollback"]
INT_FIELDS = ["thinker_fatal", "quote_found", "char_start", "keep_token",
              "keep_pos", "rolled", "bad_sentence_present", "both_dates",
              "explicit", "invented_years", "rm_ok"]
FLOAT_FIELDS = ["actor_ms", "think_ms", "rm_us", "cont_ms", "total_ms"]


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


def short(text, n=300):
    text = text.replace("\\n", " ")
    return text if len(text) <= n else text[:n] + "..."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        raise SystemExit(f"missing input: {args.csv}")
    rows = load_rows(args.csv)
    usable = [r for r in rows if r.get("note") != "skip_short_actor"]
    skipped = [r for r in rows if r.get("note") == "skip_short_actor"]
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

    out("RecycleLM span-targeted KV rollback benchmark")
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

    hdr = (f"{'arm':<13} {'n':>4} {'fatalC%':>8} {'FPc%':>6} {'quoted%':>8} "
           f"{'rolled%':>8} {'explC%':>7} {'badC%':>6} {'bothC%':>7} "
           f"{'inventC%':>9} {'med_ms':>8} {'p95_ms':>8} {'med_rm_us':>9}")
    out("Per-arm metrics")
    out(hdr)
    out("-" * len(hdr))

    per_arm = {}
    for arm in arms:
        rs = [r for r in usable if r["arm"] == arm]
        rc = [r for r in rs if r["type"] == "contradiction"]
        rk = [r for r in rs if r["type"] == "clean"]
        fatal_c = sum(1 for r in rc if r["thinker_fatal"] == 1)
        fp_k = sum(1 for r in rk if r["thinker_fatal"] == 1)
        fatal_rows = [r for r in rc if r["thinker_fatal"] == 1]
        quoted = sum(1 for r in fatal_rows if r["quote_found"] == 1)
        rolled = sum(1 for r in rc if r["rolled"] == 1)
        expl = sum(1 for r in rc if r["explicit"] == 1)
        bad = sum(1 for r in rc if r["bad_sentence_present"] == 1)
        bad_full = sum(1 for r in rc if r["quote"] and r["quote"] in r["final_text"])
        both = sum(1 for r in rc if r["both_dates"] == 1)
        inv = sum(1 for r in rc if r["invented_years"] > 0)
        totals = [r["total_ms"] for r in rs if r["total_ms"] > 0]
        rms = [r["rm_us"] for r in rs if r["rm_us"] > 0]
        per_arm[arm] = dict(rows=rs, rc=rc, rk=rk, fatal_c=fatal_c, fp_k=fp_k,
                            quoted=quoted, rolled=rolled, expl=expl, bad=bad,
                            bad_full=bad_full, both=both, inv=inv)
        out(f"{arm:<13} {len(rs):>4} {pct(fatal_c, len(rc)):>8.1f} "
            f"{pct(fp_k, len(rk)):>6.1f} {pct(quoted, len(fatal_rows)):>8.1f} "
            f"{pct(rolled, len(rc)):>8.1f} {pct(expl, len(rc)):>7.1f} "
            f"{pct(bad, len(rc)):>6.1f} {pct(both, len(rc)):>7.1f} "
            f"{pct(inv, len(rc)):>9.1f} {med(totals):>8.1f} {p95(totals):>8.1f} "
            f"{med(rms):>9.1f}")
    out("")
    out("fatalC% = thinker replied [FATAL on contradiction cases (detection)")
    out("FPc%    = thinker replied [FATAL on clean cases (false positive)")
    out("quoted% = quote located verbatim in the actor text, of fatal rows")
    out("rolled% = span erased, of contradiction rows")
    out("badC%   = offending quote still present in final text, contradictions")
    out("bothC%  = y1 and y2 both in final text, contradictions")
    out("inventC%= final text has a 4-digit year outside {y1,y2}, contradictions")
    out("")
    out("Full quote (not the 40-char fallback) still in final text, contradictions:")
    for arm in arms:
        pa = per_arm[arm]
        out(f"  {arm:<13} {pa['bad_full']}/{len(pa['rc'])} "
            f"({pct(pa['bad_full'], len(pa['rc'])):.1f}%)")
    out("")
    out("Note: the thinker runs once per case and its verdict is recorded on every")
    out("arm row. fatalC%/FPc% are therefore shared across arms; the control arm")
    out("does not consume the audit text.")
    out("")

    # -------------------------------------------------------- surgery shape
    span_rows = per_arm.get("spanrollback")
    if span_rows:
        rolled = [r for r in span_rows["rows"] if r["rolled"] == 1]
        keep0 = [r for r in rolled if r["keep_token"] == 0]
        targeted = [r for r in rolled if r["keep_token"] > 0]
        rc0 = [r for r in keep0 if r["type"] == "contradiction"]
        rct = [r for r in targeted if r["type"] == "contradiction"]
        out("Span surgery shape")
        out(f"  span rows: {len(span_rows['rows'])}  rolled: {len(rolled)}  "
            f"keep_token==0 (full actor erasure): {len(keep0)}  "
            f"targeted (keep_token>0): {len(targeted)}")
        out(f"  seq_rm returned true: {sum(1 for r in rolled if r['rm_ok'] == 1)}/"
            f"{len(rolled)}")
        out(f"  contradictions rolled: {len(rolled) - len([r for r in rolled if r['type'] == 'clean'])}  "
            f"of which keep_token==0: {len(rc0)}  targeted: {len(rct)}")
        if rc0:
            out(f"  median dropped chars (full-erasure contradictions): "
                f"{med([len(r['dropped_text']) for r in rc0]):.0f}")
        for r in targeted:
            out(f"  targeted: {r['id']} type={r['type']} char_start={r['char_start']} "
                f"keep_token={r['keep_token']} bad_after={r['bad_sentence_present']}")
        out("")

    # ---------------------------------------------------------- paired deltas
    span = per_arm.get("spanrollback")
    no = per_arm.get("norollback")
    if span and no:
        by_no = {r["id"]: r for r in no["rc"]}
        paired = [(r, by_no[r["id"]]) for r in span["rc"]
                  if r["rolled"] == 1 and r["id"] in by_no]
        erased = [(s, n) for s, n in paired
                  if s["bad_sentence_present"] == 0 and n["bad_sentence_present"] == 1]
        regressed = [(s, n) for s, n in paired
                     if s["bad_sentence_present"] == 1 and n["bad_sentence_present"] == 0]
        still = [(s, n) for s, n in paired if s["bad_sentence_present"] == 1]
        diff = [(s, n) for s, n in paired if s["final_text"] != n["final_text"]]
        expl_span = sum(1 for s, _ in paired if s["explicit"] == 1)
        expl_no = sum(1 for _, n in paired if n["explicit"] == 1)
        out("Paired span vs norollback (contradictions where span rolled)")
        out(f"  paired cases: {len(paired)}")
        out(f"  bad sentence erased by span (norollback had it): {len(erased)}/{len(paired)}")
        out(f"  bad sentence still present after span: {len(still)}/{len(paired)}")
        full_absent = sum(1 for s, _ in paired
                          if s["quote"] and s["quote"] not in s["final_text"])
        prefix_only = sum(1 for s, _ in paired
                          if s["bad_sentence_present"] == 1 and s["quote"]
                          and s["quote"] not in s["final_text"])
        out(f"  full quote absent from span final: {full_absent}/{len(paired)}; "
            f"of the {len(still)} still-present, {prefix_only} are 40-char-prefix "
            f"matches only")
        out(f"  regressions (span has it, norollback did not): {len(regressed)}")
        out(f"  explicit continuation: span {pct(expl_span, len(paired)):.1f}% "
            f"vs norollback {pct(expl_no, len(paired)):.1f}%")
        out(f"  final text differs: {len(diff)}/{len(paired)}")
        out("  (the correction instruction embeds the quote; a continuation that")
        out("   restates it counts as bad_sentence_present=1)")
        out("")

        rc_all = {r["id"] for r in span["rc"]}
        unpaired = [r for r in span["rc"] if r["rolled"] == 0]
        if unpaired:
            out(f"Contradiction cases not rolled by span: {len(unpaired)}")
            for r in unpaired:
                out(f"  {r['id']}: fatal={r['thinker_fatal']} quote_found={r['quote_found']} "
                    f"quote=\"{short(r['quote'], 110)}\"")
            out("")

        clean_rolled = [r for r in span["rk"] if r["rolled"] == 1]
        clean_bad_after = [r for r in clean_rolled if r["bad_sentence_present"] == 1]
        out(f"Clean cases span-rolled (false-positive surgeries): "
            f"{len(clean_rolled)}/{len(span['rk'])}")
        if clean_rolled:
            out(f"  of those, quote reappeared in final: {len(clean_bad_after)}")
        out("")

    # ------------------------------------------------------------- examples
    def show(r, title):
        out(title)
        out(f"  id={r['id']} type={r['type']} template={r['template']} arm={r['arm']}")
        out(f"  actor: {short(r['actor_text'], 420)}")
        out(f"  thinker: {short(r['thinker_text'], 260)}")
        out(f"  fatal={r['thinker_fatal']} quote_found={r['quote_found']} "
            f"char_start={r['char_start']} keep_token={r['keep_token']} "
            f"keep_pos={r['keep_pos']} rolled={r['rolled']} rm_ok={r['rm_ok']}")
        out(f"  quote: {short(r['quote'], 260)}")
        retained = ""
        if r["rolled"] == 1 and r["dropped_text"] and r["actor_text"].endswith(r["dropped_text"]):
            retained = r["actor_text"][:-len(r["dropped_text"])]
        out(f"  retained: {short(retained, 420)}")
        out(f"  dropped: {short(r['dropped_text'], 420)}")
        out(f"  final: {short(r['final_text'], 520)}")
        out(f"  bad_present={r['bad_sentence_present']} both_dates={r['both_dates']} "
            f"explicit={r['explicit']} invented={r['invented_years']} "
            f"total_ms={r['total_ms']:.1f} rm_us={r['rm_us']:.1f}")
        out("")

    out("Examples")
    out("-" * 64)
    if span:
        good = [r for r in span["rc"] if r["rolled"] == 1 and r["bad_sentence_present"] == 0]
        if good:
            show(good[0], "Example: contradiction, quote absent after span rollback "
                          "(keep_token=0 here: full actor erased)")
        else:
            show(span["rc"][0], "Example: contradiction, no clean span erasure found")
        fp = [r for r in span["rk"] if r["rolled"] == 1]
        if fp:
            show(fp[0], "Example: clean case flagged FATAL (false-positive surgery)")
        elif span["rc"]:
            show(span["rc"][0], "Example: contradiction (no clean false-positive rollback)")

    out("Verdict")
    out("-" * 64)
    if span and no:
        rc_span = span["rc"]
        out(f"  contradictions: {len(rc_span)}, fatal {span['fatal_c']}, "
            f"quote_found {span['quoted']}, span_rolled {span['rolled']}")
        ctrl_bad = ("%.1f%%" % pct(per_arm["control"]["bad"],
                                   len(per_arm["control"]["rc"]))
                    if "control" in per_arm else "n/a")
        out(f"  bad sentence present: control {ctrl_bad} "
            f"norollback {pct(no['bad'], len(no['rc'])):.1f}% "
            f"span {pct(span['bad'], len(rc_span)):.1f}%")
        out(f"  explicit continuation: norollback {pct(no['expl'], len(no['rc'])):.1f}% "
            f"span {pct(span['expl'], len(rc_span)):.1f}%")
        out(f"  FP on clean: {pct(span['fp_k'], len(span['rk'])):.1f}%")
        out("")

    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {os.path.relpath(args.out, ROOT)}")


if __name__ == "__main__":
    main()
