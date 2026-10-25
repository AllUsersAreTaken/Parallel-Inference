#!/usr/bin/env python3
"""Summarize results/rollback_bench.csv (stdlib only)."""

import argparse
import csv
import math
import statistics


def pct(values, p):
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p / 100.0
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return s[int(k)]
    return s[lo] * (hi - k) + s[hi] * (k - lo)


def rate(rows, key):
    if not rows:
        return float("nan")
    return 100.0 * sum(1 for r in rows if int(r[key])) / len(rows)


def fnum(x, digits=1):
    if x != x:
        return "n/a"
    return f"{x:.{digits}f}"


def one_line(text):
    return text.replace("\r", " ").replace("\n", " ")


def load_rows(path):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in ("thinker_no", "actor_tokens", "gen_actor", "keep_pos",
                  "pos_current", "rolled", "both_dates", "explicit",
                  "date_swap", "date_swap_adj", "invented_years",
                  "repeats_dropped", "rm_ok"):
            r[k] = int(r[k])
        for k in ("actor_ms", "think_ms", "cont_ms", "total_ms"):
            r[k] = float(r[k])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/rollback_bench.csv")
    ap.add_argument("--out", default="results/rollback_summary.txt")
    args = ap.parse_args()

    rows = load_rows(args.csv)
    arms = ["control", "norollback", "rollback"]
    used = [r for r in rows if r["note"] != "skip_short_actor"]
    skipped = len(rows) - len(used)

    lines = []
    def w(s=""):
        lines.append(s)

    w("RecycleLM KV-cache rollback benchmark")
    w("=" * 60)
    w(f"input: {args.csv}")
    meta = args.csv + ".meta.txt"
    try:
        with open(meta, encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if ln:
                    w("meta:  " + ln)
    except OSError:
        pass
    w(f"rows: {len(rows)}  arms: {len(used)}  skipped_short_actor: {skipped}")
    w("")

    cases = {}
    for r in used:
        cases.setdefault(r["id"], r)
    n_cases = len(cases)
    n_con = sum(1 for r in cases.values() if r["type"] == "contradiction")
    n_clean = sum(1 for r in cases.values() if r["type"] == "clean")
    w(f"cases: {n_cases}  contradictions: {n_con}  clean: {n_clean}")
    w("")

    w("Per arm (all rows, skipped cases excluded)")
    w(f"{'arm':<11}{'n':>4}{'exp%':>8}{'faith%':>8}{'fthAdj%':>9}"
      f"{'swap%':>7}{'swAdj%':>8}{'invent%':>9}{'rolled':>8}"
      f"{'med_ms':>10}{'p95_ms':>9}")
    for arm in arms:
        ar = [r for r in used if r["arm"] == arm]
        cr = [r for r in ar if r["type"] == "contradiction"]
        faithful = 100.0 * sum(1 for r in cr if int(r["explicit"]) and not int(r["date_swap"])) / len(cr) if cr else float("nan")
        faithful_adj = 100.0 * sum(1 for r in cr if int(r["explicit"]) and not int(r["date_swap_adj"])) / len(cr) if cr else float("nan")
        med = statistics.median([r["total_ms"] for r in ar]) if ar else float("nan")
        p95 = pct([r["total_ms"] for r in ar], 95)
        w(f"{arm:<11}{len(ar):>4}{rate(ar,'explicit'):>8.1f}{faithful:>8.1f}"
          f"{faithful_adj:>9.1f}{rate(ar,'date_swap'):>7.1f}"
          f"{rate(ar,'date_swap_adj'):>8.1f}"
          f"{rate(ar,'invented_years'):>9.1f}"
          f"{sum(r['rolled'] for r in ar):>8}{med:>10.1f}{p95:>9.1f}")
    w("")
    w("faith%  = explicit and not date_swap, contradictions only (spec clause rule)")
    w("fthAdj% = explicit and not date_swap_adj, contradictions only (pairing diagnostic)")
    w("invent% = final text contains a 4-digit year not in {y1,y2}")
    w("")

    w("Thinker detection (one row per case)")
    con = [r for r in cases.values() if r["type"] == "contradiction"]
    cln = [r for r in cases.values() if r["type"] == "clean"]
    con_hit = sum(1 for r in con if r["thinker_no"])
    cln_hit = sum(1 for r in cln if r["thinker_no"])
    w(f"  contradictions flagged no: {con_hit}/{len(con)} "
      f"({100.0*con_hit/len(con) if con else float('nan'):.1f}%)")
    w(f"  clean flagged no:          {cln_hit}/{len(cln)} "
      f"({100.0*cln_hit/len(cln) if cln else float('nan'):.1f}%)")
    w(f"  overall agreement:         {con_hit + (len(cln)-cln_hit)}/{len(con)+len(cln)} "
      f"({100.0*(con_hit + len(cln)-cln_hit)/(len(con)+len(cln)):.1f}%)")
    w("")

    w("Rollback surgery")
    rb = [r for r in used if r["arm"] == "rollback"]
    rolled = [r for r in rb if r["rolled"]]
    rm_ok = [r for r in rolled if r["rm_ok"] == 1]
    rep = [r for r in rolled if r["repeats_dropped"]]
    w(f"  rollback arm rows: {len(rb)}  rolled: {len(rolled)}  "
      f"seq_rm ok: {len(rm_ok)}/{len(rolled)}  dropped-text repeated: {len(rep)}/{len(rolled)}")
    if rolled:
        med_keep = statistics.median([r["keep_pos"] for r in rolled])
        w(f"  median keep_pos: {med_keep:.0f}  median actor tokens kept: "
          f"{statistics.median([r['actor_tokens'] for r in rolled]):.0f}"
          f" of {statistics.median([r['gen_actor'] for r in rolled]):.0f} generated")
    by_id_arm = {(r["id"], r["arm"]): r for r in used}
    rolled_ids = [r["id"] for r in rolled]
    same = diff = same_head = 0
    prefixes = []
    for cid in rolled_ids:
        a = by_id_arm[(cid, "norollback")]["cont_text"]
        b = by_id_arm[(cid, "rollback")]["cont_text"]
        if a == b:
            same += 1
        else:
            diff += 1
        if a[:1] == b[:1]:
            same_head += 1
        n = 0
        for x, y in zip(a, b):
            if x != y:
                break
            n += 1
        prefixes.append(n)
    w("")
    w("Rollback effect on the continuation (same case, norollback vs rollback)")
    w(f"  rolled cases: {len(rolled_ids)}  identical text: {same}  different text: {diff}")
    if prefixes:
        w(f"  same first character: {same_head}/{len(rolled_ids)}  "
          f"median common prefix: {statistics.median(prefixes):.0f} chars")
    for t in ("contradiction", "clean"):
        tids = [cid for cid in rolled_ids if by_id_arm[(cid, "rollback")]["type"] == t]
        if not tids:
            continue
        td = sum(1 for cid in tids
                 if by_id_arm[(cid, "norollback")]["cont_text"] !=
                    by_id_arm[(cid, "rollback")]["cont_text"])
        w(f"  {t}: {td}/{len(tids)} continuations differ")
    nonrolled_ids = [r["id"] for r in rb if not r["rolled"]]
    if nonrolled_ids:
        nr_same = sum(1 for cid in nonrolled_ids
                      if by_id_arm[(cid, "norollback")]["cont_text"] ==
                         by_id_arm[(cid, "rollback")]["cont_text"])
        w(f"  non-rolled cases: rollback == norollback {nr_same}/{len(nonrolled_ids)}")
    w("")

    def example(title, row):
        w(title)
        w(f"  id={row['id']} type={row['type']} template={row['template']} arm={row['arm']}")
        w(f"  actor ({row['actor_tokens']}/{row['gen_actor']} tokens): {one_line(row['actor_text'])}")
        if row["rolled"]:
            w(f"  dropped ({len(row['dropped_text'])} chars): {one_line(row['dropped_text'])}")
        w(f"  thinker (no={row['thinker_no']}): {one_line(row['thinker_text'])}")
        w(f"  continuation: {one_line(row['cont_text'])}")
        w(f"  explicit={row['explicit']} date_swap={row['date_swap']} "
          f"date_swap_adj={row['date_swap_adj']} invented_years={row['invented_years']} "
          f"total_ms={row['total_ms']:.1f}")
        w("")

    con_rolled = [r for r in rolled if r["type"] == "contradiction"]
    if con_rolled:
        example("Example: rolled contradiction", con_rolled[0])

    clean_ctrl = [r for r in used if r["type"] == "clean" and r["arm"] == "control"]
    if clean_ctrl:
        example("Example: clean control", clean_ctrl[0])

    w("Notes")
    w("  - date_swap follows the spec: a clause (split on ; , newline, '. ') that")
    w("    contains y1 with the event-2 keyword or y2 with the event-1 keyword.")
    w("    A correction that restates both years and both events in one clause is")
    w("    therefore flagged, which is why date_swap (spec) and date_swap_adj")
    w("    (nearest-keyword pairing) diverge on the correction arms.")
    w("  - explicit, date_swap and date_swap_adj scan the continuation only;")
    w("    both_dates and invented_years scan final_text.")
    w("  - Sampling: temp 0.6, top_k 20, top_p 0.95; per-case seed = 1234 + index.")
    w("    Each generation starts a fresh sampler chain at the same case seed, so")
    w("    the arms begin from the same RNG stream.")
    w("  - One RTX 4060, 8 GB. Numbers are hardware-specific.")

    text = "\n".join(lines) + "\n"
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
