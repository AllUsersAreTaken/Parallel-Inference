"""Per-template strict-artifact table for the linter policy run.

Reads an enriched linter CSV (analyze_linter.py --enrich-out) and prints
trigger counts plus literal_artifact_correct rates per template and arm.
"""
import argparse
import csv
from collections import Counter, defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    arms = ["control", "flagonly", "full_userturn", "plain_retry",
            "partial_userturn", "partial_comment", "full_comment"]
    trig = defaultdict(set)
    total = defaultdict(set)
    hit = Counter()
    den = Counter()
    seen = set()

    with open(args.csv, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            tmpl = r.get("template", "") or ""
            arm = r.get("arm", "")
            if arm == "control" and r.get("type") == "temptation":
                total[tmpl].add(r["id"])
                if r.get("first_import") == "1":
                    trig[tmpl].add(r["id"])
            key = (arm, r["id"], r["seed"])
            if key in seen:
                continue
            seen.add(key)
            if r.get("first_import") != "1":
                continue
            den[(arm, tmpl)] += 1
            if r.get("literal_correct") == "1":
                hit[(arm, tmpl)] += 1

    lines = []
    lines.append("Per-template strict artifact success (literal whole-artifact)")
    lines.append("Triggered rows only. n = triggered rows pooled over seeds.")
    lines.append("")
    lines.append("%-16s %5s %6s %14s %14s %14s %14s"
                 % ("template", "trig", "cases", "control", "flagonly",
                    "full_userturn", "plain_retry"))
    lines.append("-" * 88)
    templates = sorted(total)
    for tmpl in templates:
        def cell(arm):
            n = den[(arm, tmpl)]
            return "%d/%d" % (hit[(arm, tmpl)], n) if n else "-"
        lines.append("%-16s %5d %6d %14s %14s %14s %14s"
                     % (tmpl, len(trig[tmpl]), len(total[tmpl]),
                        cell("control"), cell("flagonly"),
                        cell("full_userturn"), cell("plain_retry")))
    lines.append("")
    lines.append("All triggered rows, pooled")
    for arm in arms:
        k = sum(hit[(arm, t)] for t in templates)
        n = sum(den[(arm, t)] for t in templates)
        lines.append("  %-18s %3d/%3d  %6.1f%%"
                     % (arm, k, n, 100.0 * k / n if n else float("nan")))
    lines.append("")
    lines.append("full_userturn vs plain_retry per template (delta pp)")
    for tmpl in templates:
        n1 = den[("full_userturn", tmpl)]
        n2 = den[("plain_retry", tmpl)]
        if not n1 or not n2:
            continue
        r1 = 100.0 * hit[("full_userturn", tmpl)] / n1
        r2 = 100.0 * hit[("plain_retry", tmpl)] / n2
        lines.append("  %-16s full=%5.1f%% (n=%d)  plain=%5.1f%% (n=%d)  "
                     "delta=%+6.1fpp" % (tmpl, r1, n1, r2, n2, r1 - r2))

    text = "\n".join(lines) + "\n"
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    print(text, end="")
    print("wrote " + args.out)


if __name__ == "__main__":
    main()
