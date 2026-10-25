"""Score the question-driven Reading Compiler run (stdlib only).

Reads data/_rc_qa.json (frozen questions), data/_rc_doc_numbered.txt (the
prefix/citation space), and the app/prefix_cli.exe result JSON, then reports:

  1. answer correctness (normalized exact match vs gold or an alias);
  2. citation validity and citation precision;
  3. grounded-answer rate;
  4. batch vs serial latency.

Usage:
    python harness/score_rc_qa.py [--results results/_rc_qa.json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+(?:\.[0-9]+)?[A-Za-z%]*")
LINE_RE = re.compile(r"^\[(\d+)\]\s?(.*)$")
CITE_RE = re.compile(r"\d+")

STOPWORDS = {
    "a", "an", "the", "of", "and", "or", "at", "in", "on", "to", "for",
    "with", "is", "are", "was", "were", "be", "by", "as", "that", "this",
    "it", "its", "from", "what", "which", "how", "many", "much", "does",
    "do", "did", "used", "use", "using", "recorded", "predicted", "was",
}


def compact(s: str) -> str:
    """Case/punctuation/space-insensitive comparison key."""
    s = s.lower().replace("\u00d7", "x").replace("\u2013", "-")
    return re.sub(r"[^a-z0-9]+", "", s)


def sig_tokens(s: str) -> set[str]:
    out: set[str] = set()
    for m in TOKEN_RE.finditer(s):
        t = m.group(0).lower()
        if t in STOPWORDS:
            continue
        if len(t) < 2 and not any(c.isdigit() for c in t):
            continue
        out.add(t)
    return out


def line_tokens(text: str) -> set[str]:
    return {m.group(0).lower() for m in TOKEN_RE.finditer(text)}


def parse_answer(text: str) -> tuple[str, str]:
    """Return (answer, citations) from the first A|...|... line."""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("A|"):
            ans, _, cites = line[2:].partition("|")
            return ans.strip(), cites.strip()
    return "", ""


def parse_citations(raw: str) -> list[int]:
    raw = raw.strip()
    if raw in ("", "-", "NOT_STATED"):
        return []
    return [int(x) for x in CITE_RE.findall(raw)]


def load_doc(path: Path) -> dict[int, str]:
    lines: dict[int, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        m = LINE_RE.match(raw)
        if m:
            lines[int(m.group(1))] = m.group(2)
    return lines


def stream_map(result: dict, key: str) -> dict[str, dict]:
    return {s["id"]: s for s in result.get(key, []) if "id" in s}


def pct(num: float, den: float) -> float | None:
    return None if den == 0 else 100.0 * num / den


def fmt_pct(v: float | None, digits: int = 1) -> str:
    return "n/a" if v is None else f"{v:.{digits}f}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", default=str(ROOT / "data" / "_rc_qa.json"))
    ap.add_argument("--doc", default=str(ROOT / "data" / "_rc_doc_numbered.txt"))
    ap.add_argument("--results", default=str(ROOT / "results" / "_rc_qa.json"))
    ap.add_argument("--out", default=str(ROOT / "results" / "_rc_qa_score.json"))
    args = ap.parse_args()

    qa = json.loads(Path(args.qa).read_text(encoding="utf-8"))
    doc = load_doc(Path(args.doc))
    num_lines = max(doc) if doc else 0
    res = json.loads(Path(args.results).read_text(encoding="utf-8"))

    batch = stream_map(res, "streams")
    serial = stream_map(res, "serial")
    batch_wall = float(res.get("wall_ms", 0.0))
    prefill = float(res.get("prefill_ms", 0.0))

    rows = []
    correct_n = 0
    strict_n = 0
    present_n = 0
    present_correct = 0
    absent_n = 0
    absent_correct = 0
    cited_total = 0
    cited_good = 0
    all_cited_total = 0
    all_cited_valid = 0
    by_type: dict[str, list[int]] = {}
    grounded_present = 0
    grounded_absent = 0
    serial_total = 0.0
    serial_decode_total = 0.0
    serial_missing = False

    for q in qa["questions"]:
        qid = q["id"]
        qtype = q["type"]
        st = batch.get(qid, {})
        raw = st.get("text", "")
        ans, cites_raw = parse_answer(raw)
        cites = parse_citations(cites_raw)

        gold = q["gold"]
        aliases = q.get("aliases", [])
        gold_key = compact(gold)
        strict = compact(ans) == gold_key or any(compact(a) == compact(ans) for a in aliases)
        if qtype == "absent":
            strict = compact(ans) == compact("NOT_STATED")
            correct = strict
        else:
            # Normalized match: exact gold/alias, or every significant gold
            # token present in the answer (covers a verbose but correct answer).
            correct = strict or (bool(sig_tokens(gold)) and sig_tokens(gold) <= sig_tokens(ans))

        valid = [n for n in cites if 1 <= n <= num_lines]
        invalid = [n for n in cites if not (1 <= n <= num_lines)]
        gold_toks = sig_tokens(gold)
        hit_lines = [n for n in valid if gold_toks & line_tokens(doc.get(n, ""))]
        all_cited_total += len(cites)
        all_cited_valid += len(valid)

        if qtype == "absent":
            precision = None
            grounded = correct and not cites
        else:
            present_n += 1
            if correct:
                present_correct += 1
            cited_total += len(cites)
            cited_good += len(hit_lines)
            precision = None if not cites else len(hit_lines) / len(cites)
            grounded = correct and bool(hit_lines)
            if grounded:
                grounded_present += 1

        if strict:
            strict_n += 1
        if qtype not in by_type:
            by_type[qtype] = [0, 0]
        by_type[qtype][1] += 1
        if correct:
            correct_n += 1
            by_type[qtype][0] += 1
            if qtype == "absent":
                absent_correct += 1
                if grounded:
                    grounded_absent += 1
        if qtype == "absent":
            absent_n += 1

        sr = serial.get(qid, {})
        dec = float(sr.get("completion_ms", 0.0)) if sr else 0.0
        if not sr:
            serial_missing = True
        wall = prefill + dec
        serial_total += wall
        serial_decode_total += dec

        rows.append({
            "id": qid,
            "type": qtype,
            "answer": ans,
            "citations": cites,
            "invalid_citations": invalid,
            "gold": gold,
            "strict_correct": strict,
            "correct": correct,
            "citation_precision": precision,
            "hit_lines": hit_lines,
            "grounded": grounded,
            "serial_wall_ms": wall,
            "serial_completion_ms": dec,
        })

    batch_decode_ms = max(0.0, batch_wall - prefill)
    speedup = None if batch_wall <= 0 or serial_missing else serial_total / batch_wall
    speedup_decode = (None if batch_decode_ms <= 0 or serial_missing
                      else serial_decode_total / batch_decode_ms)
    citation_precision = pct(cited_good, cited_total)
    citation_valid_rate = pct(all_cited_valid, all_cited_total)

    print("QUESTION-DRIVEN READING COMPILER")
    print(f"doc lines: {num_lines}   prefix tokens: {res.get('shared_prefix_tokens', 'n/a')}")
    print()
    header = (f"{'id':<22} {'type':<11} {'ok':<3} {'strict':<6} {'ans':<34} "
              f"{'gold':<26} {'cite':<12} {'prec':>6} {'grd':<4}")
    print(header)
    print("-" * len(header))
    for r in rows:
        ans = r["answer"].replace("\n", " ")[:33]
        gold = r["gold"].replace("\n", " ")[:25]
        cites = ",".join(str(x) for x in r["citations"]) or "-"
        prec = fmt_pct(None if r["citation_precision"] is None
                       else 100.0 * r["citation_precision"], 0)
        print(f"{r['id']:<22} {r['type']:<11} {str(r['correct']):<3} "
              f"{str(r['strict_correct']):<6} {ans:<34} {gold:<26} "
              f"{cites:<12} {prec:>6} {str(r['grounded']):<4}")

    print()
    print("TOTALS")
    print(f"  strict match            : {strict_n}/10")
    print(f"  answer correctness      : {correct_n}/10 ({100.0 * correct_n / 10:.1f}%)")
    print(f"    non-absent            : {present_correct}/{present_n}")
    print(f"    absent                : {absent_correct}/{absent_n}")
    for t in ("single_hop", "numeric", "multi_hop", "absent"):
        got, tot = by_type.get(t, [0, 0])
        if tot:
            print(f"    {t:<21}: {got}/{tot}")
    print(f"  citation precision      : {fmt_pct(citation_precision)} "
          f"({cited_good}/{cited_total} non-absent cited lines contain a gold token)")
    print(f"  citation in-range       : {fmt_pct(citation_valid_rate)} "
          f"({all_cited_valid}/{all_cited_total} all cited lines in 1..{num_lines})")
    print(f"  grounded present        : {grounded_present}/{present_n}")
    print(f"  grounded absent         : {grounded_absent}/{absent_n} "
          f"(correct NOT_STATED with no fabricated citation)")
    print(f"  grounded all            : {grounded_present + grounded_absent}/10")
    print()
    print("LATENCY")
    print(f"  batch wall (10 answers) : {batch_wall:.1f} ms "
          f"(prefill {prefill:.1f} + decode {batch_decode_ms:.1f})")
    print(f"  serial wall est. (sum)  : {serial_total:.1f} ms "
          f"(prefill {prefill:.1f} ms + serial completion each)")
    print(f"  serial decode-only sum  : {serial_decode_total:.1f} ms")
    print(f"  speedup (batch)         : {'n/a' if speedup is None else f'{speedup:.2f}x'}")
    print(f"  speedup (decode-only)   : {'n/a' if speedup_decode is None else f'{speedup_decode:.2f}x'}")
    print()
    g_correct = correct_n >= 7
    g_precision = citation_precision is not None and citation_precision >= 70.0
    g_speed = speedup is not None and speedup >= 4.0
    print("GATES (success = all three)")
    print(f"  correctness >= 7/10     : {'PASS' if g_correct else 'FAIL'} ({correct_n}/10)")
    print(f"  citation precision >=70%: {'PASS' if g_precision else 'FAIL'} "
          f"({fmt_pct(citation_precision)})")
    print(f"  batch speedup >= 4x     : {'PASS' if g_speed else 'FAIL'} "
          f"({'n/a' if speedup is None else f'{speedup:.2f}x'})")
    if g_correct and g_precision and g_speed:
        verdict = "SUCCESS"
    elif correct_n <= 4:
        verdict = "DEAD END (correctness <= 4/10)"
    elif 5 <= correct_n <= 6:
        verdict = "PARTIAL (correctness 5-6/10; at least one other gate failed)"
    else:
        verdict = "PARTIAL (correctness >= 7/10 but another gate failed)"
    if not g_precision and cited_total:
        verdict += "; citation grounding weak"
    print(f"VERDICT: {verdict}")

    summary = {
        "doc_lines": num_lines,
        "shared_prefix_tokens": res.get("shared_prefix_tokens"),
        "strict_correct": strict_n,
        "correct": correct_n,
        "correct_present": present_correct,
        "present": present_n,
        "correct_absent": absent_correct,
        "absent": absent_n,
        "correct_by_type": {t: v for t, v in by_type.items()},
        "citation_precision_pct": citation_precision,
        "cited_lines_present": cited_total,
        "cited_good": cited_good,
        "cited_lines_all": all_cited_total,
        "cited_in_range": all_cited_valid,
        "citation_valid_pct": citation_valid_rate,
        "grounded_present": grounded_present,
        "grounded_absent": grounded_absent,
        "batch_wall_ms": batch_wall,
        "batch_decode_ms": batch_decode_ms,
        "serial_wall_est_ms": serial_total,
        "serial_decode_ms": serial_decode_total,
        "speedup": speedup,
        "speedup_decode": speedup_decode,
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
