"""Deterministic GSM8K exemplar picker for in-context demonstration arms.

Reads 1-indexed inclusive JSONL line numbers, strips ``<<expr=result>>``
annotations from each solution, drops the ``#### <gold>`` line, and writes one
escaped exemplar per line (``\\n``, ``\\t``, ``\\\\``), matching the escaped-TSV
convention used by ``harness/engine/modality.cpp``.

Selection:
  1. build a signature per candidate from the operator sequence of its
     annotation chain (for example ``-*`` or ``/+``);
  2. greedily take one candidate per distinct signature, preferring shorter
     solutions (ties by JSONL line number);
  3. if fewer than K distinct signatures exist, fill with the candidates whose
     signature is farthest (Levenshtein) from everything already selected.

The divergent file contains the K selected exemplars. The same file repeats
the first selected exemplar K times, so both arms see equal token budgets.

Usage:
    python scripts/gen_exemplars.py --jsonl data/gsm8k_test.jsonl \
        --lines 201-240 --k 8 --out results/exemplars_div.txt
"""
import argparse
import json
import os
import re
import sys

ANNOTATION_RE = re.compile(r"<<[^>]*>>")
EXPR_RE = re.compile(r"<<([^=>]+)=[^>]*>>")
OP_RE = re.compile(r"[+\-*/]")
WS_RE = re.compile(r"\s+")
ESCAPE_MAP = {"\\": "\\\\", "\n": "\\n", "\t": "\\t", "\r": "\\r"}


def strip_annotations(answer):
    return ANNOTATION_RE.sub("", answer)


def normalize_steps(solution):
    """Drop the #### line, strip annotations, and normalize per-line whitespace."""
    gold = None
    step_lines = []
    for line in strip_annotations(solution).splitlines():
        stripped = line.strip()
        if stripped.startswith("####"):
            gold = stripped[4:].strip()
            continue
        norm = WS_RE.sub(" ", line).strip()
        if norm:
            step_lines.append(norm)
    if gold is None:
        raise ValueError("solution has no #### <gold> line")
    return "\n".join(step_lines), gold


def signature_of(solution):
    ops = []
    for expr in EXPR_RE.findall(solution):
        ops.extend(OP_RE.findall(expr))
    return "".join(ops)


def levenshtein(a, b):
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def parse_lines(spec):
    m = re.fullmatch(r"(\d+)(?:-(\d+))?", spec.strip())
    if not m:
        raise SystemExit(f"--lines must be N or N-M (1-indexed inclusive), got {spec!r}")
    lo = int(m.group(1))
    hi = int(m.group(2)) if m.group(2) else lo
    if lo < 1 or hi < lo:
        raise SystemExit(f"--lines range is invalid: {spec!r}")
    return lo, hi


def load_candidates(jsonl_path, lo, hi):
    candidates = []
    with open(jsonl_path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if lineno < lo:
                continue
            if lineno > hi:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            question = obj["question"].strip()
            solution = obj["answer"]
            steps, gold = normalize_steps(solution)
            text = f"Question: {question}\nSolution: {steps}\nThe answer is: {gold}\n"
            candidates.append({
                "line": lineno,
                "sig": signature_of(solution),
                "text": text,
                "steps_len": len(steps),
            })
    if not candidates:
        raise SystemExit(f"no candidates on lines {lo}-{hi} of {jsonl_path}")
    return candidates


def select(candidates, k):
    if len(candidates) < k:
        raise SystemExit(f"only {len(candidates)} candidates, cannot pick k={k}")

    ordered = sorted(candidates, key=lambda c: (c["steps_len"], c["line"]))
    chosen = []
    seen = set()
    for c in ordered:
        if c["sig"] in seen:
            continue
        chosen.append(c)
        seen.add(c["sig"])
        if len(chosen) == k:
            return chosen

    remaining = [c for c in ordered if c not in chosen]
    while len(chosen) < k:
        best = max(
            remaining,
            key=lambda c: (min(levenshtein(c["sig"], s["sig"]) for s in chosen),
                           -c["steps_len"], -c["line"]),
        )
        chosen.append(best)
        remaining.remove(best)
    return chosen


def escape(s):
    return "".join(ESCAPE_MAP.get(ch, ch) for ch in s)


def default_same_path(out_path):
    base, ext = os.path.splitext(out_path)
    if base.endswith("_div"):
        base = base[:-4] + "_same"
    else:
        base = base + "_same"
    return base + ext


def write_lines(path, lines):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for line in lines:
            fh.write(escape(line) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jsonl", default="data/gsm8k_test.jsonl")
    ap.add_argument("--lines", default="201-240", help="1-indexed inclusive JSONL lines")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--out", default="results/exemplars_div.txt")
    ap.add_argument("--same-out", default="",
                    help="path for the repeated-first-exemplar file "
                         "(default: <out> with _div replaced by _same)")
    args = ap.parse_args()

    if args.k < 1:
        raise SystemExit("--k must be >= 1")
    lo, hi = parse_lines(args.lines)
    candidates = load_candidates(args.jsonl, lo, hi)
    chosen = select(candidates, args.k)
    same_path = args.same_out or default_same_path(args.out)

    write_lines(args.out, [c["text"] for c in chosen])
    write_lines(same_path, [chosen[0]["text"]] * args.k)

    print(f"jsonl: {os.path.abspath(args.jsonl)}")
    print(f"lines: {lo}-{hi} (inclusive), candidates={len(candidates)}, k={args.k}")
    print("chosen (selection order):")
    for idx, c in enumerate(chosen):
        print(f"  [{idx}] line {c['line']:4d}  signature {c['sig'] or '(none)'!r}  "
              f"steps_len={c['steps_len']}")
    print(f"[written] {args.out}")
    print(f"[written] {same_path}  (first exemplar repeated {args.k}x)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
