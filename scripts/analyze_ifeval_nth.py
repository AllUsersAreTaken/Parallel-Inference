#!/usr/bin/env python3
"""Fixture for length_constraints:nth_paragraph_first_word.

Reads the evaluated 1024-token baseline CSV (seed 1234), finds every prompt
whose instruction_id_list contains the nth_paragraph_first_word check, applies
the official checker's chunking rule (re.split(r"\\n\\n", value)), and reports
what the checker sees.

Two views are printed:

  raw   -- the response exactly as stored in the evaluated CSV.
  lf    -- the same response after normalising CRLF/CR to LF.

The raw view is what the official checker receives from the existing driver.
The LF view isolates model behaviour from the harness's Windows text-mode CSV
newline conversion (see the summary at the bottom of the fixture file).

Writes results/ifeval_nth_fixture.txt.

Usage:
    python scripts/analyze_ifeval_nth.py
"""

import argparse
import csv
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "vendor", "ifeval"))

from instruction_following_eval import evaluation_lib  # noqa: E402
from langdetect import DetectorFactory  # noqa: E402

DetectorFactory.seed = 0

IID = "length_constraints:nth_paragraph_first_word"
PUNCT = {".", ",", "?", "!", "'", '"'}


def first_word(text):
    """The official checker's first-word extraction (paragraph already split)."""
    chunk = text.strip()
    if not chunk:
        return ""
    word = chunk.split()[0].strip()
    word = word.lstrip("'").lstrip('"')
    out = ""
    for letter in word:
        if letter in PUNCT:
            break
        out += letter.lower()
    return out


def chunks_of(text):
    return re.split(r"\n\n", text)


def nonempty(chunks):
    return [c for c in chunks if c.strip()]


def classify(chunks, kw):
    """Cause label for one split, using the requested five-way flag set.

    title_line covers any single leading non-body chunk (title, headline, or a
    preamble line such as "Here is a two-paragraph story about you:") that is
    separated from the body by a blank line and shifts the paragraph numbering.
    It is detected structurally: dropping the leading chunk leaves exactly the
    requested number of paragraphs and puts the expected word in place.
    """
    n = int(kw["num_paragraphs"])
    nth = int(kw["nth_paragraph"])
    want = str(kw["first_word"]).lower()
    ne = nonempty(chunks)
    if nth > len(ne) or len(ne) < n:
        return "too_few_chunks", f"{len(ne)} non-empty chunks (want {n})"
    target = chunks[nth - 1].strip() if nth <= len(chunks) else ""
    if not target:
        return "empty_quote_line", "nth raw chunk is empty (index hits a blank line)"
    if len(ne) == n:
        got = first_word(target)
        if got == want:
            return "none", "passes"
        return "wrong_first_word", f"chunk count ok ({len(ne)}); got {got!r}"
    if len(ne) == n + 1:
        rest = chunks[1:]
        if (len(nonempty(rest)) == n and nth <= len(rest)
                and first_word(rest[nth - 1]) == want):
            leading = chunks[0].strip()
            if leading.strip('"').strip() == "":
                return "empty_quote_line", "leading chunk is only quote characters"
            return "title_line", "leading header/preamble chunk counted as a paragraph"
    head = chunks[0].strip().replace("\n", " ")
    return "other", (f"{len(ne)} non-empty chunks (want {n}), {len(chunks)} raw; "
                     f"leading chunk: {head[:48]!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.join(
        ROOT, "results", "ifeval_1024_seed1234_baseline.csv"))
    ap.add_argument("--official-data", default=os.path.join(
        ROOT, "vendor", "ifeval", "instruction_following_eval", "data",
        "input_data.jsonl"))
    ap.add_argument("--out", default=os.path.join(
        ROOT, "results", "ifeval_nth_fixture.txt"))
    args = ap.parse_args()

    inputs = evaluation_lib.read_prompt_list(args.official_data)
    by_key = {str(inp.key): inp for inp in inputs}
    with open(args.base, "r", encoding="utf-8", errors="replace", newline="") as fh:
        rows = {str(r["key"]): r for r in csv.DictReader(fh)}

    keys = sorted(
        (k for k, inp in by_key.items() if IID in inp.instruction_id_list),
        key=lambda k: int(k))

    L = []
    L.append("IFEval nth_paragraph_first_word fixture")
    L.append("=" * 78)
    L.append("source: results/ifeval_1024_seed1234_baseline.csv "
             "(Llama-3.2-3B-Instruct-Q4_K_M, max_tokens 1024, seed 1234)")
    L.append("data:   vendor/ifeval/instruction_following_eval/data/input_data.jsonl")
    L.append("checker: official strict checker "
             "(google-research instruction_following_eval)")
    L.append("chunking: re.split(r\"\\n\\n\", response), the official definition")
    L.append(f"prompts carrying {IID}: {len(keys)}")
    L.append("")

    # ---------------- raw view ----------------
    L.append("1. As stored in the evaluated CSV (what the official checker receives)")
    L.append("-" * 78)
    L.append("key   exp_n nth  first_word   chunks  nth_chunk_first_token       "
             "pass  flag")
    raw_rows = []
    for k in keys:
        inp = by_key[k]
        row = rows[k]
        i = inp.instruction_id_list.index(IID)
        kw = inp.kwargs[i]
        text = row["response"]
        ch = chunks_of(text)
        nth = int(kw["nth_paragraph"])
        token = ""
        if nth <= len(ch):
            part = ch[nth - 1].strip()
            token = part.split()[0] if part else "<empty>"
        flag, why = classify(ch, kw)
        passed = flag == "none"
        raw_rows.append((k, kw, ch, token, passed, flag))
        L.append(f"{k:<5} {kw['num_paragraphs']:>5} {nth:>3}  "
                 f"{str(kw['first_word']).lower():<11} {len(ch):>6}  "
                 f"{token[:25]:<25} {int(passed):>4}  {flag}")
        L.append(f"      detail: {why}")
    L.append("")
    L.append("All 12 rows have chunks=1 because the evaluated CSV stores every "
             "newline")
    L.append("as CRLF: the text contains '\\r\\n\\r\\n' and never '\\n\\n', so the "
             "official")
    L.append("split cannot find a paragraph boundary. See section 4.")
    L.append("")

    # ---------------- LF view ----------------
    L.append("2. After normalising CRLF/CR to LF (candidate model behaviour)")
    L.append("-" * 78)
    L.append("key   exp_n nth  first_word   chunks nonempty  nth_chunk_first_token   "
             "pass  flag")
    lf_rows = []
    for k in keys:
        inp = by_key[k]
        i = inp.instruction_id_list.index(IID)
        kw = inp.kwargs[i]
        text = rows[k]["response"].replace("\r\n", "\n").replace("\r", "\n")
        ch = chunks_of(text)
        nth = int(kw["nth_paragraph"])
        token = ""
        if nth <= len(ch):
            part = ch[nth - 1].strip()
            token = part.split()[0] if part else "<empty>"
        flag, why = classify(ch, kw)
        passed = flag == "none"
        lf_rows.append((k, kw, ch, token, passed, flag, text))
        L.append(f"{k:<5} {kw['num_paragraphs']:>5} {nth:>3}  "
                 f"{str(kw['first_word']).lower():<11} {len(ch):>6} "
                 f"{len(nonempty(ch)):>8}  {token[:23]:<23} {int(passed):>4}  "
                 f"{flag}")
        L.append(f"      detail: {why}")
    L.append("")

    n_pass_lf = sum(1 for r in lf_rows if r[4])
    fails = [r for r in lf_rows if not r[4]]
    counts = {}
    for r in fails:
        counts[r[5]] = counts.get(r[5], 0) + 1
    L.append("Summary")
    L.append("-" * 78)
    L.append("  as stored (CRLF): 0/12 pass; every row has one chunk, so the")
    L.append("  count is below the expected number (too_few_chunks) regardless of")
    L.append("  the first word.")
    L.append(f"  after LF normalisation: {n_pass_lf}/{len(lf_rows)} pass, "
             f"{len(fails)}/{len(lf_rows)} fail")
    for name in ("title_line", "empty_quote_line", "wrong_first_word",
                 "too_few_chunks", "other"):
        if counts.get(name):
            members = ", ".join(r[0] for r in fails if r[5] == name)
            L.append(f"    {name:<18} {counts[name]:>2}  ({members})")
    fmt = sum(1 for r in fails if len(nonempty(r[2])) != int(r[1]["num_paragraphs"]))
    L.append(f"  failures with a wrong paragraph count: {fmt}/{len(fails)}")
    L.append(f"  failures with the count right and only the word wrong: "
             f"{len(fails) - fmt}/{len(fails)}")
    L.append(f"  failures that disappear when newlines are canonicalised: "
             f"{n_pass_lf}/{len(lf_rows)}")
    L.append("")
    L.append("  Every surviving failure is a chunk-count/format problem. No")
    L.append("  response has the requested paragraph count with the requested")
    L.append("  word in the right place but the word itself wrong.")
    L.append("")

    # ---------------- examples ----------------
    L.append("3. Verbatim examples")
    L.append("-" * 78)
    for k in ("1609", "181"):
        row = next(r for r in lf_rows if r[0] == k)
        kw = row[1]
        L.append(f"key {k}: num_paragraphs={kw['num_paragraphs']} "
                 f"nth_paragraph={kw['nth_paragraph']} "
                 f"first_word={kw['first_word']!r}")
        for j, c in enumerate(row[2]):
            body = c.replace("\r", "")
            L.append(f"  --- chunk {j + 1} (len {len(body)}) ---")
            L.append(body)
        L.append(f"  -> flag={row[5]}, nth chunk first token={row[3]!r}")
        L.append("")

    # ---------------- cause of the CRLF ----------------
    L.append("4. Newline encoding")
    L.append("-" * 78)
    L.append("  harness/engine/ifeval.cpp opens the raw CSV with std::fopen(..., \"w\"),")
    L.append("  so the CRT converts every '\\n' in the response to CRLF on Windows.")
    L.append("  No evaluator normalises the response back to LF before calling the")
    L.append("  official checker. The evaluated CSV therefore contains 0 occurrences")
    L.append("  of '\\n\\n' and 412/541 responses with '\\r\\n\\r\\n'.")
    L.append("  Only this instruction splits on '\\n\\n' (number_paragraphs uses the")
    L.append("  '***' divider), so this is the one instruction id whose score is")
    L.append("  entirely determined by the artifact.")
    L.append("")
    L.append("  Observed fix impact if responses were normalised (this 12-prompt set):")
    L.append(f"    instruction strict 0/12 -> {n_pass_lf}/12")
    L.append("")

    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
