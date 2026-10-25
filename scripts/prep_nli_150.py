#!/usr/bin/env python3
"""Write data/nli_150.tsv (50 head + 50 ambiguous + 50 tail).

Columns: id, category, prompt, answer, keywords, expected_rel
  answer       canonical short answer (reference sentence for ambiguous)
  keywords     '|' separated; any hit counts as "states the fact"
  expected_rel category prior: head/ambiguous AGREE, tail CONFLICT.
               The analysis uses actual A/B correctness, not this prior.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nli_head import HEAD
from nli_ambig import AMBIG
from nli_tail import TAIL

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "nli_150.tsv")


def esc(s):
    return (s.replace("\\", "\\\\").replace("\t", "\\t")
             .replace("\n", "\\n").replace("\r", "\\r"))


def main():
    assert len(HEAD) == 50, len(HEAD)
    assert len(AMBIG) == 50, len(AMBIG)
    assert len(TAIL) == 50, len(TAIL)
    rows = []
    for i, (p, a, k) in enumerate(HEAD, 1):
        rows.append((f"h{i:02d}", "head", p, a, k, "AGREE"))
    for i, (p, a, k) in enumerate(AMBIG, 1):
        rows.append((f"a{i:02d}", "ambiguous", p, a, k, "AGREE"))
    for i, (p, a, k) in enumerate(TAIL, 1):
        rows.append((f"t{i:02d}", "tail", p, a, k, "CONFLICT"))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write("id\tcategory\tprompt\tanswer\tkeywords\texpected_rel\n")
        for r in rows:
            f.write("\t".join(esc(x) for x in r) + "\n")
    print(f"wrote {OUT} rows={len(rows)} "
          f"head={sum(r[1] == 'head' for r in rows)} "
          f"ambiguous={sum(r[1] == 'ambiguous' for r in rows)} "
          f"tail={sum(r[1] == 'tail' for r in rows)}")


if __name__ == "__main__":
    main()
