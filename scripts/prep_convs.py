"""Convert conversations.jsonl to a line format the C++ runner can parse.

Record types (tab separated):
  C  <id>
  F  <type>  <value>            planted fact
  Q  <qtype> <fact_idx> <value> <question text>
  T  <user turn text>
Records for one conversation are contiguous; a new C starts the next.
Usage: python prep_convs.py data/conversations.jsonl data/convs.txt
"""
import json
import sys

src, dst = sys.argv[1], sys.argv[2]
with open(src, encoding="utf-8") as fi, open(dst, "w", encoding="utf-8") as fo:
    for line in fi:
        c = json.loads(line)
        fo.write(f"C\t{c['id']}\t{c['version']}\t{c['seed']}\n")
        for f in c["facts"]:
            fo.write(f"F\t{f['type']}\t{f['value']}\n")
        for q in c["questions"]:
            v = q["value"]
            if isinstance(v, list):
                v = "|".join(v)
            fo.write(f"Q\t{q['type']}\t{q['fact_idx']}\t{v}\t{q['text']}\n")
        for t in c["turns"]:
            fo.write("T\t" + t.replace("\t", " ").replace("\n", " ") + "\n")
print(f"wrote {dst}")
