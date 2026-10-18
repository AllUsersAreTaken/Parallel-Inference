"""Convert GSM8K jsonl to TSV (question, gold) for the C++ early-exit tool."""
import json
import sys

src, dst = sys.argv[1], sys.argv[2]
limit = int(sys.argv[3]) if len(sys.argv) > 3 else 200

rows = []
with open(src, encoding="utf-8") as f:
    for line in f:
        o = json.loads(line)
        q = o["question"].replace("\t", " ").replace("\n", " ").strip()
        gold = o["answer"].split("####")[-1].strip().replace(",", "")
        rows.append((q, gold))
        if len(rows) >= limit:
            break

with open(dst, "w", encoding="utf-8") as f:
    for q, g in rows:
        f.write(f"{q}\t{g}\n")
print(f"wrote {len(rows)} rows -> {dst}")
