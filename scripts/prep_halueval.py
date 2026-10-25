#!/usr/bin/env python3
"""Prepare a fixed HaluEval QA subset for the flag-only bench.

Source: https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json
(JSON Lines, 10k examples). Samples 200 pairs with seed 1234 and writes:

    data/halueval_qa_200.tsv
    id  knowledge  question  right_answer  hallucinated_answer

Tabs/newlines inside fields are escaped as \\t, \\n, \\r; backslash as \\\\.
The harness reverses this with its unescape_field().

Usage:
    python scripts/prep_halueval.py
    python scripts/prep_halueval.py --n 200 --seed 1234
"""

import argparse
import hashlib
import json
import os
import random
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_URL = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def esc(s):
    out = []
    for ch in s:
        if ch == "\\":
            out.append("\\\\")
        elif ch == "\t":
            out.append("\\t")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        else:
            out.append(ch)
    return "".join(out)


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, list):
                rows.extend(obj)
            else:
                rows.append(obj)
    return rows


def download(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"downloading {SRC_URL}")
    req = urllib.request.Request(SRC_URL, headers={"User-Agent": "RecycleLM/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(path, "wb") as out:
        out.write(resp.read())
    print(f"wrote {path} ({os.path.getsize(path)} bytes)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(ROOT, "data", "halueval_qa_data.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "halueval_qa_200.tsv"))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    if not os.path.exists(args.src):
        download(args.src)

    rows = load_jsonl(args.src)
    if len(rows) < args.n:
        sys.exit(f"source has {len(rows)} rows, need {args.n}")

    need = ("knowledge", "question", "right_answer", "hallucinated_answer")
    for i, r in enumerate(rows[:5]):
        missing = [k for k in need if k not in r]
        if missing:
            sys.exit(f"row {i} missing fields: {missing}")

    rnd = random.Random(args.seed)
    idx = sorted(rnd.sample(range(len(rows)), args.n))

    with open(args.out, "w", encoding="utf-8", newline="\n") as out:
        out.write("id\tknowledge\tquestion\tright_answer\thallucinated_answer\n")
        for j, i in enumerate(idx):
            r = rows[i]
            fields = [f"qa{i:05d}"] + [str(r[k]) for k in need]
            out.write("\t".join(esc(f) for f in fields) + "\n")

    prov = {
        "source": SRC_URL,
        "source_sha256": sha256(args.src),
        "source_rows": len(rows),
        "out": os.path.relpath(args.out, ROOT).replace("\\", "/"),
        "out_sha256": sha256(args.out),
        "n": args.n,
        "seed": args.seed,
        "sampled_indices": idx,
    }
    prov_path = args.out + ".provenance.json"
    with open(prov_path, "w", encoding="utf-8") as fh:
        json.dump(prov, fh, indent=2)
        fh.write("\n")

    print(f"wrote {args.out}: {args.n} pairs (seed {args.seed})")
    print(f"wrote {prov_path}")


if __name__ == "__main__":
    main()
