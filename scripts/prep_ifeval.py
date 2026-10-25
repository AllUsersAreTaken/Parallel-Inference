#!/usr/bin/env python3
"""Prepare the official IFEval 541-prompt set for the retry bench.

Source (vendored): vendor/ifeval/instruction_following_eval/data/input_data.jsonl
The upstream repository moved: the requested
https://github.com/google-research/instruction_following_eval no longer exists.
The official code/data live in google-research/google-research under
instruction_following_eval/. vendor/ifeval is a sparse checkout of that tree
at commit 3ef52111c1fabc301880818f5a547570e926a76b.

Writes:
    data/ifeval_541.tsv        key <TAB> prompt
    data/ifeval_541_meta.json  key -> {instruction_id_list, kwargs}

Tabs/newlines inside fields are escaped as \\t, \\n, \\r; backslash as \\\\.
The harness reverses this with its unescape_field().

Usage:
    python scripts/prep_ifeval.py
"""

import argparse
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SRC = os.path.join(
    ROOT, "vendor", "ifeval", "instruction_following_eval", "data", "input_data.jsonl"
)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "ifeval_541.tsv"))
    ap.add_argument("--meta", default=os.path.join(ROOT, "data", "ifeval_541_meta.json"))
    args = ap.parse_args()

    if not os.path.exists(args.src):
        sys.exit(f"missing official IFEval data: {args.src}")

    rows = []
    with open(args.src, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if len(rows) != 541:
        sys.exit(f"expected 541 official prompts, found {len(rows)}")

    keys = [int(r["key"]) for r in rows]
    if len(set(keys)) != len(keys):
        sys.exit("duplicate keys in input_data.jsonl")
    prompts = [r["prompt"] for r in rows]
    if len(set(prompts)) != len(prompts):
        sys.exit("duplicate prompts; official checker maps responses by prompt")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as out:
        out.write("key\tprompt\n")
        for r in rows:
            out.write(f"{int(r['key'])}\t{esc(r['prompt'])}\n")

    meta = {
        "source": os.path.relpath(args.src, ROOT).replace("\\", "/"),
        "source_sha256": sha256(args.src),
        "n": len(rows),
        "entries": {
            str(int(r["key"])): {
                "instruction_id_list": list(r["instruction_id_list"]),
                "kwargs": list(r["kwargs"]),
            }
            for r in rows
        },
    }
    with open(args.meta, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(meta, fh, indent=1)
        fh.write("\n")

    print(f"wrote {os.path.relpath(args.out, ROOT)}: {len(rows)} prompts")
    print(f"wrote {os.path.relpath(args.meta, ROOT)}")
    print(f"source sha256: {meta['source_sha256']}")
    print(f"tsv sha256:    {sha256(args.out)}")
    print(f"meta sha256:   {sha256(args.meta)}")


if __name__ == "__main__":
    main()
