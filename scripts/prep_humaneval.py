#!/usr/bin/env python3
"""Prepare the first 100 HumanEval tasks for the harness.

Source: the official openai/human-eval data/HumanEval.jsonl (164 tasks).
Writes:
  data/humaneval_100.jsonl  task_id, prompt, test, entry_point (spec form)
  data/humaneval_100.tsv    task_id, entry_point, prompt, test (escaped, for C++)

Tasks are sorted by their numeric index and the first 100 are kept
(HumanEval/0 .. HumanEval/99). The TSV escapes backslash, tab, CR and LF so
the C++ harness can read one task per line without a JSON parser.
"""

import argparse
import gzip
import hashlib
import json
import os
import urllib.request

URL = "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(dest_gz, dest_jsonl):
    print(f"downloading {URL}")
    os.makedirs(os.path.dirname(dest_gz) or ".", exist_ok=True)
    urllib.request.urlretrieve(URL, dest_gz)
    with gzip.open(dest_gz, "rb") as src, open(dest_jsonl, "wb") as dst:
        dst.write(src.read())
    print(f"wrote {dest_jsonl}")


def esc(field):
    return (field.replace("\\", "\\\\")
                 .replace("\t", "\\t")
                 .replace("\r", "\\r")
                 .replace("\n", "\\n"))


def task_index(task_id):
    return int(task_id.rsplit("/", 1)[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/HumanEval.jsonl")
    ap.add_argument("--gz", default="data/HumanEval.jsonl.gz")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out-jsonl", default="data/humaneval_100.jsonl")
    ap.add_argument("--out-tsv", default="data/humaneval_100.tsv")
    args = ap.parse_args()

    if args.download or not os.path.exists(args.src):
        download(args.gz, args.src)

    rows = []
    with open(args.src, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"no tasks in {args.src}")

    rows.sort(key=lambda r: task_index(r["task_id"]))
    selected = rows[: args.n]
    if len(selected) < args.n:
        raise SystemExit(f"only {len(selected)} tasks available, need {args.n}")

    with open(args.out_jsonl, "w", encoding="utf-8", newline="\n") as fh:
        for r in selected:
            fh.write(json.dumps({
                "task_id": r["task_id"],
                "prompt": r["prompt"],
                "test": r["test"],
                "entry_point": r["entry_point"],
            }, ensure_ascii=False) + "\n")

    with open(args.out_tsv, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("task_id\tentry_point\tprompt\ttest\n")
        for r in selected:
            fh.write("\t".join([
                esc(r["task_id"]),
                esc(r["entry_point"]),
                esc(r["prompt"]),
                esc(r["test"]),
            ]) + "\n")

    print(f"tasks: {len(selected)} (HumanEval/{task_index(selected[0]['task_id'])}"
          f"..HumanEval/{task_index(selected[-1]['task_id'])})")
    print(f"entry points: {len({r['entry_point'] for r in selected})} unique")
    print(f"{args.out_jsonl}: {os.path.getsize(args.out_jsonl)} bytes "
          f"sha256={sha256(args.out_jsonl)}")
    print(f"{args.out_tsv}: {os.path.getsize(args.out_tsv)} bytes "
          f"sha256={sha256(args.out_tsv)}")
    print(f"{args.src}: sha256={sha256(args.src)}")


if __name__ == "__main__":
    main()
