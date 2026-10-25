#!/usr/bin/env python3
"""Build a 100-prompt IFEval transfer subset stratified by primary instruction id.

Selection
---------
Primary instruction id = instruction_id_list[0] (official prompt order).
Take up to --per-type prompts per primary id, sampled without replacement with
random.Random(--seed). If the primary ids do not yield --total prompts, fill
from the remaining prompts in a second seeded pass. Rows are written in
original official order so the harness prompt index (and therefore the seed,
1234 + idx) matches the full-set runs.

The official meta/checker are not modified: prompts, instruction ids, and
kwargs come from vendor/ifeval/instruction_following_eval/data/input_data.jsonl
and are re-verified by the run script.

Writes
------
    data/ifeval_qwen_100.tsv   key <TAB> prompt <TAB> idx
    data/ifeval_qwen_100_provenance.json

Usage
-----
    python scripts/prep_ifeval_100.py
"""

import argparse
import collections
import hashlib
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "vendor", "ifeval"))

from instruction_following_eval import evaluation_lib  # noqa: E402


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


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--official-data", default=os.path.join(
        ROOT, "vendor", "ifeval", "instruction_following_eval", "data",
        "input_data.jsonl"))
    ap.add_argument("--out", default=os.path.join(
        ROOT, "data", "ifeval_qwen_100.tsv"))
    ap.add_argument("--provenance", default=os.path.join(
        ROOT, "data", "ifeval_qwen_100_provenance.json"))
    ap.add_argument("--total", type=int, default=100)
    ap.add_argument("--per-type", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    inputs = evaluation_lib.read_prompt_list(args.official_data)
    if len(inputs) != 541:
        raise SystemExit(f"official data has {len(inputs)} prompts, expected 541")

    pools = collections.defaultdict(list)
    for i, inp in enumerate(inputs):
        pools[inp.instruction_id_list[0]].append(i)

    rng = random.Random(args.seed)
    chosen = []
    per_type = {}
    for iid in sorted(pools):
        pool = pools[iid]
        take = rng.sample(pool, min(args.per_type, len(pool)))
        chosen.extend(take)
        per_type[iid] = len(take)

    if len(chosen) < args.total:
        rest = [i for i in range(len(inputs)) if i not in set(chosen)]
        need = args.total - len(chosen)
        chosen.extend(rng.sample(rest, min(need, len(rest))))

    chosen = sorted(set(chosen))[: args.total]
    if len(chosen) != args.total:
        raise SystemExit(f"selected {len(chosen)} prompts, wanted {args.total}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("key\tprompt\tidx\n")
        for i in chosen:
            inp = inputs[i]
            fh.write(f"{inp.key}\t{esc(inp.prompt)}\t{i}\n")

    prov = {
        "source": os.path.relpath(args.official_data, ROOT).replace("\\", "/"),
        "source_sha256": sha256(args.official_data),
        "selection": {
            "rule": ("primary instruction id = instruction_id_list[0]; up to "
                     f"{args.per_type} per primary id, sampled with "
                     f"random.Random({args.seed}); rows in official order"),
            "seed": args.seed,
            "total": len(chosen),
        },
        "per_primary_type_selected": per_type,
        "per_primary_type_pool": {k: len(v) for k, v in sorted(pools.items())},
        "keys": [str(inputs[i].key) for i in chosen],
        "tsv_sha256": sha256(args.out),
    }
    with open(args.provenance, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(prov, fh, indent=1)
        fh.write("\n")

    print(f"wrote {os.path.relpath(args.out, ROOT)}: {len(chosen)} prompts")
    print(f"wrote {os.path.relpath(args.provenance, ROOT)}")
    print(f"tsv sha256: {prov['tsv_sha256']}")
    print("primary types selected: "
          f"{len(per_type)}; counts {sorted(set(per_type.values()))}")


if __name__ == "__main__":
    main()
