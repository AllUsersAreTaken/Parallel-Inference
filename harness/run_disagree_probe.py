#!/usr/bin/env python3
"""Disagreement probe: can inter-stream agreement on GSM8K act as a
hallucination score?

Design (echoes the earlier E2 N-stream sweep, results/e2_sweep200.csv):
  * 6 problems from data/gsm8k_heldout.tsv picked deterministically by
    question-length quantile (0.10/0.20/0.30 easy-ish, 0.70/0.80/0.90
    harder) -- no outcome-based cherry-picking.
  * N=8 streams per problem, one prompt repeated 8x in a batch TSV so
    parallel_cli derives seeds 1234+row (i.i.d. samples, temp 0.6,
    top_p 0.95, top_k 20 -- the E2 sampling chain).
  * Answers extracted with the proven extractor from
    harness/analyze_modality_vote.py (balanced \boxed{} -> "the answer
    is" line -> last number), canonicalized with M.canon.
  * Per problem: distinct answers, plurality + share, pairwise agreement
    among answered streams, unanimity, gold support, per-stream hits.
    Across problems: unanimous/split x correct/wrong buckets.

Two models by default: Qwen2.5-3B (math-capable) and Llama-3.2-3B
(playground default). Flags for follow-up runs:

  --n 30                  30 problems, evenly spaced length quantiles
  --models llama3.2-3b    one model only
  --out disagree_probe30  artifact name (results/<name>.{json,csv})

Original row-38 run: default flags (6 problems x 2 models, disagree_probe).

Usage:  python harness/run_disagree_probe.py [--n N] [--models a,b] [--out name]
Writes: results/<out>.json, results/<out>.csv
"""
import argparse
import itertools
import json
import math
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "harness"))
import analyze_modality_vote as M  # noqa: E402

CLI = os.path.join(ROOT, "app", "parallel_cli.exe")
BIN = os.path.join(ROOT, "build", "cuda-graphs", "bin")
DATA = os.path.join(ROOT, "data", "gsm8k_heldout.tsv")
SCRATCH = os.path.join(ROOT, "data", "_disagree_probe.tsv")
OUT = os.path.join(ROOT, "data", "_disagree_probe.json")

MODELS = {
    "qwen2.5-3b": os.path.join(ROOT, "models", "Qwen2.5-3B-Instruct-Q4_K_M.gguf"),
    "llama3.2-3b": os.path.join(ROOT, "models", "Llama-3.2-3B-Instruct-Q4_K_M.gguf"),
}
FRACS = (0.10, 0.20, 0.30, 0.70, 0.80, 0.90)
N = 8
MAX_TOK = 512
TEMP, TOP_P, SEED = 0.6, 0.95, 1234

INSTR = ("Solve the problem step by step. Then end with a line in the "
         "exact form: The answer is: <number>\n\nProblem: ")


def esc(s):
    return (s.replace("\\", "\\\\").replace("\n", "\\n")
             .replace("\r", "\\r").replace("\t", "\\t"))


def load_problems(n_problems):
    rows = []
    with open(DATA, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\r\n")
            if not line:
                continue
            q, gold = line.split("\t")[-2], line.split("\t")[-1]
            rows.append((q, gold))
    rows.sort(key=lambda r: len(r[0]))
    if n_problems == len(FRACS):
        fracs = FRACS
    else:
        # evenly spaced length quantiles, excluding the extremes
        fracs = tuple(round((i + 0.5) / n_problems, 4)
                      for i in range(n_problems))
    return [rows[int(f * len(rows))] for f in fracs], fracs


def ent_conf(answers):
    """Normalized-Shannon confidence: 1 - H(answer dist)/log2(k)."""
    ans = [a for a in answers if a is not None]
    n = len(ans)
    if n == 0:
        return 0.0
    c = {}
    for a in ans:
        c[a] = c.get(a, 0) + 1
    if len(c) == 1:
        return 1.0
    H = -sum((v / n) * math.log2(v / n) for v in c.values())
    return 1 - H / math.log2(len(c))


def margin_score(answers):
    """(top count - runner-up count) / answered streams."""
    ans = [a for a in answers if a is not None]
    if not ans:
        return 0.0
    c = {}
    for a in ans:
        c[a] = c.get(a, 0) + 1
    top2 = sorted(c.values(), reverse=True)[:2]
    return ((top2[0] - top2[1]) / len(ans)) if len(top2) > 1 else 1.0


def run_model(model_path, problems, nstreams, seed):
    results = []
    env = dict(os.environ)
    env["PATH"] = BIN + os.pathsep + env.get("PATH", "")
    for qi, (q, gold) in enumerate(problems):
        prompt = INSTR + q
        with open(SCRATCH, "w", encoding="utf-8") as fh:
            for i in range(nstreams):
                fh.write(f"{i + 1}\t{esc(prompt)}\n")
        r = subprocess.run(
            [CLI, "--model", model_path, "--data", SCRATCH, "--out", OUT,
             "--mode", "batch", "--max-tokens", str(MAX_TOK),
             "--temperature", str(TEMP), "--top-p", str(TOP_P),
             "--seed", str(seed)],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            raise SystemExit(f"parallel_cli failed ({r.returncode}):\n{r.stderr[-800:]}")
        with open(OUT, encoding="utf-8") as fh:
            out = json.load(fh)
        results.append((qi, q, gold, out))
    return results


def analyze(model_name, results, nstreams):
    recs = []
    for qi, q, gold, out in results:
        g = M.canon(M.normalize_number(gold))
        answers, paths, gens, finishes = [], [], [], []
        for st in out["streams"]:
            a, path = M.extract_cot(st["text"])
            answers.append(M.canon(a))
            paths.append(path)
            gens.append(st["generated"])
            finishes.append(st["finish"])
        ans = [a for a in answers if a is not None]
        n_ans = len(ans)
        counts = {}
        for a in ans:
            counts[a] = counts.get(a, 0) + 1
        distinct = len(counts)
        maj, maj_cnt = (max(counts.items(), key=lambda kv: kv[1]) if counts
                        else (None, 0))
        pairs = list(itertools.combinations(ans, 2))
        agree = (sum(1 for a, b in pairs if a == b) / len(pairs)
                 if len(pairs) >= 2 else (1.0 if n_ans == 1 else 0.0))
        recs.append({
            "model": model_name,
            "id": qi + 1,
            "q_len": len(q),
            "q": q,
            "gold": g,
            "answers": answers,
            "paths": paths,
            "finish": finishes,
            "generated": sum(gens),
            "n_answered": n_ans,
            "distinct": distinct,
            "maj": maj,
            "maj_share": (maj_cnt / n_ans) if n_ans else 0.0,
            "pairwise_agree": agree,
            "ent_conf": round(ent_conf(answers), 4),
            "margin": round(margin_score(answers), 4),
            "unanimous": distinct == 1 and n_ans == nstreams,
            "maj_ok": maj is not None and maj == g,
            "gold_support": sum(1 for a in answers if a == g),
            "oracle": any(a == g for a in answers),
            "stream1_ok": answers[0] is not None and answers[0] == g,
            "wall_ms": out["wall_ms"],
        })
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=len(FRACS),
                    help="number of problems (default 6 = the original quantiles)")
    ap.add_argument("--nstreams", type=int, default=N,
                    help="streams per problem (default 8)")
    ap.add_argument("--models", default=",".join(MODELS),
                    help="comma-separated keys: " + ",".join(MODELS))
    ap.add_argument("--out", default="disagree_probe",
                    help="artifact basename under results/")
    ap.add_argument("--seed", type=int, default=SEED,
                    help="base sampling seed (default 1234; per-stream seeds "
                         "are seed+row)")
    args = ap.parse_args()

    model_keys = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in model_keys if m not in MODELS]
    if unknown:
        raise SystemExit(f"unknown model keys {unknown}; have {list(MODELS)}")
    selected = {m: MODELS[m] for m in model_keys}

    problems, fracs = load_problems(args.n)
    print(f"picked {len(problems)} problems (length quantiles {fracs}):")
    for i, (q, g) in enumerate(problems):
        print(f"  {i + 1} gold={g:>6} len={len(q):3d}  {q[:72]}")
    all_recs = []
    for name, path in selected.items():
        print(f"\n=== {name} ===")
        results = run_model(path, problems, args.nstreams, args.seed)
        recs = analyze(name, results, args.nstreams)
        for r in recs:
            print(f"  p{r['id']} wall={r['wall_ms']:>7.0f}ms "
                  f"distinct={r['distinct']} maj={r['maj']!r} "
                  f"share={r['maj_share']:.2f} agree={r['pairwise_agree']:.2f} "
                  f"gold={r['gold']} maj_ok={int(r['maj_ok'])} "
                  f"gold_sup={r['gold_support']}/{args.nstreams} "
                  f"fin={r['finish'].count('eos')}eos/"
                  f"{args.nstreams - r['finish'].count('eos')}cap")
        all_recs.extend(recs)

    # summary: does agreement separate correct from incorrect majorities?
    print("\n=== agreement vs correctness ===")
    for model_name in selected:
        recs = [r for r in all_recs if r["model"] == model_name]
        uni = [r for r in recs if r["unanimous"]]
        split = [r for r in recs if not r["unanimous"]]
        print(f"{model_name}:")
        print(f"  unanimous  {len(uni)}/{len(recs)}  "
              f"maj-correct {sum(r['maj_ok'] for r in uni)}/{len(uni)}")
        print(f"  split      {len(split)}/{len(recs)}  "
              f"maj-correct {sum(r['maj_ok'] for r in split)}/{len(split)}")
        for r in recs:
            tag = ("UNANIMOUS+OK" if r["unanimous"] and r["maj_ok"] else
                   "UNANIMOUS+WRONG" if r["unanimous"] else
                   "SPLIT+OK" if r["maj_ok"] else "SPLIT+WRONG")
            print(f"    p{r['id']} {tag:<16} agree={r['pairwise_agree']:.2f} "
                  f"share={r['maj_share']:.2f} gold_support={r['gold_support']} "
                  f"oracle={int(r['oracle'])}")

    # split-run tabulation + confidence-score ranking of the wrong pluralities
    print("\n=== split-run tabulation (confidence scores, worst first) ===")
    print(f"{'model':<13}{'p':>3} {'maj':>7} {'dist':>4} {'share':>5} "
          f"{'agree':>5} {'ent':>4} {'margin':>6} {'maj_ok':>6} "
          f"{'gold#':>5} {'any_ok':>6}")
    for model_name in selected:
        split = sorted((r for r in all_recs
                        if r["model"] == model_name and not r["unanimous"]),
                       key=lambda r: r["pairwise_agree"])
        for r in split:
            print(f"{r['model']:<13}{r['id']:>3} {str(r['maj']):>7} "
                  f"{r['distinct']:>4} {r['maj_share']:>5.2f} "
                  f"{r['pairwise_agree']:>5.2f} {r['ent_conf']:>4.2f} "
                  f"{r['margin']:>6.2f} {r['maj_ok']:>6} "
                  f"{r['gold_support']:>5} {int(r['gold_support'] >= 1):>6}")
        # does each score put every wrong plurality in the bottom rank?
        for score in ("pairwise_agree", "maj_share", "ent_conf", "margin"):
            order = sorted(split, key=lambda r: r[score])
            worst_ok = min((i for i, r in enumerate(order) if r["maj_ok"]),
                           default=len(order))
            n_wrong = sum(1 for r in split if not r["maj_ok"])
            wrong_ranks = [i for i, r in enumerate(order) if not r["maj_ok"]]
            print(f"  {score}: wrong pluralities at ranks {wrong_ranks} "
                  f"of {len(order)} (0=lowest score); first correct at {worst_ok}")
        n_wrong = sum(1 for r in split if not r["maj_ok"])
        gold_pool = sum(1 for r in split if r["gold_support"] >= 1)
        print(f"  split runs {len(split)}, majority correct "
              f"{len(split) - n_wrong}/{len(split)}, >=1 correct stream "
              f"{gold_pool}/{len(split)}")

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    jpath = os.path.join(ROOT, "results", args.out + ".json")
    with open(jpath, "w", encoding="utf-8") as fh:
        json.dump({"n_streams": args.nstreams, "temp": TEMP, "top_p": TOP_P,
                   "seed": args.seed, "max_tokens": MAX_TOK,
                   "selection": "question-length quantiles " + str(fracs),
                   "models": list(selected), "records": all_recs}, fh, indent=1)
    cpath = os.path.join(ROOT, "results", args.out + ".csv")
    cols = ["model", "id", "q_len", "gold", "answers", "distinct", "maj",
            "maj_share", "pairwise_agree", "ent_conf", "margin",
            "unanimous", "maj_ok", "gold_support", "oracle", "stream1_ok",
            "n_answered", "wall_ms"]
    with open(cpath, "w", encoding="utf-8", newline="") as fh:
        fh.write(",".join(cols) + "\n")
        for r in all_recs:
            fh.write(",".join(str(r[c]) for c in cols) + "\n")
    print(f"\nwrote {jpath}\nwrote {cpath}")


if __name__ == "__main__":
    main()
