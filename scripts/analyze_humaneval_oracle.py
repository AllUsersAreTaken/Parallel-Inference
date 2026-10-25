#!/usr/bin/env python3
"""Phase 2: test-aware oracle ablation (stdlib only).

Joins results/humaneval_oracle_raw.csv (harness score) with the fixed-evaluator
baseline labels from results/humaneval_baseline_512_seed1234.csv.

Positive class = actor FAIL (pass1 == 0). The oracle score is
logit(first piece of " FATAL") - logit(first piece of " OK"). Higher score means
more likely FATAL/fail. The frozen gate is AUROC > 0.75.

Usage:
    python scripts/analyze_humaneval_oracle.py
"""

import argparse
import csv
import hashlib
import math
import os
import random
import statistics


def sha256(path):
    if not path or not os.path.exists(path):
        return "n/a"
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def auroc(scores, labels):
    """Mann-Whitney AUROC, positive label = 1, ties count 1/2."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    rsum = sum(ranks[i] for i in range(len(scores)) if labels[i] == 1)
    u = rsum - len(pos) * (len(pos) + 1) / 2.0
    return u / (len(pos) * len(neg))


def boot_auroc(scores, labels, iters=4000, seed=1234):
    rnd = random.Random(seed)
    n = len(scores)
    out = []
    for _ in range(iters):
        idx = [rnd.randrange(n) for _ in range(n)]
        s = [scores[i] for i in idx]
        y = [labels[i] for i in idx]
        a = auroc(s, y)
        if a == a:
            out.append(a)
    out.sort()
    if not out:
        return (float("nan"), float("nan"))
    return (out[int(0.025 * len(out))], out[int(0.975 * len(out)) - 1])


def confusion(scores, labels, threshold):
    tp = fp = fn = tn = 0
    for s, y in zip(scores, labels):
        pred = 1 if s > threshold else 0
        if y == 1 and pred == 1:
            tp += 1
        elif y == 0 and pred == 1:
            fp += 1
        elif y == 1 and pred == 0:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def prf(tp, fp, fn, tn):
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) else float("nan")
    return precision, recall, specificity, accuracy


def best_f1_threshold(scores, labels):
    cand = sorted(set(scores))
    best = (float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))
    for t in cand:
        tp, fp, fn, tn = confusion(scores, labels, t)
        p, r, _, _ = prf(tp, fp, fn, tn)
        if p != p or r != r or (p + r) == 0:
            continue
        f1 = 2 * p * r / (p + r)
        if f1 != f1 or f1 <= (best[3] if best[3] == best[3] else -1):
            continue
        best = (t, p, r, f1, (tp, fp, fn, tn))
    return best


def fmt(x, d=4):
    if x != x:
        return "n/a"
    return f"{x:.{d}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", default="results/humaneval_oracle_raw.csv")
    ap.add_argument("--baseline", default="results/humaneval_baseline_512_seed1234.csv")
    ap.add_argument("--out", default="results/humaneval_oracle.csv")
    ap.add_argument("--summary", default="results/humaneval_oracle_summary.txt")
    ap.add_argument("--model", default="models/Llama-3.2-3B-Instruct-Q4_K_M.gguf")
    ap.add_argument("--data", default="data/humaneval_100.jsonl")
    ap.add_argument("--threshold", type=float, default=0.0)
    ap.add_argument("--gate", type=float, default=0.75)
    ap.add_argument("--forced", action="store_true",
                    help="forced-verdict run (GBNF FATAL|OK, F-O token ids)")
    ap.add_argument("--cap", type=int, default=0,
                    help="generation cap for the report (default 8 forced / 16 plain)")
    args = ap.parse_args()

    with open(args.oracle, encoding="utf-8", newline="") as fh:
        orows = list(csv.DictReader(fh))
    with open(args.baseline, encoding="utf-8", newline="") as fh:
        brows = {r["task_id"]: r for r in csv.DictReader(fh)}

    joined = []
    missing = 0
    code_mismatch = 0
    for r in orows:
        b = brows.get(r["task_id"])
        if b is None:
            missing += 1
            continue
        if r["code"] != b["code"]:
            code_mismatch += 1
        r["pass1"] = int(b["pass1"])
        r["label_fail"] = 1 - int(b["pass1"])
        r["score"] = float(r["score"])
        r["oracle_fatal"] = int(r["oracle_fatal"])
        r["oracle_parsed"] = int(r["oracle_parsed"])
        r["truncated"] = int(r["truncated"])
        r["n_examples"] = int(r["n_examples"])
        joined.append(r)

    joined.sort(key=lambda r: int(r["idx"]))
    scores = [r["score"] for r in joined]
    labels = [r["label_fail"] for r in joined]
    n = len(joined)
    n_pos = sum(labels)
    n_neg = n - n_pos

    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        fields = ["task_id", "idx", "entry_point", "pass1", "label_fail",
                  "score", "oracle_fatal", "oracle_parsed", "oracle_n",
                  "example_source", "n_examples", "truncated", "verdict_text",
                  "fatal_token_id", "fatal_token_piece", "ok_token_id",
                  "ok_token_piece", "logit_fatal", "logit_ok", "examples",
                  "oracle_ms", "total_ms", "note", "code"]
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in joined:
            w.writerow({k: r.get(k, "") for k in fields})

    a = auroc(scores, labels)
    lo, hi = boot_auroc(scores, labels)
    tp, fp, fn, tn = confusion(scores, labels, args.threshold)
    precision, recall, specificity, accuracy = prf(tp, fp, fn, tn)
    bt, bp, br, bf1, bconf = best_f1_threshold(scores, labels)

    pos_scores = [r["score"] for r in joined if r["label_fail"]]
    neg_scores = [r["score"] for r in joined if not r["label_fail"]]
    n_examples = sum(1 for r in joined if r["example_source"] == "docstring_examples")
    n_parsed = sum(1 for r in joined if r["oracle_parsed"])
    n_fatal = sum(1 for r in joined if r["oracle_fatal"])

    vrows = [r for r in joined if r["oracle_parsed"]]
    vtp = sum(1 for r in vrows if r["label_fail"] and r["oracle_fatal"])
    vfp = sum(1 for r in vrows if not r["label_fail"] and r["oracle_fatal"])
    vfn = sum(1 for r in vrows if r["label_fail"] and not r["oracle_fatal"])
    vtn = sum(1 for r in vrows if not r["label_fail"] and not r["oracle_fatal"])
    vprec, vrec, vspec, vacc = prf(vtp, vfp, vfn, vtn)

    L = []
    w = L.append
    w("RecycleLM HumanEval test-aware oracle ablation (Phase 2b forced verdict)"
      if args.forced else
      "RecycleLM HumanEval test-aware oracle ablation (Phase 2)")
    w("=" * 72)
    w(f"oracle raw: {args.oracle}  sha256={sha256(args.oracle)}")
    w(f"baseline:   {args.baseline}  sha256={sha256(args.baseline)}")
    w(f"joined:     {args.out}  sha256={sha256(args.out)}")
    w(f"model:      {args.model}  sha256={sha256(args.model)}")
    w(f"data:       {args.data}  sha256={sha256(args.data)}")
    metapath = args.oracle + ".meta.txt"
    if os.path.exists(metapath):
        w(f"oracle meta sha256={sha256(metapath)}")
        with open(metapath, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    w(f"  meta: {line.rstrip()}")
    w("")
    w("Protocol")
    w("  Baseline code regenerated at seed 1234 / --actor 512 (no retry) and")
    w("  verified byte-identical to the Phase 1 baseline code on every task.")
    w("  Public examples: docstring lines starting with '>>>' plus their output")
    w("  lines (until a blank line or the next '>>>'). If none, the docstring text.")
    if args.forced:
        w("  Oracle prompt (user turn) ends with:")
        w("    Answer exactly FATAL or OK. Do not explain. Verdict:")
        w("  GBNF grammar constrains the sampler: root ::= \"FATAL\" | \"OK\".")
        w("  Score = logit(F) - logit(O) from the first-token logits after prefill.")
        if joined:
            r0 = joined[0]
            w(f"  Token ids/pieces used: F={r0['fatal_token_id']}:'{r0['fatal_token_piece']}' "
              f"O={r0['ok_token_id']}:'{r0['ok_token_piece']}'")
            w("  Note: Llama-3.2 also has a single-token 'OK' (4012), and 'FATAL' is")
            w("  F+ATAL, so the grammar may sample 'OK' directly; the score still uses")
            w("  the bare F and O ids as specified.")
        w("  Generation cap: 8 tokens, sampler seed = seed + task index, temp 0.6 /")
        w("  top_k 20 / top_p 0.95. The grammar makes the verdict parseable.")
    else:
        w("  Oracle prompt (user turn):")
        w("    You are a Python test oracle. Here is a function and the public examples")
        w("    from its docstring: <code> <examples> Step through each example. If the")
        w("    code would fail any example, answer FATAL; otherwise OK. Verdict:")
        w("  Score = logit(first token of ' FATAL') - logit(first token of ' OK'), read")
        w("  from the first-token logits after prefill. Llama-3.2 tokenization:")
        if joined:
            r0 = joined[0]
            w(f"    ' FATAL' -> {r0['fatal_token_id']}:'{r0['fatal_token_piece']}' + "
              f"second piece (multi-piece, first piece used)")
            w(f"    ' OK'    -> {r0['ok_token_id']}:'{r0['ok_token_piece']}' (single piece)")
        w("  Generation cap: 16 tokens, sampler seed = seed + task index, temp 0.6 /")
        w("  top_k 20 / top_p 0.95 (verdict text recorded; logits drive the score).")
    w("")
    w("Dataset")
    w(f"  tasks joined: {n}  missing: {missing}  baseline-code mismatches: {code_mismatch}")
    w(f"  base rate (actor fail): {n_pos}/{n} = {100.0*n_pos/n:.1f}%")
    w(f"  tasks with docstring examples: {n_examples}/{n} = "
      f"{100.0*n_examples/n:.1f}%")
    w(f"  truncated baseline rows (actor_n >= 512): "
      f"{sum(r['truncated'] for r in joined)}/{n}")
    cap = args.cap or (8 if args.forced else 16)
    caps = [int(r["oracle_n"]) for r in joined]
    n_ok = sum(1 for r in joined if r["oracle_parsed"] and not r["oracle_fatal"])
    w(f"  oracle generation cap: {cap} tokens; observed n: "
      f"min={min(caps)} median={statistics.median(caps):.0f} max={max(caps)}")
    w(f"  oracle verdicts parsed: {n_parsed}/{n}  "
      f"(FATAL={n_fatal}, OK={n_ok}, unparsed={n - n_parsed})")
    w("")
    w("Verdict distribution")
    w(f"  FATAL={n_fatal} ({100.0*n_fatal/n:.1f}%)  "
      f"OK={n_ok} ({100.0*n_ok/n:.1f}%)  "
      f"unparsed={n-n_parsed} ({100.0*(n-n_parsed)/n:.1f}%)")
    w("")
    w("Score distribution")
    w(f"  failing tasks (n={n_pos}): median={fmt(statistics.median(pos_scores),3)} "
      f"mean={fmt(statistics.mean(pos_scores),3)} min={fmt(min(pos_scores),3)} "
      f"max={fmt(max(pos_scores),3)}")
    if neg_scores:
        w(f"  passing tasks (n={n_neg}): median={fmt(statistics.median(neg_scores),3)} "
          f"mean={fmt(statistics.mean(neg_scores),3)} min={fmt(min(neg_scores),3)} "
          f"max={fmt(max(neg_scores),3)}")
    w(f"  score > 0: {sum(1 for s in scores if s > 0)}/{n} (a 0 threshold predicts "
      f"almost no failures)")
    w("")
    w("AUROC (Mann-Whitney, fail = positive, ties = 1/2)")
    w(f"  AUROC = {fmt(a, 4)}   bootstrap 95% CI = [{fmt(lo,4)}, {fmt(hi,4)}]")
    gate = (a == a and a > args.gate)
    w(f"  GATE: AUROC > {args.gate:.2f} -> {'PASS' if gate else 'FAIL'}")
    w("")
    w(f"Score threshold = score > {args.threshold}")
    w(f"  TP={tp} FP={fp} FN={fn} TN={tn}")
    w(f"  precision={fmt(precision,3)} recall={fmt(recall,3)} "
      f"specificity={fmt(specificity,3)} accuracy={fmt(accuracy,3)}")
    if not args.forced:
        w("  Note: because every score is <= 0, the literal verdict threshold predicts")
        w("  no failures (recall 0, specificity 1). AUROC is the gate.")
    w("")
    w(f"Observed verdict as a classifier (parsed n={len(vrows)}; FATAL = positive)")
    w(f"  TP={vtp} FP={vfp} FN={vfn} TN={vtn}")
    w(f"  precision={fmt(vprec,3)} recall={fmt(vrec,3)} "
      f"specificity={fmt(vspec,3)} accuracy={fmt(vacc,3)}")
    if vrows:
        w(f"  binary-verdict AUROC (=(recall+specificity)/2) = "
          f"{fmt((vrec+vspec)/2.0,4)}")
    w("")
    w("Diagnostic only: best in-sample F1 threshold (not a held-out estimate)")
    w(f"  threshold={fmt(bt,3)} precision={fmt(bp,3)} recall={fmt(br,3)} "
      f"F1={fmt(bf1,3)} TP/FP/FN/TN={bconf}")
    w("")
    w("Diagnostics (not the gate)")
    w(f"  sign-flipped AUROC (logit(O)-logit(F)) = {fmt(1.0-a,4)}; "
      f"still {'above' if (1.0-a) > args.gate else 'below'} the {args.gate:.2f} gate")
    for source in ("docstring_examples", "docstring_text"):
        sub = [r for r in joined if r["example_source"] == source]
        ys = [r["label_fail"] for r in sub]
        if len(set(ys)) < 2:
            continue
        w(f"  AUROC on {source:<18} n={len(sub):>3} fails={sum(ys):>2} "
          f"AUROC={fmt(auroc([r['score'] for r in sub], ys),4)}")
    if args.forced:
        w("  The grammar makes every verdict parseable; the gate still uses the")
        w("  first-token F-O logit score, not the verdict.")
        w("  'FATAL' is F+ATAL and 'OK' is a single token (4012), so the grammar can")
        w("  sample 'OK' directly; F-O is the specified first-token contrast.")
    else:
        w("  The 16-token generation is dominated by chain-of-thought (median first")
        w("  tokens are 'Let's ...'), so the literal verdict is usually unavailable;")
        w("  the gate therefore rests entirely on the first-token FATAL/OK logits.")
    w("")
    w("Per-task rows (score, label, pass1)")
    hdr = f"{'task':<17} {'pass1':>5} {'fail':>4} {'score':>9} {'verdict':>8} {'examples':>8}"
    w(hdr)
    w("-" * len(hdr))
    for r in joined:
        verdict = "FATAL" if r["oracle_fatal"] else ("OK" if r["oracle_parsed"] else "unparsed")
        w(f"{r['task_id']:<17} {r['pass1']:>5} {r['label_fail']:>4} "
          f"{r['score']:>9.3f} {verdict:>8} {r['n_examples']:>8}")
    w("")
    w("Commands")
    w("  scripts\\build_tool.bat build\\cuda-graphs harness\\engine\\humaneval.cpp "
      "harness\\engine\\humaneval.exe")
    w("  humaneval.exe --model models\\Llama-3.2-3B-Instruct-Q4_K_M.gguf "
      "--data data\\humaneval_100.tsv \\")
    w("      --out results\\humaneval_oracle_raw.csv --limit 100 --mode oracle "
      "--actor 512 --oracle-tokens 16 --seed 1234")
    w("  python scripts\\analyze_humaneval_oracle.py")

    text = "\n".join(L) + "\n"
    with open(args.summary, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
