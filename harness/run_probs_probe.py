#!/usr/bin/env python3
"""Row 40: per-token raw-logit probability capture (disagreement line, row 38+39 follow-up).

Captures, for every emitted token, the FULL-vocab (pre temperature/top-k/top-p)
distribution stats: entropy H (nats), top-1 prob p1, top-1 minus top-2 margin,
emitted-token log-prob, byte span in the stream text. One sidecar JSONL per
problem via prob_cli.exe --probs (raw-logit snapshot in runtime.cpp).

Design:
  * 32 problems = 30 question-length quantiles (i+0.5)/30 -- same selection
    rule as the n=30 probe, no outcome cherry-picking -- PLUS the 2 known
    unanimous-wrong questions from the n=100 run (ids 28, 29), deduplicated
    by question text. The 2 extras are labeled src=n100_uw; the blind-spot
    (unanimous-wrong) base rate is ~4% of unanimous runs, so a plain n=30
    would yield 0-1 UW cases.
  * N=8 streams per problem, llama3.2-3b, temp 0.6 / top_p 0.95 / seed 1234 /
    max 512 tok -- identical sampling to rows 38-39.
  * Group labels (UR / SP / UW) come from THIS run's own outcomes via
    run_disagree_probe.analyze().

Writes:
  data/_probs_pNN.json    raw CLI JSON per problem (streams)
  data/_probs_pNN.jsonl   prob sidecar, one JSON object per emitted token
  results/probs_probe30.{json,csv}  combined records + analysis fields

Usage:  python harness/run_probs_probe.py [--nstreams 8] [--limit N] [--dry-run]
"""
import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "harness"))
import run_disagree_probe as D  # noqa: E402

CLI = os.path.join(ROOT, "app", "prob_cli.exe")
BIN = os.path.join(ROOT, "build", "cuda-graphs", "bin")
N100 = os.path.join(ROOT, "results", "disagree_probe100_n5.json")
BASE = "probs_probe30"
NQ = 30


def pick_problems():
    quants, fracs = D.load_problems(NQ)
    seen = {q for q, _ in quants}
    probs = [{"q": q, "gold": g, "frac": f, "src": "quantile"}
             for (q, g), f in zip(quants, fracs)]
    with open(N100, encoding="utf-8") as fh:
        n100 = json.load(fh)
    for r in n100["records"]:
        if r["unanimous"] and not r["maj_ok"]:
            if r["q"] in seen:
                continue
            seen.add(r["q"])
            probs.append({"q": r["q"], "gold": r["gold"],
                          "frac": None, "src": f"n100_id{r['id']}"})
    return probs


def run_one(qi, prompt, nstreams, seed, dry):
    tsv = os.path.join(ROOT, "data", "_probs_p%02d.tsv" % qi)
    out = os.path.join(ROOT, "data", "_probs_p%02d.json" % qi)
    probs = os.path.join(ROOT, "data", "_probs_p%02d.jsonl" % qi)
    with open(tsv, "w", encoding="utf-8") as fh:
        for i in range(nstreams):
            fh.write("%d\t%s\n" % (i + 1, D.esc(prompt)))
    if dry:
        return out, probs, None
    env = dict(os.environ)
    env["PATH"] = BIN + os.pathsep + env.get("PATH", "")
    r = subprocess.run(
        [CLI, "--model", D.MODELS["llama3.2-3b"], "--data", tsv, "--out", out,
         "--mode", "batch", "--max-tokens", str(D.MAX_TOK),
         "--temperature", str(D.TEMP), "--top-p", str(D.TOP_P),
         "--seed", str(seed), "--probs", probs],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise SystemExit("prob_cli failed (p%d, rc=%d):\n%s"
                         % (qi, r.returncode, r.stderr[-800:]))
    with open(out, encoding="utf-8") as fh:
        return out, probs, json.load(fh)


def count_prob_lines(path):
    n = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nstreams", type=int, default=D.N)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    probs = pick_problems()
    if args.limit:
        probs = probs[:args.limit]
    print("picked %d problems (%d quantile + %d UW-boost):"
          % (len(probs), sum(1 for p in probs if p["src"] == "quantile"),
             sum(1 for p in probs if p["src"] != "quantile")))
    for i, p in enumerate(probs):
        print("  p%02d %-10s gold=%-7s len=%d  %s"
              % (i + 1, p["src"], p["gold"], len(p["q"]), p["q"][:60]))

    # capacity guard: est_prompt + max_tok per stream <= 8192 ctx
    est = max(len(p["q"]) for p in probs) // 4 + 40 + D.MAX_TOK
    need = args.nstreams * est
    if need > 8192:
        raise SystemExit("capacity: need ~%d tokens (8192 limit); "
                         "reduce --nstreams" % need)
    print("capacity estimate: ~%d/%d tokens per invocation" % (need, 8192))

    if args.dry_run:
        for i, p in enumerate(probs):
            run_one(i + 1, D.INSTR + p["q"], args.nstreams, D.SEED, True)
        print("dry-run ok (no GPU); exiting")
        return

    results = []
    for i, p in enumerate(probs):
        qi = i + 1
        _, sidecar, out = run_one(qi, D.INSTR + p["q"], args.nstreams,
                                  D.SEED, False)
        nrec = count_prob_lines(sidecar)
        ntok = sum(st["generated"] for st in out["streams"])
        flag = "" if nrec == ntok else "  ** RECORD MISMATCH %d vs %d **" % (nrec, ntok)
        print("  p%02d tokens=%d records=%d wall=%dms%s"
              % (qi, ntok, nrec, out["wall_ms"], flag), flush=True)
        if nrec != ntok:
            raise SystemExit("record/token count mismatch on p%d" % qi)
        results.append((i, p["q"], p["gold"], out))

    recs = D.analyze("llama3.2-3b", results, args.nstreams)
    for i, r in enumerate(recs):
        p = probs[i]
        r["src"] = p["src"]
        r["qid"] = i + 1
        r["group"] = ("UR" if r["unanimous"] and r["maj_ok"] else
                      "UW" if r["unanimous"] else "SP")
        r["probs_jsonl"] = "_probs_p%02d.jsonl" % (i + 1)
        r["probs_json"] = "_probs_p%02d.json" % (i + 1)
        print("  p%02d %-10s group=%s distinct=%d maj=%s maj_ok=%d "
              "agree=%.2f gold=%s"
              % (i + 1, r["src"], r["group"], r["distinct"], r["maj"],
                 int(r["maj_ok"]), r["pairwise_agree"], r["gold"]))

    counts = {"UR": 0, "SP": 0, "UW": 0}
    for r in recs:
        counts[r["group"]] += 1
    print("groups: UR=%d SP=%d UW=%d" % (counts["UR"], counts["SP"], counts["UW"]))

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    jpath = os.path.join(ROOT, "results", BASE + ".json")
    with open(jpath, "w", encoding="utf-8") as fh:
        json.dump({"n_streams": args.nstreams, "temp": D.TEMP,
                   "top_p": D.TOP_P, "seed": D.SEED,
                   "max_tokens": D.MAX_TOK,
                   "cli": os.path.basename(CLI),
                   "selection": "30 length quantiles (i+0.5)/30 + n100 UW boost",
                   "models": ["llama3.2-3b"], "groups": counts,
                   "records": recs}, fh, indent=1)
    cpath = os.path.join(ROOT, "results", BASE + ".csv")
    cols = ["qid", "src", "group", "gold", "distinct", "maj", "maj_share",
            "pairwise_agree", "ent_conf", "margin", "unanimous", "maj_ok",
            "gold_support", "oracle", "n_answered", "wall_ms", "probs_jsonl"]
    with open(cpath, "w", encoding="utf-8", newline="") as fh:
        fh.write(",".join(cols) + "\n")
        for r in recs:
            fh.write(",".join(str(r[c]) for c in cols) + "\n")
    print("wrote %s\nwrote %s" % (jpath, cpath))


if __name__ == "__main__":
    main()
