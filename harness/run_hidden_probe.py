#!/usr/bin/env python3
"""Row 41: last-layer hidden-state capture probe (parallel engine, record_hidden).

Same 32 problems, model, sampling and answer extraction as row 40
(results/probs_probe30.json), so the two runs are comparable. One invocation per
problem with N=8 streams; prefix_cli --hiddens writes a scalars-only JSONL
sidecar (k=h per stream step, k=c per cross-stream consensus step).

The capture is real engine state: llama_get_embeddings_ith() on the same row
that requested logits (last-layer output after the final norm). No vectors are
stored offline.

Writes:
  data/_hid_pNN.tsv/json/jsonl          per-problem input, CLI JSON, sidecar
  results/hidden_probe30.json           records with the row-40 fields + sidecar

Usage: python harness/run_hidden_probe.py [--nstreams 8] [--limit N] [--dry-run]
"""
import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "harness"))
import run_disagree_probe as D  # noqa: E402
import run_probs_probe as P  # noqa: E402

CLI = os.path.join(ROOT, "app", "prefix_cli.exe")
BIN = os.path.join(ROOT, "build", "cuda-graphs", "bin")
BASE = "hidden_probe30"


def run_one(qi, prompt, nstreams, seed, dry):
    tsv = os.path.join(ROOT, "data", "_hid_p%02d.tsv" % qi)
    out = os.path.join(ROOT, "data", "_hid_p%02d.json" % qi)
    hid = os.path.join(ROOT, "data", "_hid_p%02d.jsonl" % qi)
    with open(tsv, "w", encoding="utf-8") as fh:
        for i in range(nstreams):
            fh.write("%d\t%s\n" % (i + 1, D.esc(prompt)))
    if dry:
        return out, hid, None
    env = dict(os.environ)
    env["PATH"] = BIN + os.pathsep + env.get("PATH", "")
    r = subprocess.run(
        [CLI, "--model", D.MODELS["llama3.2-3b"], "--data", tsv, "--out", out,
         "--mode", "batch", "--max-tokens", str(D.MAX_TOK),
         "--temperature", str(D.TEMP), "--top-p", str(D.TOP_P),
         "--seed", str(seed), "--hiddens", hid],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        raise SystemExit("prefix_cli failed (p%d, rc=%d):\n%s"
                         % (qi, r.returncode, r.stderr[-1200:]))
    with open(out, encoding="utf-8") as fh:
        return out, hid, json.load(fh)


def read_sidecar(path):
    hrows, crows = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            if o["k"] == "h":
                hrows.append(o)
            elif o["k"] == "c":
                crows.append(o)
            else:
                raise SystemExit("bad sidecar row kind %r in %s" % (o["k"], path))
    return hrows, crows


def check_sidecar(qi, out, hrows, crows):
    """Hard invariants; any violation stops the run (no silent bad data)."""
    streams = out["streams"]
    exp_h = sum(st["generated"] for st in streams)
    if len(hrows) != exp_h:
        raise SystemExit("p%d: h rows %d != generated tokens %d" % (qi, len(hrows), exp_h))
    hist = out.get("active_hist") or []
    exp_c = sum(hist[2:]) if len(hist) > 2 else 0
    if len(crows) != exp_c:
        raise SystemExit("p%d: c rows %d != ticks with >=2 live streams %d"
                         % (qi, len(crows), exp_c))
    steps = [r["step"] for r in crows]
    if steps != list(range(len(crows))):
        raise SystemExit("p%d: consensus steps not contiguous from 0: %s"
                         % (qi, steps[:20]))
    if len(crows) != out.get("hidden_consensus"):
        raise SystemExit("p%d: hidden_consensus %s != %d rows"
                         % (qi, out.get("hidden_consensus"), len(crows)))
    by_s = {}
    for r in hrows:
        by_s.setdefault(r["s"], []).append(r)
    for st in streams:
        rs = sorted(by_s.get(st["id"], []), key=lambda x: x["step"])
        if len(rs) != st["generated"]:
            raise SystemExit("p%d stream %s: h rows %d != generated %d"
                             % (qi, st["id"], len(rs), st["generated"]))
        if [r["step"] for r in rs] != list(range(1, st["generated"] + 1)):
            raise SystemExit("p%d stream %s: h steps not contiguous 1..N"
                             % (qi, st["id"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nstreams", type=int, default=D.N)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    probs = P.pick_problems()
    if args.limit:
        probs = probs[:args.limit]
    print("picked %d problems (%d quantile + %d UW-boost):"
          % (len(probs), sum(1 for p in probs if p["src"] == "quantile"),
             sum(1 for p in probs if p["src"] != "quantile")))
    for i, p in enumerate(probs):
        print("  p%02d %-10s gold=%-7s len=%d  %s"
              % (i + 1, p["src"], p["gold"], len(p["q"]), p["q"][:60]))

    est = max(len(p["q"]) for p in probs) // 4 + 40 + D.MAX_TOK
    need = args.nstreams * est
    if need > 8192:
        raise SystemExit("capacity: need ~%d tokens (8192 limit)" % need)
    print("capacity estimate: ~%d/%d tokens per invocation" % (need, 8192))

    if args.dry_run:
        for i, p in enumerate(probs):
            run_one(i + 1, D.INSTR + p["q"], args.nstreams, D.SEED, True)
        print("dry-run ok (no GPU); exiting")
        return

    results, sidecars = [], []
    for i, p in enumerate(probs):
        qi = i + 1
        _, hid, out = run_one(qi, D.INSTR + p["q"], args.nstreams,
                              D.SEED, False)
        hrows, crows = read_sidecar(hid)
        check_sidecar(qi, out, hrows, crows)
        ntok = sum(st["generated"] for st in out["streams"])
        print("  p%02d tokens=%d hrows=%d crows=%d steps=%d wall=%dms"
              % (qi, ntok, len(hrows), len(crows), out["steps"], out["wall_ms"]),
              flush=True)
        results.append((i, p["q"], p["gold"], out))
        sidecars.append((hid, hrows, crows))

    recs = D.analyze("llama3.2-3b", results, args.nstreams)
    for i, r in enumerate(recs):
        p = probs[i]
        r["src"] = p["src"]
        r["qid"] = i + 1
        r["group"] = ("UR" if r["unanimous"] and r["maj_ok"] else
                      "UW" if r["unanimous"] else "SP")
        r["hidden_jsonl"] = "_hid_p%02d.jsonl" % (i + 1)
        r["hidden_json"] = "_hid_p%02d.json" % (i + 1)
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
                   "selection": "30 length quantiles (i+0.5)/30 + n100 UW boost "
                                "(same as probs_probe30)",
                   "capture": "llama_get_embeddings_ith, last-layer embedding "
                              "after final norm, per emitted step",
                   "models": ["llama3.2-3b"], "groups": counts,
                   "records": recs}, fh, indent=1)
    print("wrote %s" % jpath)


if __name__ == "__main__":
    main()
