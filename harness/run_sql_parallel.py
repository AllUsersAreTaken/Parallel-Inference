"""Per-row (segmented) LoRA decode on the SQL expert panel.

Reuses the Spider loading, prompt building and execution-consensus code from
run_sql_ensemble.py. Three jobs:

  global  reproduce five questions of the old base1 / ad_korg arms byte-for-byte
  smoke   2 adapters x 2 rows in one segmented decode vs the same rows run with
          a single global adapter (exact-match report)
  panel   arm S = 4 sequential adapter invocations per question; arm P = one
          invocation with 8 segmented rows (4 adapters x 2 seeds). Reports wall,
          speedup, VRAM, per-row exact match and execution-consensus accuracy.

Rows are always ordered contiguously by adapter: [a0, a0, a1, a1, ...].
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time

import run_sql_ensemble as sq

OUTDIR = os.path.join(sq.ROOT, "results", "_sql_parallel")
os.makedirs(OUTDIR, exist_ok=True)

NAMES = list(sq.ADAPTERS)          # varshitha, amer, korg, khanh
PATHS = [sq.ADAPTERS[n] for n in NAMES]

ENV = dict(os.environ)
ENV["PATH"] = sq.BIN + os.pathsep + ENV.get("PATH", "")


class Vram:
    def __init__(self):
        self.peak = 0
        self.stop = threading.Event()
        self.th = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop.is_set():
            try:
                r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                                    "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=5)
                v = int(r.stdout.strip().splitlines()[0])
                self.peak = max(self.peak, v)
            except Exception:
                pass
            time.sleep(0.25)

    def __enter__(self):
        self.th.start()
        return self

    def __exit__(self, *a):
        self.stop.set()
        self.th.join(timeout=5)


_counter = [0]


def run_cli(rows, lora_paths, per_row=None, seeds=None, timeout=1800):
    """rows: list of prompt strings. Returns parsed JSON + process wall ms."""
    _counter[0] += 1
    tag = "%d_%d" % (os.getpid(), _counter[0])
    tsv = os.path.join(OUTDIR, "_rows_%s.tsv" % tag)
    jout = os.path.join(OUTDIR, "_rows_%s.json" % tag)
    with open(tsv, "w", encoding="utf-8") as fh:
        for i, p in enumerate(rows):
            fh.write("%d\t%s\n" % (i + 1, sq.esc(p)))

    cmd = [sq.CLI, "--model", sq.MODEL, "--data", tsv, "--out", jout,
           "--mode", "batch", "--max-tokens", str(sq.MAXTOK),
           "--temperature", str(sq.TEMP), "--top-p", "0.95",
           "--seed", "1234", "--system", sq.SYSTEM]
    for p in lora_paths:
        cmd += ["--lora", p]
    if per_row is not None:
        cmd += ["--lora-per-row", ",".join(str(x) for x in per_row)]
    if seeds is not None:
        cmd += ["--seed-per-row", ",".join(str(x) for x in seeds)]

    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=sq.ROOT, env=ENV, capture_output=True,
                       text=True, timeout=timeout)
    wall = (time.perf_counter() - t0) * 1000.0
    if r.returncode != 0:
        raise SystemExit("prefix_cli failed:\n%s\n%s" % (r.stdout[-800:], r.stderr[-800:]))
    d = json.load(open(jout, encoding="utf-8"))
    try:
        os.remove(tsv)
        os.remove(jout)
    except OSError:
        pass
    return d, wall


def load_questions(n):
    qs, tmap = sq.load()
    qs = qs[:n]
    prompts = [(q["db_id"],
                "Schema:\n%s\n\nQuestion: %s" % (sq.schema_for(q["db_id"], tmap), q["question"]))
               for q in qs]
    return qs, prompts


def gold_for(qs):
    return [sq.exec_sql(q["db_id"], q["query"]) for q in qs]


def consensus_and_acc(db_id, gold, cand):
    res = [sq.exec_sql(db_id, sq.extract_sql(t)) for t in cand]
    c = sq.consensus(res)
    return sq.same_result(c, gold), res


def job_global(nq):
    # reproduce the original batching: base1 chunks 8 questions, adapter arms
    # chunk 4 questions x 2 seeds. Compare only the first nq questions.
    qs, prompts = load_questions(max(nq, 8))

    def cmp(name, fresh, cached, per_q):
        eq = []
        for i in range(min(per_q, len(fresh))):
            for k in range(len(cached[i])):
                eq.append(fresh[i][k] == cached[i][k])
        print("  %-12s rows=%d exact=%d/%d" % (name, len(eq), sum(eq), len(eq)))
        return eq

    d, w = run_cli([p for _, p in prompts[:8]], [], per_row=None)
    base_fresh = [[s["text"]] for s in d["streams"]]
    cached = json.load(open(os.path.join(sq.OUTDIR, "arm_base1.json"), encoding="utf-8"))
    cmp("base1", base_fresh, cached, nq)

    ad_fresh = []
    for start in range(0, nq, 4):
        chunk = prompts[start:start + 4]
        d, w = run_cli([p for _, p in chunk for _ in range(2)], [sq.ADAPTERS["korg"]])
        for qi in range(len(chunk)):
            ad_fresh.append([s["text"] for s in d["streams"][2 * qi:2 * qi + 2]])

    cached = json.load(open(os.path.join(sq.OUTDIR, "arm_ens_korg.json"), encoding="utf-8"))
    cmp("ens_korg", ad_fresh, cached, nq)


def job_smoke():
    qs, prompts = load_questions(1)
    prompt = prompts[0][1]
    rows = [prompt] * 4
    seeds = [1234, 1235, 1234, 1235]
    ads = [0, 0, 1, 1]
    loras = [sq.ADAPTERS["korg"], sq.ADAPTERS["amer"]]

    d, w = run_cli(rows, loras, per_row=ads, seeds=seeds)
    seg = [s["text"] for s in d["streams"]]
    print("segmented wall %.0f ms" % w)
    print("lora_per_row", d.get("lora_per_row"), "seed_per_row", d.get("seed_per_row"),
          "row_order", d.get("row_order"), "reordered", d.get("reordered"))

    eq = []
    for ai, name in enumerate(["korg", "amer"]):
        d1, w1 = run_cli([prompt] * 2, [sq.ADAPTERS[name]])
        one = [s["text"] for s in d1["streams"]]
        for k in range(2):
            same = one[k] == seg[2 * ai + k]
            eq.append(same)
            print("  %-5s seed %d exact=%s len=%d/%d" %
                  (name, seeds[2 * ai + k], same, len(one[k]), len(seg[2 * ai + k])))
    print("smoke exact %d/%d" % (sum(eq), len(eq)))
    json.dump({"exact": eq, "segmented": seg}, open(os.path.join(sq.ROOT, "results", "sql_parallel_smoke.json"), "w"), indent=1)


def run_panel(nq, tag):
    qs, prompts = load_questions(nq)
    gold = gold_for(qs)

    s_texts, p_texts = [], []
    s_wall = 0.0
    p_wall = 0.0

    with Vram() as vram_s:
        for i, (db, p) in enumerate(prompts):
            per = []
            for ai, path in enumerate(PATHS):
                d, w = run_cli([p, p], [path])
                s_wall += w
                per.extend([s["text"] for s in d["streams"]])
            s_texts.append(per)
            if (i + 1) % 10 == 0:
                print("  S %d/%d wall %.1f s" % (i + 1, nq, s_wall / 1000.0), flush=True)

    with Vram() as vram_p:
        for i, (db, p) in enumerate(prompts):
            rows = [p] * 8
            d, w = run_cli(rows, PATHS, per_row=[0, 0, 1, 1, 2, 2, 3, 3],
                           seeds=[1234, 1235] * 4)
            p_wall += w
            p_texts.append([s["text"] for s in d["streams"]])
            if (i + 1) % 10 == 0:
                print("  P %d/%d wall %.1f s" % (i + 1, nq, p_wall / 1000.0), flush=True)

    # per-row exact match
    exact = 0
    total = 0
    for i in range(nq):
        for k in range(8):
            exact += s_texts[i][k] == p_texts[i][k]
            total += 1

    # execution consensus
    def arm_acc(texts):
        acc = 0
        pools = []
        for i in range(nq):
            ok, res = consensus_and_acc(qs[i]["db_id"], gold[i], texts[i])
            acc += bool(ok)
            pools.append(res)
        return acc, pools

    s_acc, s_res = arm_acc(s_texts)
    p_acc, p_res = arm_acc(p_texts)

    out = {
        "n": nq,
        "temp": sq.TEMP,
        "adapters": NAMES,
        "seeds": [1234, 1235],
        "wall_s": {"sequential": s_wall / 1000.0, "parallel": p_wall / 1000.0,
                   "speedup": s_wall / p_wall if p_wall else 0.0},
        "vram_mb": {"sequential": vram_s.peak, "parallel": vram_p.peak},
        "row_exact": {"match": exact, "total": total, "rate": exact / total if total else 0.0},
        "consensus": {"sequential": s_acc, "parallel": p_acc,
                      "sequential_pct": 100.0 * s_acc / nq, "parallel_pct": 100.0 * p_acc / nq},
        "sequential": s_texts,
        "parallel": p_texts,
    }
    path = os.path.join(sq.ROOT, "results", "sql_parallel_%s.json" % tag)
    json.dump(out, open(path, "w"), indent=1)
    print(json.dumps({k: out[k] for k in
                      ("wall_s", "vram_mb", "row_exact", "consensus")}, indent=1))
    print("wrote", path)


def job_panel_cached(nq, tag):
    """P with the seed schedule of the cached adapt8 arm; S = cached arm texts."""
    qs, prompts = load_questions(nq)
    gold = gold_for(qs)

    cached_arms = {name: json.load(open(os.path.join(sq.OUTDIR, "arm_ens_%s.json" % name),
                                        encoding="utf-8")) for name in NAMES}
    s_texts = []
    for i in range(nq):
        row = []
        for name in NAMES:
            row.extend(cached_arms[name][i][:2])
        s_texts.append(row)

    base = json.load(open(os.path.join(sq.ROOT, "results", "sql_parallel_panel.json"),
                            encoding="utf-8"))
    s_wall = base["wall_s"]["sequential"]
    s_vram = base["vram_mb"]["sequential"]

    p_texts = []
    p_wall = 0.0
    with Vram() as vram_p:
        for i, (db, p) in enumerate(prompts):
            j = i % 4
            seeds = [1234 + 2 * j, 1235 + 2 * j] * 4
            d, w = run_cli([p] * 8, PATHS, per_row=[0, 0, 1, 1, 2, 2, 3, 3], seeds=seeds)
            p_wall += w
            p_texts.append([s["text"] for s in d["streams"]])
            if (i + 1) % 10 == 0:
                print("  P %d/%d wall %.1f s" % (i + 1, nq, p_wall / 1000.0), flush=True)

    exact = sum(s_texts[i][k] == p_texts[i][k] for i in range(nq) for k in range(8))

    def arm_acc(texts):
        acc = 0
        for i in range(nq):
            ok, _ = consensus_and_acc(qs[i]["db_id"], gold[i], texts[i])
            acc += bool(ok)
        return acc

    s_acc = arm_acc(s_texts)
    p_acc = arm_acc(p_texts)
    out = {
        "n": nq,
        "seed_schedule": "cached adapt8 (1234+2*(i%4), 1235+2*(i%4))",
        "adapters": NAMES,
        "wall_s": {"sequential": s_wall, "parallel": p_wall / 1000.0,
                   "speedup": s_wall / (p_wall / 1000.0) if p_wall else 0.0},
        "vram_mb": {"sequential": s_vram, "parallel": vram_p.peak},
        "row_exact": {"match": exact, "total": nq * 8, "rate": exact / (nq * 8)},
        "consensus": {"sequential": s_acc, "parallel": p_acc,
                      "sequential_pct": 100.0 * s_acc / nq, "parallel_pct": 100.0 * p_acc / nq},
        "sequential": s_texts,
        "parallel": p_texts,
    }
    path = os.path.join(sq.ROOT, "results", "sql_parallel_%s.json" % tag)
    json.dump(out, open(path, "w"), indent=1)
    print(json.dumps({k: out[k] for k in
                      ("wall_s", "vram_mb", "row_exact", "consensus")}, indent=1))
    print("wrote", path)


def job_s_chunked(nq, tag):
    """Throughput-optimal sequential schedule: 4 questions x 2 seeds per adapter pass."""
    qs, prompts = load_questions(nq)
    cached_arms = {name: json.load(open(os.path.join(sq.OUTDIR, "arm_ens_%s.json" % name),
                                        encoding="utf-8")) for name in NAMES}
    texts = [[] for _ in range(nq)]
    wall = 0.0
    with Vram() as vram:
        for start in range(0, nq, 4):
            chunk = prompts[start:start + 4]
            for name in NAMES:
                rows = [p for _, p in chunk for _ in range(2)]
                d, w = run_cli(rows, [sq.ADAPTERS[name]])
                wall += w
                for qi in range(len(chunk)):
                    texts[start + qi].extend([s["text"] for s in d["streams"][2 * qi:2 * qi + 2]])
            if (start + 4) % 40 == 0:
                print("  chunk %d/%d wall %.1f s" % (start + 4, nq, wall / 1000.0), flush=True)

    exact = 0
    total = 0
    for i in range(nq):
        j = 0
        for name in NAMES:
            for k in range(2):
                exact += texts[i][j] == cached_arms[name][i][k]
                total += 1
                j += 1

    out = {"n": nq, "wall_s": wall / 1000.0, "vram_mb": vram.peak,
           "exact_vs_cached": {"match": exact, "total": total}}
    path = os.path.join(sq.ROOT, "results", "sql_parallel_%s.json" % tag)
    json.dump(out, open(path, "w"), indent=1)
    print(json.dumps(out, indent=1))
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("job", choices=["global", "smoke", "panel", "panel-cached", "s-chunked"])
    ap.add_argument("--nq", type=int, default=100)
    ap.add_argument("--tag", default="panel")
    a = ap.parse_args()
    if a.job == "global":
        job_global(a.nq)
    elif a.job == "smoke":
        job_smoke()
    elif a.job == "panel-cached":
        job_panel_cached(a.nq, a.tag)
    elif a.job == "s-chunked":
        job_s_chunked(a.nq, a.tag)
    else:
        run_panel(a.nq, a.tag)


if __name__ == "__main__":
    main()
