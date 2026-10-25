"""Spider text-to-SQL bake-off on Qwen2.5-Coder-3B-Instruct.

Arms (100 dev questions, temp 0.6, 128 tokens):
  base1        one sample of the base model
  ad_<name>    one sample of each of 4 SQL adapters
  seed8        8 samples of the base (seed ensemble)
  adapt8       4 adapters x 2 samples (adapter ensemble)

Combiner for the ensembles: execution consensus -- run every candidate against
the database, cluster by normalised result set, take the largest cluster.
That is a *semantic* vote: different SQL strings with the same result are
equivalent by construction.
"""
import io
import json
import os
import random
import re
import sqlite3
import subprocess
import sys

ROOT = r"S:\Agent\QSExplore\RecycleLM"
CLI = os.path.join(ROOT, "app", "prefix_cli.exe")
BIN = os.path.join(ROOT, "build", "cuda-graphs", "bin")
MODEL = os.path.join(ROOT, "models", "qwen2.5-coder-3b-instruct-q4_k_m.gguf")
SP = os.path.join(ROOT, "data", "_spider", "spider_data")
OUTDIR = os.path.join(ROOT, "results", "_sql")
os.makedirs(OUTDIR, exist_ok=True)

SYSTEM = "You are a SQL expert. Given a table schema and a question, reply with only the SQL query."
ADAPTERS = {
    "varshitha": os.path.join(ROOT, "data", "_hf_lora", "sql_varshitha.gguf"),
    "amer": os.path.join(ROOT, "data", "_hf_lora", "sql_amer.gguf"),
    "korg": os.path.join(ROOT, "data", "_hf_lora", "sql_korg.gguf"),
    "khanh": os.path.join(ROOT, "data", "_hf_lora", "sql_khanh.gguf"),
}
NQ = 100
MAXTOK = 128
TEMP = 0.6


def esc(s):
    return s.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")


def load():
    dev = json.load(open(os.path.join(SP, "dev.json"), encoding="utf-8"))
    tables = json.load(open(os.path.join(SP, "tables.json"), encoding="utf-8"))
    tmap = {t["db_id"]: t for t in tables}
    random.Random(0).shuffle(dev)
    return dev[:NQ], tmap


def schema_for(db_id, tmap):
    t = tmap[db_id]
    names = t["table_names_original"]
    cols = t["column_names_original"]
    types = t["column_types"]
    by = {}
    for (ti, cn), ct in zip(cols, types):
        if ti < 0:
            continue
        by.setdefault(names[ti], []).append("%s %s" % (cn, "INTEGER" if ct == "number" else "TEXT"))
    return "\n".join("CREATE TABLE %s (%s);" % (tn, ", ".join(cs)) for tn, cs in by.items())


def extract_sql(text):
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, re.S | re.I)
    s = m.group(1) if m else text
    m2 = re.search(r"(?is)\b(select|with)\b", s)
    if m2:
        s = s[m2.start():]
    s = s.strip()
    # cut at a trailing prose line
    lines = [l for l in s.splitlines() if not re.match(r"^\s*(here|this|the above|note|explanation)", l, re.I)]
    s = "\n".join(lines).strip().rstrip(";").strip()
    return s


def norm_cell(v):
    if isinstance(v, float):
        return round(v, 6)
    if isinstance(v, str):
        return v.strip()
    return v


def exec_sql(db_id, sql):
    if not sql or not re.search(r"(?is)^\s*(select|with)\b", sql):
        return None
    path = os.path.join(SP, "database", db_id, db_id + ".sqlite")
    if not os.path.exists(path):
        return None
    try:
        con = sqlite3.connect("file:%s?mode=ro" % path.replace("\\", "/"), uri=True, timeout=3)
        steps = [0]
        def handler():
            steps[0] += 1
            return 1 if steps[0] > 20000 else 0
        con.set_progress_handler(handler, 1000)
        rows = con.execute(sql).fetchall()
        con.close()
        out = tuple(sorted(tuple(norm_cell(v) for v in r) for r in rows))
        return out
    except Exception:
        return None


def run_batch(prompts, n_per_prompt, lora, tag, chunk_q=8):
    """prompts: list of (db_id, user_prompt). Returns list of candidate lists."""
    env = dict(os.environ); env["PATH"] = BIN + os.pathsep + env.get("PATH", "")
    out = []
    per_call = max(1, chunk_q // max(1, n_per_prompt)) if n_per_prompt > 1 else chunk_q
    for start in range(0, len(prompts), per_call):
        batch = prompts[start:start + per_call]
        rows = []
        for q in batch:
            for _ in range(n_per_prompt):
                rows.append(q[1])
        tsv = os.path.join(OUTDIR, "_b.tsv")
        with open(tsv, "w", encoding="utf-8") as fh:
            for i, p in enumerate(rows):
                fh.write("%d\t%s\n" % (i + 1, esc(p)))
        jout = os.path.join(OUTDIR, "_b.json")
        cmd = [CLI, "--model", MODEL, "--data", tsv, "--out", jout, "--mode", "batch",
               "--max-tokens", str(MAXTOK), "--temperature", str(TEMP), "--top-p", "0.95",
               "--seed", "1234", "--system", SYSTEM]
        if lora:
            cmd += ["--lora", lora]
        r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800)
        if r.returncode != 0:
            raise SystemExit("%s failed: %s" % (tag, r.stderr[-400:]))
        d = json.load(open(jout, encoding="utf-8"))
        texts = [s["text"] for s in d["streams"]]
        for qi in range(len(batch)):
            out.append(texts[qi * n_per_prompt:(qi + 1) * n_per_prompt])
    return out


def same_result(a, b):
    if a is None or b is None:
        return False
    if a == b:
        return True
    na = len(a[0]) if a else 0
    nb = len(b[0]) if b else 0
    if na != nb or len(a) != len(b) or na > 4:
        return False
    from itertools import permutations
    for perm in permutations(range(na)):
        if tuple(sorted(tuple(r[i] for i in perm) for r in a)) == b:
            return True
    return False


def score_candidates(cands, db_id, gold):
    res = [exec_sql(db_id, extract_sql(t)) for t in cands]
    ok = [same_result(r, gold) for r in res]
    return res, ok


def consensus(res):
    groups = {}
    for r in res:
        if r is None:
            continue
        groups.setdefault(r, 0)
        groups[r] += 1
    if not groups:
        return None
    return max(groups.items(), key=lambda kv: (kv[1], -len(kv[0])))[0]


def main():
    qs, tmap = load()
    gold = []
    for q in qs:
        gold.append(exec_sql(q["db_id"], q["query"]))
    prompts = [(q["db_id"], "Schema:\n%s\n\nQuestion: %s" % (schema_for(q["db_id"], tmap), q["question"])) for q in qs]
    print("questions", len(qs), "dbs", len({q['db_id'] for q in qs}), "gold_exec", sum(g is not None for g in gold))

    arms = {}

    def cached(name, fn):
        path = os.path.join(OUTDIR, "arm_%s.json" % name)
        if os.path.exists(path):
            print("  [cached]", name, flush=True)
            return json.load(open(path, encoding="utf-8"))
        v = fn()
        json.dump(v, open(path, "w"), indent=0)
        print("  [done]", name, flush=True)
        return v

    arms["base1"] = cached("base1", lambda: run_batch(prompts, 1, None, "base1"))
    for name, path in ADAPTERS.items():
        arms["ad_" + name] = cached("ad_" + name, lambda name=name, path=path: run_batch(prompts, 1, path, "ad_" + name))
    arms["seed8"] = cached("seed8", lambda: run_batch(prompts, 8, None, "seed8", chunk_q=8))

    def ens_arm():
        ens = [[] for _ in qs]
        for name, path in ADAPTERS.items():
            per = cached("ens_" + name, lambda name=name, path=path: run_batch(prompts, 2, path, "ens_" + name, chunk_q=8))
            for i in range(len(qs)):
                ens[i].extend(per[i])
        return ens

    arms["adapt8"] = cached("adapt8", ens_arm)

    report = {"n": len(qs), "temp": TEMP, "arms": {}}
    for name, cand in arms.items():
        acc = 0
        oracle = 0
        for i in range(len(qs)):
            res, ok = score_candidates(cand[i], qs[i]["db_id"], gold[i])
            acc += bool(ok and ok[0])
            oracle += any(ok)
            if len(cand[i]) > 1:
                c = consensus(res)
                report["arms"].setdefault(name + "_cons", {"acc": 0, "oracle": 0})
                report["arms"][name + "_cons"]["acc"] += bool(same_result(c, gold[i]))
                report["arms"][name + "_cons"]["oracle"] += any(ok)
        report["arms"].setdefault(name, {"acc": 0, "oracle": 0})
        report["arms"][name]["acc"] += acc
        report["arms"][name]["oracle"] += oracle

    n = len(qs)
    print("\n%-18s %8s %8s" % ("arm", "pass@1", "oracle"))
    for name in arms:
        a = report["arms"][name]
        print("%-18s %7.1f%% %7.1f%%" % (name, 100 * a["acc"] / n, 100 * a["oracle"] / n))
        c = report["arms"].get(name + "_cons")
        if c:
            print("%-18s %7.1f%% %7.1f%%" % (name + " consensus", 100 * c["acc"] / n, 100 * c["oracle"] / n))
    json.dump(report, open(os.path.join(ROOT, "results", "sql_ensemble.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
