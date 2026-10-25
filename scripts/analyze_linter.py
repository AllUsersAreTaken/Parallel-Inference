"""Linter / partial-span KV rollback 2x2 summary (stdlib only).

Reads one or more linter CSVs (one row per case x arm), extracts the Python
code from each row's final_text, lints every block with ast, and runs the
generated tests on the last block in a short-lived child process.

Scoring:
  first_import    = the harness linter fired (an import line in actor_text)
  first_tests     = actor's first block passes the tests (imports allowed)
  literal_syntax  = every ```python block in the final artifact concatenates
                    into one parseable module (SyntaxError anywhere => False)
  literal_import  = the concatenated module contains Import/ImportFrom
  literal_tests   = the concatenated module executes and passes the tests
  literal_artifact_correct = literal_syntax and not literal_import and
                    literal_tests                    (PRIMARY metric)
  artifact_import = any code block in the final artifact contains an import
  last_tests      = the last code block executes and passes the tests
  artifact_correct = not artifact_import and last_tests   (secondary metric)

The literal metric is deliberately strict: it scores the whole artifact the
model produced, including any retained prefix before a correction. Both
secondary metrics are kept for continuity with the earlier 2x2 report.

The injected orchestration text is not part of the model's code artifact.
For partial_userturn, full_comment, and full_userturn the exact inject_text is
removed before block extraction. For partial_comment the injected LINT comment
is kept because it supplies the body indentation.

The 2x2 arms are:
  partial_comment, partial_userturn, full_comment, full_userturn
with control and flagonly as baselines.

Statistics:
  - naive paired McNemar exact p and paired bootstrap 95% CI on (case, seed)
  - cluster-aware case-level sign test and case-cluster bootstrap 95% CI
    (the same id appears once per seed, so rows are not independent)

Usage:
    python scripts/analyze_linter.py
    python scripts/analyze_linter.py --csv a.csv,b.csv
    python scripts/analyze_linter.py --csv "results/linter_2x2_*.csv"
    python scripts/analyze_linter.py --csv a.csv,b.csv --enrich-out out.csv
"""
import argparse
import ast
import csv
import glob
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
DEFAULT_CSV = os.path.join(RESULTS, "linter_2x2_bench.csv")
DEFAULT_OUT = os.path.join(RESULTS, "linter_2x2_summary.txt")
DEFAULT_TESTS = os.path.join(ROOT, "data", "linter_tests.json")

ARM_ORDER = ["control", "partial_comment", "partial_userturn", "full_comment",
             "full_userturn", "flagonly", "plain_retry"]
CORRECTION_ARMS = ["partial_comment", "partial_userturn", "full_comment",
                   "full_userturn", "plain_retry"]
PAIRINGS = [
    ("partial_comment", "full_comment"),
    ("partial_userturn", "full_userturn"),
    ("partial_comment", "partial_userturn"),
    ("full_comment", "full_userturn"),
    ("control", "full_userturn"),
    ("flagonly", "full_userturn"),
    ("full_userturn", "plain_retry"),
    ("control", "plain_retry"),
    ("flagonly", "plain_retry"),
]
POLICY_PAIRINGS = [
    ("control", "full_userturn"),
    ("flagonly", "full_userturn"),
    ("full_userturn", "plain_retry"),
    ("control", "plain_retry"),
    ("flagonly", "plain_retry"),
]
INT_FIELDS = ["first_import", "first_tests", "artifact_import", "last_tests",
              "artifact_correct", "char_start", "keep_token", "keep_pos",
              "partial", "map_exact", "rm_ok", "pos_max_before",
              "pos_max_after", "seed", "flagged"]
FLOAT_FIELDS = ["rm_us", "actor_ms", "cont_ms", "total_ms", "prefill_ms",
                "actor_prefill_ms"]
OUT_FIELDS = ["id", "type", "seed", "arm", "actor_text", "cont_text",
              "final_text", "first_import", "first_tests", "artifact_import",
              "last_tests", "artifact_correct", "literal_syntax",
              "literal_import", "literal_tests", "literal_correct",
              "import_pos", "char_start", "keep_token", "keep_pos", "partial",
              "map_exact", "rm_ok", "rm_us", "pos_max_before", "pos_max_after",
              "flagged", "actor_ms", "cont_ms", "total_ms", "template",
              "inject_text", "prefill_ms", "actor_prefill_ms"]

FENCE_ANY_RE = re.compile(r"```")
FENCE_LANG_RE = re.compile(r"(?:python|py)[ \t]*\r?\n?", re.I)
IMPORT_RE = re.compile(r"^\s*(?:import|from)\s+", re.M)

CHILD = r"""
import json, sys
d = json.load(sys.stdin)
ns = {}

def canon(v):
    if isinstance(v, dict):
        return {str(k): canon(x) for k, x in v.items()}
    if isinstance(v, list):
        return [canon(x) for x in v]
    return v

try:
    exec(compile(d["code"], "<artifact>", "exec"), ns)
    fn = ns.get(d["func"])
    if not callable(fn):
        result = {"pass": False, "error": "missing function"}
    else:
        result = {"pass": True, "error": ""}
        for args, exp in d["cases"]:
            got = fn(*args)
            if isinstance(exp, bool):
                ok = isinstance(got, bool) and got == exp
            elif isinstance(exp, float):
                ok = isinstance(got, (int, float)) and abs(got - exp) <= 1e-6
            else:
                ok = canon(got) == canon(exp)
            if not ok:
                result = {"pass": False, "error": "mismatch",
                          "args": args, "got": repr(got), "exp": repr(exp)}
                break
except BaseException as ex:
    result = {"pass": False, "error": type(ex).__name__ + ": " + str(ex)}
print(json.dumps(result))
"""


def resolve_csvs(spec):
    paths = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        hits = glob.glob(part)
        if hits:
            paths.extend(hits)
        else:
            paths.append(part)
    return paths


def load_rows(paths):
    rows = []
    for path in paths:
        with open(path, encoding="utf-8", errors="replace", newline="") as fh:
            for raw in csv.DictReader(fh):
                r = dict(raw)
                for k in INT_FIELDS:
                    try:
                        r[k] = int(float(r.get(k, "-1") or -1))
                    except ValueError:
                        r[k] = -1
                for k in FLOAT_FIELDS:
                    try:
                        r[k] = float(r.get(k, "0") or 0.0)
                    except ValueError:
                        r[k] = 0.0
                if r.get("seed", -1) in (-1, None):
                    r["seed"] = 0
                for k in ("actor_text", "cont_text", "final_text", "inject_text"):
                    r[k] = (r.get(k, "") or "").replace("\r\n", "\n").replace("\r", "\n")
                r["import_pos"] = r.get("import_pos", "") or "na"
                r["inject_text"] = r.get("inject_text", "") or ""
                r["template"] = r.get("template", "") or ""
                r["_src"] = os.path.basename(path)
                rows.append(r)
    return rows


def pct(num, den):
    return 100.0 * num / den if den else float("nan")


def p95(values):
    if not values:
        return float("nan")
    v = sorted(values)
    if len(v) == 1:
        return v[0]
    k = int(math.ceil(0.95 * (len(v) - 1)))
    return v[min(k, len(v) - 1)]


def med(values):
    return statistics.median(values) if values else float("nan")


def short(text, n=900):
    text = (text or "").replace(chr(10), " ")
    return text if len(text) <= n else text[:n] + "..."


def extract_blocks(text):
    """Split fenced code into blocks.

    A line starting with ```python opens a block; a bare ``` closes it. A new
    ```python while a block is open is treated as a fresh opener: this happens
    when a continuation restarts with its own fence inside a retained prefix.
    """
    text = text or ""
    blocks = []
    in_block = False
    content_start = 0
    pos = 0
    while True:
        m = FENCE_ANY_RE.search(text, pos)
        if not m:
            break
        lang = FENCE_LANG_RE.match(text, m.end())
        if lang:
            if in_block:
                blocks.append(text[content_start:m.start()])
            in_block = True
            content_start = lang.end()
            pos = content_start
        else:
            if in_block:
                blocks.append(text[content_start:m.start()])
                in_block = False
            pos = m.end()
    if in_block:
        blocks.append(text[content_start:])
    if not blocks:
        stripped = text.strip()
        return [(stripped, False)] if stripped else [("", False)]
    return [(b, True) for b in blocks]


def import_regex(code):
    return bool(IMPORT_RE.search(code or ""))


def parse_import(code):
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return False, False
    found = any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(tree))
    return True, found


def concat_blocks(blocks):
    """Concatenate every extracted block in order with newlines."""
    return "\n".join(code for code, _ in blocks if code is not None)


def literal_eval(ev, code, cid):
    """Score the concatenated artifact as one module.

    Returns (syntax_ok, has_import, tests_pass, error). A SyntaxError anywhere
    fails the artifact; imports are detected on the parsed AST; tests run on the
    named function after exec of the whole module.
    """
    if not (code or "").strip():
        return False, False, False, "empty artifact"
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError) as ex:
        return False, False, False, "syntax: %s" % ex
    has_import = any(isinstance(n, (ast.Import, ast.ImportFrom))
                     for n in ast.walk(tree))
    ok, err = ev.tests(code, cid)
    return True, has_import, ok, err


def run_child(code, func, cases, timeout=2.0):
    payload = json.dumps({"code": code, "func": func, "cases": cases})
    try:
        proc = subprocess.run([sys.executable, "-I", "-c", CHILD],
                              input=payload, capture_output=True, text=True,
                              timeout=timeout, encoding="utf-8",
                              errors="replace")
    except subprocess.TimeoutExpired:
        return False, "timeout (>%.1fs)" % timeout
    except OSError as ex:
        return False, "spawn failed: %s" % ex
    out = (proc.stdout or "").strip().splitlines()
    if not out:
        return False, "no result (rc=%d)" % proc.returncode
    try:
        d = json.loads(out[-1])
    except ValueError:
        return False, "bad result: %s" % out[-1][:120]
    return bool(d.get("pass")), str(d.get("error", ""))


class Evaluator:
    def __init__(self, specs):
        self.specs = specs
        self.cache = {}
        self.n_child = 0

    def block(self, code):
        syntax, has_import = parse_import(code)
        if not syntax:
            has_import = import_regex(code)
        return syntax, has_import

    def tests(self, code, cid):
        if cid not in self.specs:
            return False, "no spec"
        key = (code, cid)
        if key not in self.cache:
            self.n_child += 1
            self.cache[key] = run_child(code, self.specs[cid]["func"],
                                        self.specs[cid]["cases"])
        return self.cache[key]


def code_source(r):
    text = r.get("final_text", "") or ""
    inj = r.get("inject_text", "") or ""
    if inj and r.get("arm") != "partial_comment" and inj in text:
        text = text.replace(inj, "", 1)
    return text


def analyze(rows, ev):
    for r in rows:
        src = code_source(r)
        final_blocks = extract_blocks(src)
        actor_blocks = extract_blocks(r.get("actor_text", ""))

        artifact_import = False
        for code, _ in final_blocks:
            _, has_import = ev.block(code)
            artifact_import = artifact_import or has_import

        last_code = final_blocks[-1][0] if final_blocks else ""
        last_syntax, _ = ev.block(last_code)
        last_tests, last_err = ev.tests(last_code, r["id"])

        literal_code = concat_blocks(final_blocks)
        lit_syntax, lit_import, lit_tests, lit_err = literal_eval(
            ev, literal_code, r["id"])

        first_code = actor_blocks[0][0] if actor_blocks else ""
        first_syntax, first_import = ev.block(first_code)
        first_tests, first_err = ev.tests(first_code, r["id"])

        r["_code_source"] = src
        r["_n_blocks"] = len(final_blocks)
        r["_artifact_import"] = artifact_import
        r["_last_code"] = last_code
        r["_last_syntax"] = last_syntax
        r["_last_tests"] = last_tests
        r["_last_err"] = last_err
        r["_literal_code"] = literal_code
        r["_literal_syntax"] = lit_syntax
        r["_literal_import"] = lit_import
        r["_literal_tests"] = lit_tests
        r["_literal_err"] = lit_err
        r["_first_code"] = first_code
        r["_first_syntax"] = first_syntax
        r["_first_import"] = first_import
        r["_first_tests"] = first_tests
        r["_first_err"] = first_err
        r["_artifact_correct"] = bool(not artifact_import and last_tests)
        r["_literal_correct"] = bool(lit_syntax and not lit_import and lit_tests)
        r["_repair"] = bool(r.get("first_import") == 1 and
                            r["_literal_correct"])
        r["_hint_lcm"] = "lcm" in (r.get("template", "") or "")

        if first_tests and r["_literal_correct"]:
            r["_lit_transition"] = "pass->pass"
        elif r["_literal_correct"]:
            r["_lit_transition"] = "fail->pass" if r.get("first_import") == 1 \
                else "fail->fail"
        elif first_tests:
            r["_lit_transition"] = "pass->fail"
        else:
            r["_lit_transition"] = "fail->fail"

        if first_tests and r["_artifact_correct"]:
            r["_transition"] = "pass->pass"
        elif r["_artifact_correct"]:
            r["_transition"] = "fail->pass" if r.get("first_import") == 1 \
                else "fail->fail"
        elif first_tests:
            r["_transition"] = "pass->fail"
        else:
            r["_transition"] = "fail->fail"
    return rows


def triggered(rows):
    return [r for r in rows if r.get("first_import") == 1]


def seeds_of(rows):
    return sorted({int(r["seed"]) for r in rows})


def trigger_table(out, rows):
    out("Trigger rates (harness first_import on the actor text)")
    out("=" * 78)
    actors = [r for r in rows if r["arm"] == "control"]
    if not actors:
        actors = rows
    templates = sorted({r.get("template", "") for r in actors})
    for seed in seeds_of(actors):
        srows = [r for r in actors if r["seed"] == seed]
        tempt = [r for r in srows if r["type"] == "temptation"]
        clean = [r for r in srows if r["type"] == "clean"]
        out("seed=%d  temptation=%d  clean=%d" % (seed, len(tempt), len(clean)))
        out("  %-16s %3s %4s %7s" % ("template", "n", "trig", "rate%"))
        for tmpl in templates:
            trs = [r for r in tempt if r.get("template") == tmpl]
            if not trs:
                continue
            k = sum(1 for r in trs if r.get("first_import") == 1)
            out("  %-16s %3d %4d %7.1f" % (tmpl, len(trs), k, pct(k, len(trs))))
        k = sum(1 for r in tempt if r.get("first_import") == 1)
        kc = sum(1 for r in clean if r.get("first_import") == 1)
        out("  %-16s %3d %4d %7.1f" % ("ALL TEMPTATION", len(tempt), k,
                                        pct(k, len(tempt))))
        out("  %-16s %3d %4d %7.1f" % ("ALL CLEAN", len(clean), kc,
                                        pct(kc, len(clean))))
        out("")
    if len(seeds_of(actors)) > 1:
        tempt = [r for r in actors if r["type"] == "temptation"]
        k = sum(1 for r in tempt if r.get("first_import") == 1)
        out("pooled temptation trigger: %d/%d (%.1f%%)"
            % (k, len(tempt), pct(k, len(tempt))))
        out("")


def arm_metrics(rs):
    n = len(rs)
    return {
        "n": n,
        "first_tests": sum(1 for r in rs if r["_first_tests"]),
        "artifact_import": sum(1 for r in rs if r["_artifact_import"]),
        "literal_import": sum(1 for r in rs if r["_literal_import"]),
        "literal_correct": sum(1 for r in rs if r["_literal_correct"]),
        "last_tests": sum(1 for r in rs if r["_last_tests"]),
        "artifact_correct": sum(1 for r in rs if r["_artifact_correct"]),
        "lit_transitions": {k: sum(1 for r in rs if r["_lit_transition"] == k)
                            for k in ("fail->pass", "pass->fail",
                                      "pass->pass", "fail->fail")},
        "transitions": {k: sum(1 for r in rs if r["_transition"] == k)
                        for k in ("fail->pass", "pass->fail",
                                  "pass->pass", "fail->fail")},
    }


def arm_table(out, rows, title, pred=None):
    pred = pred or (lambda r: True)
    out(title)
    hdr = ("%-18s %4s %6s %8s %7s %9s %7s %8s %6s %6s %6s %6s"
           % ("arm", "n", "first%", "lit_imp%", "lit_ok%", "last_tests%",
              "art_ok%", "lit_f->p", "lit_p->f", "lit_p->p", "lit_f->f",
              "flag%"))
    out(hdr)
    out("-" * len(hdr))
    stats = {}
    for arm in ARM_ORDER:
        rs = [r for r in rows if r["arm"] == arm and pred(r)]
        if not rs:
            continue
        m = arm_metrics(rs)
        tr = m["lit_transitions"]
        stats[arm] = m
        out("%-18s %4d %6.1f %8.1f %7.1f %9.1f %7.1f %8d %6d %6d %6d %6.1f"
            % (arm, m["n"], pct(m["first_tests"], m["n"]),
               pct(m["literal_import"], m["n"]),
               pct(m["literal_correct"], m["n"]),
               pct(m["last_tests"], m["n"]),
               pct(m["artifact_correct"], m["n"]),
               tr["fail->pass"], tr["pass->fail"], tr["pass->pass"],
               tr["fail->fail"],
               pct(sum(1 for r in rs if r["flagged"]), m["n"])))
    out("")
    return stats


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * p)


def sign_test(wins, losses):
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    p = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * p)


def bootstrap_diff(pairs, rng, iters=4000):
    if not pairs:
        return (float("nan"), float("nan"), float("nan"))
    diffs = [a - b for a, b in pairs]
    obs = sum(diffs) / len(diffs)
    n = len(diffs)
    boots = []
    for _ in range(iters):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        boots.append(s / n)
    boots.sort()
    lo = boots[int(0.025 * (iters - 1))]
    hi = boots[int(0.975 * (iters - 1))]
    return obs, lo, hi


def cluster_bootstrap(pairs_by_case, rng, iters=4000):
    ids = sorted(pairs_by_case)
    if not ids:
        return (float("nan"), float("nan"), float("nan"))
    obs_num = sum(a - b for cid in ids for a, b in pairs_by_case[cid])
    obs_den = sum(len(pairs_by_case[cid]) for cid in ids)
    obs = obs_num / obs_den if obs_den else float("nan")
    boots = []
    for _ in range(iters):
        num = 0
        den = 0
        for _ in range(len(ids)):
            cid = ids[rng.randrange(len(ids))]
            for a, b in pairs_by_case[cid]:
                num += a - b
                den += 1
        boots.append(num / den if den else 0.0)
    boots.sort()
    lo = boots[int(0.025 * (iters - 1))]
    hi = boots[int(0.975 * (iters - 1))]
    return obs, lo, hi


def paired_rows(rows, arm_a, arm_b, metric_fn, subset=None):
    base = subset if subset is not None else triggered(rows)
    a = {}
    b = {}
    meta = {}
    for r in base:
        if r["arm"] == arm_a:
            a[(r["id"], r["seed"])] = int(bool(metric_fn(r)))
            meta[(r["id"], r["seed"])] = r
        elif r["arm"] == arm_b:
            b[(r["id"], r["seed"])] = int(bool(metric_fn(r)))
    keys = sorted(set(a) & set(b))
    pairs = [(a[k], b[k]) for k in keys]
    return keys, pairs, meta


def pair_summary(rows, arm_a, arm_b, metric_fn, label, subset=None,
                 cluster_key="id", iters=4000):
    keys, pairs, meta = paired_rows(rows, arm_a, arm_b, metric_fn, subset)
    if not pairs:
        return None
    n = len(pairs)
    a_rate = sum(x for x, _ in pairs)
    b_rate = sum(y for _, y in pairs)
    b = sum(1 for x, y in pairs if x and not y)
    c = sum(1 for x, y in pairs if y and not x)
    diff, lo, hi = bootstrap_diff(pairs, random.Random(1234))
    p = mcnemar_exact(b, c)

    by_cluster = {}
    for (cid, seed), (x, y) in zip(keys, pairs):
        if cluster_key == "id":
            group = cid
        else:
            group = (meta[(cid, seed)].get(cluster_key, "") or "")
        by_cluster.setdefault(group, []).append((x, y))
    wins = losses = ties = 0
    cluster_rates_a = []
    cluster_rates_b = []
    for cid in sorted(by_cluster):
        cp = by_cluster[cid]
        ra = sum(x for x, _ in cp) / len(cp)
        rb = sum(y for _, y in cp) / len(cp)
        cluster_rates_a.append(ra)
        cluster_rates_b.append(rb)
        if ra > rb:
            wins += 1
        elif ra < rb:
            losses += 1
        else:
            ties += 1
    cdiff, clo, chi = cluster_bootstrap(by_cluster, random.Random(1234), iters)
    cp = sign_test(wins, losses)
    return {
        "label": label, "n": n, "arm_a": arm_a, "arm_b": arm_b,
        "cluster_key": cluster_key,
        "a_rate": a_rate / n, "b_rate": b_rate / n,
        "diff": diff, "lo": lo, "hi": hi, "p": p, "b": b, "c": c,
        "n_case": len(by_cluster), "wins": wins, "losses": losses, "ties": ties,
        "case_a": statistics.mean(cluster_rates_a),
        "case_b": statistics.mean(cluster_rates_b),
        "cdiff": cdiff, "clo": clo, "chi": chi, "cp": cp,
        "per_seed": {},
    }


def add_per_seed(s, rows, metric_fn, subset=None):
    base = subset if subset is not None else triggered(rows)
    for seed in seeds_of(base):
        keys, pairs, _ = paired_rows(rows, s["arm_a"], s["arm_b"], metric_fn,
                                     [r for r in base if r["seed"] == seed])
        if not pairs:
            continue
        n = len(pairs)
        b = sum(1 for x, y in pairs if x and not y)
        c = sum(1 for x, y in pairs if y and not x)
        diff, lo, hi = bootstrap_diff(pairs, random.Random(1234), 2000)
        s["per_seed"][seed] = {
            "n": n,
            "a_rate": sum(x for x, _ in pairs) / n,
            "b_rate": sum(y for _, y in pairs) / n,
            "diff": diff, "lo": lo, "hi": hi, "p": mcnemar_exact(b, c),
            "b": b, "c": c,
        }
    return s


def pair_table(out, rows, metric_fn, label):
    out(label)
    out("=" * 78)
    hdr = ("%-18s %-18s %4s %7s %7s %7s %15s %9s %4s %4s"
           % ("arm A", "arm B", "n", "A%", "B%", "diff", "naive 95% CI",
              "McNemar p", "b", "c"))
    out(hdr)
    out("-" * len(hdr))
    for arm_a, arm_b in PAIRINGS:
        s = pair_summary(rows, arm_a, arm_b, metric_fn, label)
        if not s:
            continue
        add_per_seed(s, rows, metric_fn)
        out("%-18s %-18s %4d %7.1f %7.1f %+7.1f [%6.1f,%6.1f] %9.4f %4d %4d"
            % (arm_a, arm_b, s["n"], 100.0 * s["a_rate"],
               100.0 * s["b_rate"], 100.0 * s["diff"],
               100.0 * s["lo"], 100.0 * s["hi"], s["p"], s["b"], s["c"]))
    out("")
    out("cluster-aware (same case id grouped across seeds)")
    hdr2 = ("%-18s %-18s %6s %7s %7s %7s %15s %9s %5s %4s %4s"
            % ("arm A", "arm B", "cases", "A%", "B%", "diff",
               "cluster 95% CI", "sign p", "W/L/T", "b", "c"))
    out(hdr2)
    out("-" * len(hdr2))
    for arm_a, arm_b in PAIRINGS:
        s = pair_summary(rows, arm_a, arm_b, metric_fn, label)
        if not s:
            continue
        out("%-18s %-18s %6d %7.1f %7.1f %+7.1f [%6.1f,%6.1f] %9.4f %2d/%1d/%d %4d %4d"
            % (arm_a, arm_b, s["n_case"], 100.0 * s["case_a"],
               100.0 * s["case_b"], 100.0 * s["cdiff"],
               100.0 * s["clo"], 100.0 * s["chi"], s["cp"],
               s["wins"], s["losses"], s["ties"], s["b"], s["c"]))
    out("")
    for arm_a, arm_b in PAIRINGS:
        s0 = pair_summary(rows, arm_a, arm_b, metric_fn, label)
        if not s0:
            continue
        add_per_seed(s0, rows, metric_fn)
        out("per seed: %s vs %s (naive paired rows)" % (arm_a, arm_b))
        for seed in sorted(s0["per_seed"]):
            d = s0["per_seed"][seed]
            out("  seed=%d  n=%d  A=%.1f%%  B=%.1f%%  diff=%+.1fpp "
                "CI[%.1f,%.1f]  p=%.4f  b=%d c=%d"
                % (seed, d["n"], 100.0 * d["a_rate"], 100.0 * d["b_rate"],
                   100.0 * d["diff"], 100.0 * d["lo"], 100.0 * d["hi"],
                   d["p"], d["b"], d["c"]))
        out("")


def template_cluster_table(out, rows, metric_fn, label):
    out(label)
    out("=" * 78)
    out("template-clustered bootstrap (10,000 resamples, seed 1234)")
    hdr = ("%-18s %-18s %7s %7s %7s %15s %9s %7s %7s"
           % ("arm A", "arm B", "templates", "A%", "B%", "diff",
              "cluster 95% CI", "sign p", "W/L/T"))
    out(hdr)
    out("-" * len(hdr))
    for arm_a, arm_b in POLICY_PAIRINGS:
        s = pair_summary(rows, arm_a, arm_b, metric_fn, label,
                         cluster_key="template", iters=10000)
        if not s:
            continue
        out("%-18s %-18s %7d %7.1f %7.1f %+7.1f [%6.1f,%6.1f] %9.4f "
            "%2d/%1d/%d"
            % (arm_a, arm_b, s["n_case"], 100.0 * s["case_a"],
               100.0 * s["case_b"], 100.0 * s["cdiff"],
               100.0 * s["clo"], 100.0 * s["chi"], s["cp"],
               s["wins"], s["losses"], s["ties"]))
    out("")


def gate_report(out, rows, metric_fn):
    def rate(arm, subset=None):
        rs = [r for r in (subset if subset is not None else triggered(rows))
              if r["arm"] == arm]
        k = sum(1 for r in rs if metric_fn(r))
        return k, len(rs), pct(k, len(rs))

    out("FROZEN GATE REPORT (literal whole-artifact metric)")
    out("=" * 78)
    for arm in ("control", "flagonly", "full_userturn", "plain_retry"):
        k, n, r = rate(arm)
        out("  S(%-13s) = %3d/%3d = %5.1f%%" % (arm, k, n, r))
    fu = rate("full_userturn")
    pr = rate("plain_retry")
    gate1a = fu[2] >= 60.0
    gate1b = fu[2] >= pr[2] - 5.0
    out("  gate 1a S(full_userturn) >= 60%%: %s (%.1f%%)"
        % ("PASS" if gate1a else "FAIL", fu[2]))
    out("  gate 1b S(full_userturn) >= S(plain_retry) - 5pp: %s "
        "(%.1f%% vs %.1f%%, delta %+.1fpp)"
        % ("PASS" if gate1b else "FAIL", fu[2], pr[2], fu[2] - pr[2]))
    gate2_rows = []
    for tmpl in sorted({r.get("template", "") for r in triggered(rows)}):
        sub = [r for r in triggered(rows) if r.get("template") == tmpl]
        fk, fn, fr = rate("full_userturn", sub)
        if fn >= 3:
            gate2_rows.append((tmpl, fk, fn, fr, fr >= 20.0))
    if gate2_rows:
        worst = min(gate2_rows, key=lambda x: x[3])
        out("  gate 2 no template with n>=3 below 20%%: %s "
            "(worst %s %d/%d=%.1f%%)"
            % ("PASS" if all(x[4] for x in gate2_rows) else "FAIL",
               worst[0], worst[1], worst[2], worst[3]))
        for tmpl, fk, fn, fr, ok in gate2_rows:
            out("    %-16s %3d/%3d %6.1f%%  %s"
                % (tmpl, fk, fn, fr, "ok" if ok else "FAIL"))
    else:
        out("  gate 2: no template had >=3 triggered rows (not evaluable)")
    dead = fu[2] < 40.0 or pr[2] > fu[2] + 10.0
    out("  gate 3 dead end (S(full_userturn)<40%% or "
        "S(plain_retry)>S(full_userturn)+10pp): %s"
        % ("DEAD END" if dead else "not triggered"))
    if fu[1] < 20:
        out("  NOTE: fewer than 20 triggered rows; result is underpowered.")
    verdict = "PASS" if (gate1a and gate1b and
                         all(x[4] for x in gate2_rows) and not dead) else "FAIL"
    out("  OVERALL: %s" % verdict)
    out("")


def two_by_two_transitions(out, rows):
    out("2x2 transitions on triggered rows (literal whole-artifact metric)")
    out("=" * 78)
    trig = triggered(rows)
    out("  %-8s %-28s %-28s" % ("shape", "comment", "userturn"))
    for shape in ("partial", "full"):
        cells = []
        for turn in ("comment", "userturn"):
            arm = "%s_%s" % (shape, turn)
            rs = [r for r in trig if r["arm"] == arm]
            tr = {k: sum(1 for r in rs if r["_lit_transition"] == k)
                  for k in ("fail->pass", "pass->fail", "pass->pass",
                            "fail->fail")}
            ft = sum(1 for r in rs if r["_first_tests"])
            cells.append("n=%d first_tests=%d %d/%d/%d/%d"
                         % (len(rs), ft, tr["fail->pass"], tr["pass->fail"],
                            tr["pass->pass"], tr["fail->fail"]))
        out("  %-8s %-28s %-28s" % (shape, cells[0], cells[1]))
    out("  order: fail->pass / pass->fail / pass->pass / fail->fail")
    out("")


def transition_compare(out, rows):
    out("Transition counts, literal vs old whole-artifact")
    out("=" * 78)
    trig = triggered(rows)
    out("  %-18s %4s %24s %24s" % ("arm", "n",
                                   "literal f->p/p->f/p->p/f->f",
                                   "artifact f->p/p->f/p->p/f->f"))
    for arm in CORRECTION_ARMS:
        rs = [r for r in trig if r["arm"] == arm]
        if not rs:
            continue

        def counts(key):
            return [sum(1 for r in rs if r[key] == k)
                    for k in ("fail->pass", "pass->fail", "pass->pass",
                              "fail->fail")]

        lt = counts("_lit_transition")
        at = counts("_transition")
        out("  %-18s %4d %24s %24s"
            % (arm, len(rs), "%d/%d/%d/%d" % tuple(lt),
               "%d/%d/%d/%d" % tuple(at)))
    out("")


def two_by_two_grid(out, rows, metric_fn, label):
    out(label)
    out("=" * 78)
    trig = triggered(rows)
    seeds = seeds_of(trig)
    groups = [(("seed", s), [r for r in trig if r["seed"] == s]) for s in seeds]
    groups.append((("pooled", None), trig))
    for (name, seed), rs in groups:
        if not rs:
            continue
        if name == "seed":
            out("seed=%d  triggered rows=%d  unique cases=%d"
                % (seed, len(rs), len({r["id"] for r in rs})))
        else:
            out("pooled  triggered rows=%d  unique cases=%d"
                % (len(rs), len({r["id"] for r in rs})))
        out("  %-10s %20s %20s" % ("", "comment", "userturn"))
        for shape in ("partial", "full"):
            cells = []
            for turn in ("comment", "userturn"):
                arm = "%s_%s" % (shape, turn)
                ars = [r for r in rs if r["arm"] == arm]
                k = sum(1 for r in ars if metric_fn(r))
                cells.append("%d/%d (%.1f%%)" % (k, len(ars), pct(k, len(ars))))
            out("  %-10s %20s %20s" % (shape, cells[0], cells[1]))
        out("")
    out("")


def stratified(out, rows, metric_fn, label):
    out(label)
    out("=" * 78)
    trig = triggered(rows)
    subsets = []
    for pos in ("top", "inside", "after_def"):
        subsets.append(("import_pos=%s" % pos,
                        [r for r in trig if r["import_pos"] == pos]))
    subsets.append(("hint-relevant: lcm", [r for r in trig if r["_hint_lcm"]]))
    for tmpl in sorted({r.get("template", "") for r in trig}):
        subsets.append(("template=%s" % tmpl,
                        [r for r in trig if r.get("template") == tmpl]))
    for name, rs in subsets:
        ids = {r["id"] for r in rs}
        out("%s  rows=%d  cases=%d" % (name, len(rs), len(ids)))
        if not rs:
            out("")
            continue
        out("  %-18s %4s %7s %8s %7s %7s" % ("arm", "n", "first%",
                                              "lit_ok%", "last_t%", "art_ok%"))
        for arm in ARM_ORDER:
            ars = [r for r in rs if r["arm"] == arm]
            if not ars:
                continue
            out("  %-18s %4d %7.1f %8.1f %7.1f %7.1f"
                % (arm, len(ars), pct(sum(1 for r in ars if r["_first_tests"]),
                                      len(ars)),
                   pct(sum(1 for r in ars if r["_literal_correct"]), len(ars)),
                   pct(sum(1 for r in ars if r["_last_tests"]), len(ars)),
                   pct(sum(1 for r in ars if r["_artifact_correct"]),
                       len(ars))))
        for arm_a, arm_b in (("partial_comment", "full_comment"),
                             ("partial_userturn", "full_userturn")):
            s = pair_summary(rows, arm_a, arm_b, metric_fn, label, rs)
            if s:
                out("  %s - %s: %+.1fpp  cluster CI[%+.1f,%+.1f]  "
                    "sign p=%.4f  W/L/T=%d/%d/%d"
                    % (arm_a, arm_b, 100.0 * s["cdiff"],
                       100.0 * s["clo"], 100.0 * s["chi"], s["cp"],
                       s["wins"], s["losses"], s["ties"]))
        out("")
    out("")


def latency_table(out, rows):
    out("Latency on triggered rows (actor_ms shared per case)")
    hdr = ("  %-18s %4s %12s %12s %11s %11s %10s %10s" % (
        "arm", "n", "med_total_ms", "p95_total_ms", "med_cont_ms",
        "med_prefill", "med_actor_pf", "med_rm_us"))
    out(hdr)
    out("  " + "-" * (len(hdr) - 2))
    for arm in ARM_ORDER:
        rs = [r for r in triggered(rows) if r["arm"] == arm]
        if not rs:
            continue
        totals = [r["total_ms"] for r in rs if r["total_ms"] > 0]
        conts = [r["cont_ms"] for r in rs if r["cont_ms"] > 0]
        rms = [r["rm_us"] for r in rs if r["rm_us"] > 0]
        pref = [r.get("prefill_ms", 0.0) for r in rs]
        apf = [r.get("actor_prefill_ms", 0.0) for r in rs]
        out("  %-18s %4d %12.1f %12.1f %11.1f %11.1f %10.1f %10.1f"
            % (arm, len(rs), med(totals), p95(totals), med(conts),
               med(pref), med(apf), med(rms)))
    out("")
    out("med_prefill = fresh re-prefill paid by plain_retry (0 if KV was reused);")
    out("med_actor_pf = original actor prompt prefill, the amount seq_rm avoids.")
    out("")


def paired_latency(out, rows, arm_a, arm_b):
    base = triggered(rows)
    a = {(r["id"], r["seed"]): r for r in base if r["arm"] == arm_a}
    b = {(r["id"], r["seed"]): r for r in base if r["arm"] == arm_b}
    keys = sorted(set(a) & set(b))
    if not keys:
        return
    out("PAIRED retry-path comparison: %s (A) vs %s (B)" % (arm_a, arm_b))
    out("=" * 78)
    out("  paired triggered rows=%d  unique cases=%d"
        % (len(keys), len({k[0] for k in keys})))
    out("  %-16s %10s %10s %12s %12s %10s"
        % ("metric", "A_med", "B_med", "B-A_med", "B-A_mean", "B-A_min"))
    out("  " + "-" * 76)
    for key in ("prefill_ms", "actor_prefill_ms", "cont_ms", "total_ms", "rm_us"):
        diffs = []
        avals = []
        bvals = []
        for cid in keys:
            x = float(a[cid].get(key, 0.0) or 0.0)
            y = float(b[cid].get(key, 0.0) or 0.0)
            avals.append(x)
            bvals.append(y)
            diffs.append(y - x)
        out("  %-16s %10.2f %10.2f %+12.2f %+12.2f %+10.2f"
            % (key, med(avals), med(bvals), med(diffs),
               sum(diffs) / len(diffs), min(diffs)))
    out("  %-16s %10s %10s" % ("metric", "A", "B"))
    for key, label in (("_literal_correct", "literal_correct"),
                       ("_literal_import", "has_import"),
                       ("_literal_tests", "literal_tests"),
                       ("_last_tests", "last_tests")):
        ka = sum(1 for k in keys if a[k][key])
        kb = sum(1 for k in keys if b[k][key])
        out("  %-16s %10d %10d" % (label, ka, kb))
    out("")


def retry_comparison(out, rows):
    if not any(r["arm"] == "plain_retry" for r in rows):
        return
    paired_latency(out, rows, "full_userturn", "plain_retry")
    paired_latency(out, rows, "control", "plain_retry")
    paired_latency(out, rows, "flagonly", "plain_retry")


def show_row(out, r, title):
    out(title)
    out("  id=%s arm=%s template=%s seed=%d import_pos=%s"
        % (r["id"], r["arm"], r["template"], r["seed"], r["import_pos"]))
    out("  first_import=%d first_tests=%d literal_syntax=%d literal_import=%d "
        "literal_tests=%d literal_correct=%d"
        % (r["first_import"], int(r["_first_tests"]),
           int(r["_literal_syntax"]), int(r["_literal_import"]),
           int(r["_literal_tests"]), int(r["_literal_correct"])))
    out("  secondary: artifact_import=%d last_tests=%d artifact_correct=%d "
        "transition=%s"
        % (int(r["_artifact_import"]), int(r["_last_tests"]),
           int(r["_artifact_correct"]), r["_transition"]))
    out("  char_start=%d keep_token=%d keep_pos=%d partial=%d map_exact=%d "
        "rm_ok=%d rm_us=%.1f total_ms=%.1f"
        % (r["char_start"], r["keep_token"], r["keep_pos"], r["partial"],
           r["map_exact"], r["rm_ok"], r["rm_us"], r["total_ms"]))
    if r["_literal_err"]:
        out("  literal_test_error: %s" % short(r["_literal_err"], 240))
    if r["_last_err"]:
        out("  last_test_error: %s" % short(r["_last_err"], 240))
    out("  inject_text:")
    out("    " + (r["inject_text"] or "").replace("\n", "\n    "))
    out("  continuation:")
    out("    " + (r["cont_text"] or "").replace("\n", "\n    "))
    out("  literal artifact (all blocks concatenated):")
    out("    " + (r["_literal_code"] or "").replace("\n", "\n    "))
    out("  last block:")
    out("    " + (r["_last_code"] or "").replace("\n", "\n    "))
    out("")


def examples(out, rows):
    trig = triggered(rows)
    out("Examples (literal whole-artifact metric is primary)")
    out("=" * 78)
    picks = [
        ("partial_comment", True, "partial_comment success"),
        ("partial_comment", False, "partial_comment failure"),
        ("partial_userturn", True, "partial_userturn success"),
        ("partial_userturn", False, "partial_userturn failure"),
        ("full_comment", True, "full_comment success"),
        ("full_comment", False, "full_comment failure"),
        ("full_userturn", True, "full_userturn success"),
        ("full_userturn", False, "full_userturn failure"),
    ]
    used = set()
    for arm, want, title in picks:
        cand = [r for r in trig if r["arm"] == arm and
                bool(r["_literal_correct"]) == want and
                (r["id"], r["seed"]) not in used]
        if not cand:
            out("%s: none observed" % title)
            out("")
            continue
        r = cand[0]
        used.add((r["id"], r["seed"]))
        show_row(out, r, "Example: " + title)


def enrich_csv(path, rows):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=OUT_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({
                "id": r["id"], "type": r["type"], "seed": r["seed"],
                "arm": r["arm"], "actor_text": r.get("actor_text", ""),
                "cont_text": r.get("cont_text", ""),
                "final_text": r.get("final_text", ""),
                "first_import": r["first_import"],
                "first_tests": int(r["_first_tests"]),
                "artifact_import": int(r["_artifact_import"]),
                "last_tests": int(r["_last_tests"]),
                "artifact_correct": int(r["_artifact_correct"]),
                "literal_syntax": int(r["_literal_syntax"]),
                "literal_import": int(r["_literal_import"]),
                "literal_tests": int(r["_literal_tests"]),
                "literal_correct": int(r["_literal_correct"]),
                "import_pos": r["import_pos"],
                "char_start": r["char_start"],
                "keep_token": r["keep_token"],
                "keep_pos": r["keep_pos"],
                "partial": r["partial"],
                "map_exact": r["map_exact"],
                "rm_ok": r["rm_ok"],
                "rm_us": r["rm_us"],
                "pos_max_before": r["pos_max_before"],
                "pos_max_after": r["pos_max_after"],
                "flagged": r["flagged"],
                "actor_ms": r["actor_ms"],
                "cont_ms": r["cont_ms"],
                "total_ms": r["total_ms"],
                "template": r["template"],
                "inject_text": r["inject_text"],
                "prefill_ms": r.get("prefill_ms", 0.0),
                "actor_prefill_ms": r.get("actor_prefill_ms", 0.0),
            })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV,
                    help="one CSV, comma-separated list, or glob")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--enrich-out", default="",
                    help="write a CSV with computed test fields")
    ap.add_argument("--tests", default=DEFAULT_TESTS)
    ap.add_argument("--cases", default="",
                    help="optional comma-separated case ids")
    args = ap.parse_args()

    paths = resolve_csvs(args.csv)
    for path in paths:
        if not os.path.exists(path):
            raise SystemExit("missing input: %s" % path)
    with open(args.tests, encoding="utf-8") as fh:
        specs = json.load(fh)
    rows = load_rows(paths)
    if not rows:
        raise SystemExit("no rows in %s" % args.csv)
    if args.cases:
        keep = {c.strip() for c in args.cases.split(",") if c.strip()}
        rows = [r for r in rows if r["id"] in keep]
        if not rows:
            raise SystemExit("--cases selected no rows")
    ev = Evaluator(specs)
    rows = analyze(rows, ev)

    lines = []

    def out(s=""):
        lines.append(s)
        print(s)

    out("RecycleLM linter 2x2 correction experiment")
    out("=" * 78)
    out("inputs: " + ", ".join(os.path.relpath(p, ROOT) for p in paths))
    out("tests:  " + os.path.relpath(args.tests, ROOT))
    ids = {r["id"] for r in rows}
    n_t = len({r["id"] for r in rows if r["type"] == "temptation"})
    n_k = len({r["id"] for r in rows if r["type"] == "clean"})
    seeds = seeds_of(rows)
    arms = [a for a in ARM_ORDER if any(r["arm"] == a for r in rows)]
    out("rows: %d  cases: %d  temptation: %d  clean: %d"
        % (len(rows), len(ids), n_t, n_k))
    out("seeds: " + ", ".join(str(s) for s in seeds))
    out("arms: " + ", ".join(arms))
    out("test subprocesses: %d (unique block,case)" % ev.n_child)
    out("")
    out("PRIMARY metric: literal_artifact_correct = every ```python block in")
    out("final_text concatenated in order parses as one module, has no")
    out("Import/ImportFrom anywhere, and passes the tests on the named function.")
    out("SECONDARY metrics: last-block tests pass and old whole-artifact check.")
    out("")

    trigger_table(out, rows)

    arm_table(out, rows, "Per-arm metrics (all cases)")
    arm_table(out, rows, "Triggered cases only",
              pred=lambda r: r.get("first_import") == 1)
    arm_table(out, rows, "Clean cases only (no linter trigger)",
              pred=lambda r: r["type"] == "clean")
    out("Triggered rows are the linter positives. first% = actor's first block")
    out("passes tests (imports allowed). lit_ok% = literal whole-artifact")
    out("correctness (primary). art_ok% = no block imports and the last block")
    out("passes (old secondary metric). Transition columns are literal")
    out("fail->pass, pass->fail, pass->pass, fail->fail on triggered rows.")
    out("flag% = linter fired.")
    out("")

    two_by_two_grid(out, rows, lambda r: r["_literal_correct"],
                    "LITERAL whole-artifact correctness (primary)")
    two_by_two_grid(out, rows, lambda r: r["_artifact_correct"],
                    "Old whole-artifact correctness (secondary)")
    two_by_two_grid(out, rows, lambda r: r["_last_tests"],
                    "Last-block tests pass (secondary)")
    two_by_two_transitions(out, rows)
    transition_compare(out, rows)

    pair_table(out, rows, lambda r: r["_literal_correct"],
               "PAIRED: literal whole-artifact correctness (primary)")
    pair_table(out, rows, lambda r: r["_artifact_correct"],
               "PAIRED: old whole-artifact correctness (secondary)")
    pair_table(out, rows, lambda r: r["_last_tests"],
               "PAIRED: last-block tests pass (secondary)")
    pair_table(out, rows, lambda r: r["_lit_transition"] == "fail->pass",
               "PAIRED: literal fail->pass (first block failed -> literal correct)")

    stratified(out, rows, lambda r: r["_literal_correct"],
               "STRATIFIED literal whole-artifact correctness (primary)")
    has_plain = any(r["arm"] == "plain_retry" for r in rows)
    if has_plain:
        template_cluster_table(out, rows, lambda r: r["_literal_correct"],
                               "TEMPLATE-CLUSTERED literal correctness")
        if len({r.get("template", "") for r in rows}) >= 20:
            gate_report(out, rows, lambda r: r["_literal_correct"])

    latency_table(out, rows)
    retry_comparison(out, rows)
    examples(out, rows)

    out("Verdict (literal whole-artifact metric)")
    out("=" * 78)
    trig = triggered(rows)
    for arm in ARM_ORDER:
        rs = [r for r in trig if r["arm"] == arm]
        if not rs:
            continue
        m = arm_metrics(rs)
        out("  %-18s triggered=%3d literal_correct=%3d (%5.1f%%) "
            "artifact_correct=%3d (%5.1f%%) last_tests=%3d (%5.1f%%)"
            % (arm, m["n"], m["literal_correct"],
               pct(m["literal_correct"], m["n"]), m["artifact_correct"],
               pct(m["artifact_correct"], m["n"]), m["last_tests"],
               pct(m["last_tests"], m["n"])))
    s_pc = pair_summary(rows, "partial_comment", "full_comment",
                        lambda r: r["_literal_correct"], "verdict")
    s_pu = pair_summary(rows, "partial_userturn", "full_userturn",
                        lambda r: r["_literal_correct"], "verdict")
    if s_pc:
        out("  retained prefix, comment: %+.1fpp (cluster sign p=%.4f)"
            % (100.0 * s_pc["cdiff"], s_pc["cp"]))
    if s_pu:
        out("  retained prefix, userturn: %+.1fpp (cluster sign p=%.4f)"
            % (100.0 * s_pu["cdiff"], s_pu["cp"]))
    out("")

    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote " + os.path.relpath(args.out, ROOT))

    if args.enrich_out:
        enrich_csv(args.enrich_out, rows)
        print("wrote " + os.path.relpath(args.enrich_out, ROOT))


if __name__ == "__main__":
    main()
