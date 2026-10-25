r"""Cross-substrate (text-CoT vs executable-Python) ensembling probe, v2.

Reads the clean, escaped stream dumps written by harness/engine/modality.cpp:

    results/m2_<model>_mixed.tsv   8 cot + 8 code streams (N=16)
    results/m2_<model>_cot16.tsv   16 cot streams
    results/m2_<model>_code16.tsv  16 code streams

TSV columns: id, stream, mode, finish, generated, ttft_ms, done_ms, text
(text is escaped: \\ -> \, \\n -> newline, \\t -> tab, \\r -> CR)

The analyzer:
  * extracts CoT answers (boxed -> answer-is -> last number)
  * extracts and executes the last python/py fenced block with
    `python -I -S`, 8 s timeout, temp cwd, minimal env
  * builds C8/P8/M16_pool/M16_gate (plus C16/P16 when homogeneous runs exist)
  * reports tie-aware plurality metrics
  * conditions wrong-answer overlap on both substrates answering, and labels
    abstention separately instead of counting it as wrong
  * runs exact McNemar and a paired 10k bootstrap (seed 1234), 95% CI

Usage:
    python harness/analyze_modality_vote.py \
      --run coder15b,results/m2_coder15b_mixed.tsv,results/m2_coder15b_cot16.tsv,results/m2_coder15b_code16.tsv \
      --run 1b,results/m2_1b_mixed.tsv,results/m2_1b_cot16.tsv,results/m2_1b_code16.tsv

A single group can also be given with --label/--mixed/--cot16/--code16.
"""
import argparse
import ast
import csv
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
DEFAULT_GOLD = os.path.join(ROOT, "data", "gsm8k_dev200.tsv")
DEFAULT_OUT = os.path.join(RESULTS, "modality_probe2_summary.txt")

EXEC_TIMEOUT = 8.0
BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 1234
TIE_BOOTSTRAP_N = 2000
TIE_BOOTSTRAP_SEED = 1234

NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
ESC_RE = re.compile(r"\\(\\|n|t|r)")
HEADER = ["id", "stream", "mode", "finish", "generated", "ttft_ms", "done_ms", "text"]


# ---------------------------------------------------------------- loading ----


def unescape_tsv(s):
    return ESC_RE.sub(lambda m: {"\\": "\\", "n": "\n", "t": "\t", "r": "\r"}[m.group(1)], s)


def load_stream_tsv(path):
    rows = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        header = fh.readline().rstrip("\r\n").split("\t")
        if header != HEADER:
            raise SystemExit(f"{path}: unexpected header {header!r}; expected {HEADER!r}")
        for lineno, line in enumerate(fh, 2):
            line = line.rstrip("\r\n")
            if not line:
                continue
            parts = line.split("\t", 7)
            if len(parts) != 8:
                raise SystemExit(f"{path}:{lineno}: expected 8 tab fields, got {len(parts)}")
            try:
                row = {
                    "id": int(parts[0]),
                    "stream": int(parts[1]),
                    "mode": parts[2],
                    "finish": parts[3],
                    "generated": int(parts[4]),
                    "ttft_ms": float(parts[5]),
                    "done_ms": float(parts[6]),
                    "text": unescape_tsv(parts[7]),
                }
            except ValueError as exc:
                raise SystemExit(f"{path}:{lineno}: bad numeric field ({exc})")
            if row["mode"] not in ("cot", "code"):
                raise SystemExit(f"{path}:{lineno}: bad mode {row['mode']!r}")
            if row["finish"] not in ("eos", "answer", "fence", "cap"):
                raise SystemExit(f"{path}:{lineno}: bad finish {row['finish']!r}")
            rows.append(row)
    if not rows:
        raise SystemExit(f"{path}: no rows")
    by_id = {}
    for r in rows:
        seen = by_id.setdefault(r["id"], {})
        if r["stream"] in seen:
            raise SystemExit(f"{path}: duplicate id={r['id']} stream={r['stream']}")
        seen[r["stream"]] = r
    return rows, by_id


def load_sidecar(path):
    sc = path + ".run.json"
    if not os.path.exists(sc):
        return None, None
    with open(sc, encoding="utf-8") as fh:
        return json.load(fh), sc


def load_gold(path):
    """id == 1-indexed line number, gold == last tab field of that line."""
    gold = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.rstrip("\r\n")
            if not line:
                continue
            gold[lineno] = line.split("\t")[-1]
    return gold


def normalize_number(s):
    if s is None:
        return None
    s = s.strip().replace("$", "").replace(",", "")
    s = s.lstrip("+")
    s = re.sub(r"[^0-9.\-]", "", s)
    s = re.sub(r"[^0-9]+$", "", s)
    return s or None


def canon(s):
    if s is None:
        return None
    try:
        f = float(s)
    except (TypeError, ValueError):
        return s
    if f == int(f):
        return str(int(f))
    return repr(f)


# ------------------------------------------------------------- extraction ----


# Balanced-brace \boxed{...} scanning and content classification. A box is
# accepted only when its braces balance; nested braces (frac/text) are handled.

def _think_block_end(text):
    low = text.lower()
    p = low.rfind("</think")
    if p == -1:
        return 0
    close = text.find(">", p)
    return (close + 1) if close != -1 else len(text)


def _balanced_from(text, open_idx):
    """Content and index of the '}' matching text[open_idx] == '{'."""
    depth = 0
    for j in range(open_idx, len(text)):
        c = text[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[open_idx + 1:j], j
    return None, None


def find_balanced_boxed(text):
    """Content of the last balanced \\boxed{...} after any think block, else None."""
    start = _think_block_end(text)
    key = "\\boxed{"
    last, pos = None, start
    while True:
        i = text.find(key, pos)
        if i == -1:
            return last
        content, end = _balanced_from(text, i + len(key) - 1)
        if content is None:
            pos = i + len(key)
        else:
            last, pos = content, end + 1


_PURE_NUM_RE = re.compile(r"^-?[0-9][0-9,]*(?:\.[0-9]+)?$")
_FRAC_RE = re.compile(r"^\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}$")
_RATIO_RE = re.compile(r"^(-?[0-9]+(?:\.[0-9]+)?)\s*/\s*(-?[0-9]+(?:\.[0-9]+)?)$")
_NUM_LITERAL_RE = re.compile(r"^-?[0-9]+(?:\.[0-9]+)?$")
_ARITH_CHARS_RE = re.compile(r"^[0-9+\-*/().\s]+$")


def _strip_money(s):
    return s.strip().replace("$", "").strip()


def normalize_pure_numeric(s):
    """A single numeric literal; thousands separators allowed. Returns a
    canonical string, or None for multiple comma-separated numbers / non-numeric."""
    if s is None:
        return None
    s = _strip_money(s)
    if not _PURE_NUM_RE.match(s):
        return None
    negative = s.startswith("-")
    body = s[1:] if negative else s
    intpart, dot, frac = body.partition(".")
    if "," in intpart:
        groups = intpart.split(",")
        if len(groups) < 2 or not (1 <= len(groups[0]) <= 3):
            return None
        if any(len(g) != 3 for g in groups[1:]):
            return None
        intpart = "".join(groups)
    if not intpart:
        return None
    return ("-" if negative else "") + intpart + (("." + frac) if dot else "")


def _num_literal(s):
    if s is None:
        return None
    s = _strip_money(s)
    if not _NUM_LITERAL_RE.match(s):
        return None
    return float(s) if "." in s else int(s)


def safe_arith(expr):
    """Evaluate a numeric-literal-only expression (+ - * / () ). No names,
    calls or attributes; returns a number or None."""
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except (SyntaxError, ValueError):
        return None

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError("bad constant")
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            v = ev(node.operand)
            return v if isinstance(node.op, ast.UAdd) else -v
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            a, b = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Add):
                return a + b
            if isinstance(node.op, ast.Sub):
                return a - b
            if isinstance(node.op, ast.Mult):
                return a * b
            if b == 0:
                raise ValueError("div0")
            return a / b
        raise ValueError("unsupported")

    try:
        return ev(tree)
    except (ValueError, ZeroDivisionError, RecursionError):
        return None


def classify_boxed(content):
    """Balanced-box content -> (canonical value string, path), or (None, None)."""
    s = content.strip()
    if not s:
        return None, None
    m = _FRAC_RE.match(s)
    if m:
        a, b = _num_literal(m.group(1)), _num_literal(m.group(2))
        if a is not None and b is not None and b != 0:
            return canon(a / b), "boxed_frac"
    m = _RATIO_RE.match(s)
    if m:
        a, b = _num_literal(m.group(1)), _num_literal(m.group(2))
        if a is not None and b is not None and b != 0:
            return canon(a / b), "boxed_frac"
    v = normalize_pure_numeric(s)
    if v is not None:
        return canon(v), "boxed_numeric"
    if _ARITH_CHARS_RE.match(s):
        val = safe_arith(s)
        if val is not None:
            return canon(val), "boxed_expr"
    return None, None


def extract_boxed(text):
    content = find_balanced_boxed(text)
    if content is None:
        return None, None
    return classify_boxed(content)


def extract_cot(text):
    """Return (answer, path). Paths: boxed_numeric | boxed_frac | boxed_expr |
    answer_line | last_number | none. A balanced box takes priority; a box with
    non-numeric content falls back to the final answer line, then the last
    number in the text."""
    v, path = extract_boxed(text)
    if v is not None:
        return v, path

    low = text.lower()
    apos = low.rfind("the answer is")
    if apos != -1:
        tail = text[apos + len("the answer is"):]
        matches = list(NUM_RE.finditer(tail))
        if matches:
            v = normalize_number(matches[-1].group(0))
            if v is not None:
                return v, "answer_line"

    matches = list(NUM_RE.finditer(text))
    if matches:
        v = normalize_number(matches[-1].group(0))
        if v is not None:
            return v, "last_number"
    return None, "none"


def extract_code_block(text):
    """Last fenced block tagged python/py, else last untagged block.

    Blocks tagged `output` (or any other non-python tag) are ignored. An
    unterminated final fence is tolerated: its body runs to end of text.
    """
    blocks = []
    pos = 0
    while True:
        o = text.find("```", pos)
        if o == -1:
            break
        k = o + 3
        while k < len(text) and text[k] in " \t":
            k += 1
        tag_start = k
        while k < len(text) and (text[k].isalnum() or text[k] in "_+-"):
            k += 1
        tag = text[tag_start:k].lower()
        if k < len(text) and text[k] == "\r":
            k += 1
        if k < len(text) and text[k] == "\n":
            k += 1
        body_start = k
        c = text.find("```", body_start)
        if c == -1:
            blocks.append((tag, text[body_start:]))
            break
        blocks.append((tag, text[body_start:c]))
        pos = c + 3

    py = [b for b in blocks if b[0] in ("python", "py")]
    if py:
        return py[-1][1].strip() or None
    plain = [b for b in blocks if b[0] == ""]
    if plain:
        return plain[-1][1].strip() or None
    return None


def _last_print_call(code):
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return None
    best = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "print":
            if best is None or (node.lineno, node.col_offset) > (best.lineno, best.col_offset):
                best = node
    return best


def code_is_compliant(code, stdout):
    """True when stdout has exactly one numeric token, or the last print(...)
    call has a single positional argument that statically evaluates to a
    number (numeric literals/arithmetic only). Voting itself is unchanged."""
    if len(NUM_RE.findall(stdout or "")) == 1:
        return True
    node = _last_print_call(code or "")
    if node is None or node.keywords or len(node.args) != 1:
        return False
    if isinstance(node.args[0], ast.Starred):
        return False
    seg = ast.get_source_segment(code, node.args[0])
    return seg is not None and safe_arith(seg) is not None


def run_code(code, timeout=EXEC_TIMEOUT):
    """Execute in an isolated subprocess; return (answer, status, compliant).

    status in ok | empty | error | timeout. `ok` requires exit 0 and >=1
    number on stdout. Voting still uses the last numeric token on stdout;
    `compliant` records a single-number output (see code_is_compliant).
    `-I -S` disables site/user env; cwd is a fresh temp dir.
    """
    with tempfile.TemporaryDirectory() as d:
        prog = os.path.join(d, "prog.py")
        with open(prog, "w", encoding="utf-8") as fh:
            fh.write(code)
        env = {
            "PATH": os.environ.get("PATH", ""),
            "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
            "TEMP": d,
            "TMP": d,
        }
        try:
            r = subprocess.run(
                [sys.executable, "-I", "-S", prog],
                capture_output=True, text=True, timeout=timeout,
                cwd=d, env=env, encoding="utf-8", errors="replace",
            )
        except subprocess.TimeoutExpired:
            return None, "timeout", False
        stdout = r.stdout or ""
        compliant = code_is_compliant(code, stdout)
        if r.returncode != 0:
            return None, "error", compliant
        nums = NUM_RE.findall(stdout)
        if not nums:
            return None, "empty", compliant
        return normalize_number(nums[-1]), "ok", compliant


# ------------------------------------------------------------------ voting ----


def plurality(items):
    """items: list of (answer, substrate, original_stream_index).

    Tie-break: highest count, then most distinct substrates, then lowest sum
    of stream indices, then canonical key. Ties are reported on the primary
    count criterion.
    """
    groups = {}
    for ans, sub, idx in items:
        if ans is None:
            continue
        key = canon(ans)
        g = groups.setdefault(key, {"count": 0, "subs": set(), "idx": 0, "rep": ans})
        g["count"] += 1
        g["subs"].add(sub)
        g["idx"] += idx
    if not groups:
        return {"answer": None, "tied": False, "top": [], "n_votes": 0}
    max_count = max(g["count"] for g in groups.values())
    top_keys = sorted(k for k, g in groups.items() if g["count"] == max_count)
    ranked = sorted(
        groups.items(),
        key=lambda kv: (-kv[1]["count"], -len(kv[1]["subs"]), kv[1]["idx"], kv[0]),
    )
    return {
        "answer": ranked[0][1]["rep"],
        "tied": len(top_keys) > 1,
        "top": [groups[k]["rep"] for k in top_keys],
        "n_votes": sum(g["count"] for g in groups.values()),
    }


def gate_answer(cot_items, code_items):
    """Answers supported by >=1 cot and >=1 code; highest total support."""
    groups = {}
    for ans, sub, _ in cot_items + code_items:
        if ans is None:
            continue
        key = canon(ans)
        g = groups.setdefault(key, {"cot": 0, "code": 0, "rep": ans})
        g[sub] += 1
    eligible = {k: g for k, g in groups.items() if g["cot"] >= 1 and g["code"] >= 1}
    if not eligible:
        return None
    max_total = max(g["cot"] + g["code"] for g in eligible.values())
    top_keys = sorted(k for k, g in eligible.items() if g["cot"] + g["code"] == max_total)
    ranked = sorted(
        eligible.items(),
        key=lambda kv: (-(kv[1]["cot"] + kv[1]["code"]), -min(kv[1]["cot"], kv[1]["code"]), kv[0]),
    )
    return {
        "answer": ranked[0][1]["rep"],
        "tied": len(top_keys) > 1,
        "top": [eligible[k]["rep"] for k in top_keys],
        "n_votes": sum(g["cot"] + g["code"] for g in eligible.values()),
    }


# --------------------------------------------------------------- statistics ----


def pct(x, n):
    return 100.0 * x / n if n else 0.0


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1))
    return min(1.0, 2.0 * tail * (0.5 ** n))


def bootstrap_ci(a, b, seed=BOOTSTRAP_SEED, reps=BOOTSTRAP_N):
    rng = random.Random(seed)
    n = len(a)
    if n == 0:
        return 0.0, 0.0
    diffs = []
    for _ in range(reps):
        s = 0
        for _ in range(n):
            k = rng.randrange(n)
            s += a[k] - b[k]
        diffs.append(s / n)
    diffs.sort()
    lo = diffs[int(0.025 * reps)]
    hi = diffs[min(reps - 1, int(0.975 * reps))]
    return lo, hi


def percentile(values, q):
    if not values:
        return float("nan")
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = q * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def _pick_answer(rng, result):
    """A single answer for one problem, breaking ties uniformly at random."""
    if result["tied"] and result["top"]:
        return rng.choice(result["top"])
    return result["answer"]


def tie_aware_paired_bootstrap(arm_a, arm_b, ids, gold, seed=TIE_BOOTSTRAP_SEED, reps=TIE_BOOTSTRAP_N):
    """Primary paired CI. Each resample picks n problems with replacement (same
    index for both arms), then breaks every tied plurality uniformly at random
    before scoring. Returns (lo, hi) percentiles in points."""
    rng = random.Random(seed)
    n = len(ids)
    if n == 0:
        return 0.0, 0.0
    g = {i: canon(gold[i]) for i in ids}
    diffs = []
    for _ in range(reps):
        idx = [ids[rng.randrange(n)] for _ in range(n)]
        ca = cb = 0
        for i in idx:
            va = _pick_answer(rng, arm_a[i])
            vb = _pick_answer(rng, arm_b[i])
            ca += 1 if va is not None and canon(va) == g[i] else 0
            cb += 1 if vb is not None and canon(vb) == g[i] else 0
        diffs.append(100.0 * (ca - cb) / n)
    diffs.sort()
    lo = diffs[int(0.025 * reps)]
    hi = diffs[min(reps - 1, int(0.975 * reps))]
    return lo, hi


def load_provenance(label, mixed_path, explicit):
    """Read key: value / key=value provenance written by the run wrapper."""
    cands = []
    if explicit:
        cands.append(explicit)
    base_dir = os.path.dirname(os.path.abspath(mixed_path)) if mixed_path else RESULTS
    cands += [
        os.path.join(base_dir, f"{label}_provenance.txt"),
        os.path.join(RESULTS, f"{label}_provenance.txt"),
        os.path.join(RESULTS, "m3_provenance.txt"),
    ]
    for c in cands:
        if c and os.path.exists(c):
            d = {}
            with open(c, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if ":" in line:
                        k, v = line.split(":", 1)
                    elif "=" in line:
                        k, v = line.split("=", 1)
                    else:
                        continue
                    d[k.strip()] = v.strip()
            return d, c
    return None, None


def tie_metrics(arm_results, ids, gold):
    """Tie-aware accuracy: point (deterministic tie-break), random-tie
    expectation, and the upper bound achievable by picking any tied answer."""
    correct = tied = 0
    rand_exp = 0.0
    upper = 0
    for i in ids:
        r = arm_results[i]
        g = canon(gold[i])
        a = r["answer"]
        ok = a is not None and canon(a) == g
        correct += 1 if ok else 0
        if r["tied"] and r["top"]:
            tied += 1
            k = len(r["top"])
            rand_exp += sum(1 for x in r["top"] if canon(x) == g) / k
            upper += 1 if any(canon(x) == g for x in r["top"]) else 0
        else:
            rand_exp += 1 if ok else 0
            upper += 1 if ok else 0
    n = len(ids)
    return {
        "correct": correct,
        "point": pct(correct, n),
        "tied": tied,
        "rand": pct(rand_exp, n),
        "upper": pct(upper, n),
    }


# ------------------------------------------------------------------ analysis ----


def analyze_group(label, mixed_path, cot16_path, code16_path, gold_raw, exec_timeout,
                  provenance_path=None):
    mixed_rows, mixed_by_id = load_stream_tsv(mixed_path)
    mixed_sidecar, mixed_sidecar_path = load_sidecar(mixed_path)
    prov, prov_path = load_provenance(label, mixed_path, provenance_path)

    cot16_by_id, code16_by_id = None, None
    cot16_sidecar = code16_sidecar = None
    if cot16_path:
        _, cot16_by_id = load_stream_tsv(cot16_path)
        cot16_sidecar, _ = load_sidecar(cot16_path)
    if code16_path:
        _, code16_by_id = load_stream_tsv(code16_path)
        code16_sidecar, _ = load_sidecar(code16_path)

    ids = sorted(mixed_by_id)
    for i in ids:
        if i not in gold_raw:
            raise SystemExit(f"{label}: no gold for id={i} in gold file")
    if cot16_by_id is not None and set(cot16_by_id) != set(ids):
        raise SystemExit(f"{label}: cot16 ids differ from mixed ({sorted(set(cot16_by_id) ^ set(ids))})")
    if code16_by_id is not None and set(code16_by_id) != set(ids):
        raise SystemExit(f"{label}: code16 ids differ from mixed ({sorted(set(code16_by_id) ^ set(ids))})")

    gold = {i: normalize_number(gold_raw[i]) for i in ids}

    def split(by_id):
        cot, code = {}, {}
        for i in ids:
            streams = by_id[i]
            cot[i] = [streams[s] for s in sorted(streams) if streams[s]["mode"] == "cot"]
            code[i] = [streams[s] for s in sorted(streams) if streams[s]["mode"] == "code"]
        return cot, code

    mixed_cot_rows, mixed_code_rows = split(mixed_by_id)

    # ---- extract CoT answers (value + path) ---------------------------------
    cot_ans, cot_paths = {}, {}
    for i in ids:
        pairs = [extract_cot(r["text"]) for r in mixed_cot_rows[i]]
        cot_ans[i] = [v for v, _ in pairs]
        cot_paths[i] = [p for _, p in pairs]
    cot16_ans = cot16_paths = None
    if cot16_by_id is not None:
        cot16_ans, cot16_paths = {}, {}
        for i in ids:
            pairs = [extract_cot(cot16_by_id[i][s]["text"]) for s in sorted(cot16_by_id[i])]
            cot16_ans[i] = [v for v, _ in pairs]
            cot16_paths[i] = [p for _, p in pairs]

    # ---- extract + execute code ---------------------------------------------
    code_src = {i: [extract_code_block(r["text"]) for r in mixed_code_rows[i]] for i in ids}
    code16_src = None
    if code16_by_id is not None:
        code16_src = {i: [extract_code_block(code16_by_id[i][s]["text"]) for s in sorted(code16_by_id[i])] for i in ids}

    all_sources = set()
    for i in ids:
        all_sources.update(c for c in code_src[i] if c)
        if code16_src is not None:
            all_sources.update(c for c in code16_src[i] if c)
    all_sources = sorted(all_sources)
    exec_cache = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for src, res in zip(all_sources, pool.map(lambda s: run_code(s, exec_timeout), all_sources)):
            exec_cache[src] = res

    def resolve(sources):
        ans, st, comp = [], [], []
        for c in sources:
            if not c:
                ans.append(None)
                st.append("nocode")
                comp.append(False)
            else:
                a, s, k = exec_cache[c]
                ans.append(a)
                st.append(s)
                comp.append(k)
        return ans, st, comp

    code_ans, code_status, code_ok = {}, {}, {}
    for i in ids:
        code_ans[i], code_status[i], code_ok[i] = resolve(code_src[i])
    code16_ans = code16_status = code16_ok = None
    if code16_src is not None:
        code16_ans, code16_status, code16_ok = {}, {}, {}
        for i in ids:
            code16_ans[i], code16_status[i], code16_ok[i] = resolve(code16_src[i])

    # ---- arm votes -----------------------------------------------------------
    def cot_items(i):
        return [(a, "cot", idx) for idx, a in enumerate(cot_ans[i])]

    def code_items(i):
        return [(a, "code", idx) for idx, a in enumerate(code_ans[i])]

    arms = {}
    for i in ids:
        c8 = plurality(cot_items(i))
        p8 = plurality(code_items(i))
        mpool = plurality(cot_items(i) + code_items(i))
        mgate = gate_answer(cot_items(i), code_items(i))
        gate_fallback = False
        if mgate is None:
            mgate = dict(mpool)
            gate_fallback = True
        arms.setdefault("C8", {})[i] = c8
        arms.setdefault("P8", {})[i] = p8
        arms.setdefault("M16_pool", {})[i] = mpool
        arms.setdefault("M16_gate", {})[i] = mgate
        arms["M16_gate_fallback"] = arms.get("M16_gate_fallback", {})
        arms["M16_gate_fallback"][i] = gate_fallback

        if cot16_ans is not None:
            arms.setdefault("C16", {})[i] = plurality(
                [(a, "cot", idx) for idx, a in enumerate(cot16_ans[i])])
        if code16_ans is not None:
            arms.setdefault("P16", {})[i] = plurality(
                [(a, "code", idx) for idx, a in enumerate(code16_ans[i])])

    # ---- oracle coverage -----------------------------------------------------
    oracle = {}
    for i in ids:
        g = canon(gold[i])
        oc = any(canon(a) == g for a in cot_ans[i] if a is not None)
        op = any(canon(a) == g for a in code_ans[i] if a is not None)
        oracle.setdefault("cot_mixed", {})[i] = oc
        oracle.setdefault("code_mixed", {})[i] = op
        oracle.setdefault("union_mixed", {})[i] = oc or op
        if cot16_ans is not None:
            oracle.setdefault("cot16", {})[i] = any(canon(a) == g for a in cot16_ans[i] if a is not None)
        if code16_ans is not None:
            oracle.setdefault("code16", {})[i] = any(canon(a) == g for a in code16_ans[i] if a is not None)
        if cot16_ans is not None and code16_ans is not None:
            oracle.setdefault("union16", {})[i] = oracle["cot16"][i] or oracle["code16"][i]

    # ---- tie-aware arm metrics ----------------------------------------------
    arm_names = [a for a in ("C8", "P8", "C16", "P16", "M16_pool", "M16_gate") if a in arms]
    arm_metrics = {a: tie_metrics(arms[a], ids, gold) for a in arm_names}

    # ---- per-stream accuracy -------------------------------------------------
    def stream_accuracy(ans, rows):
        total = answered = correct = 0
        per_index = {}
        for i in ids:
            g = canon(gold[i])
            for idx, a in enumerate(ans[i]):
                total += 1
                ok = a is not None and canon(a) == g
                if a is not None:
                    answered += 1
                correct += 1 if ok else 0
                d = per_index.setdefault(idx, [0, 0])
                d[0] += 1
                d[1] += 1 if ok else 0
        return {
            "total": total, "answered": answered, "correct": correct,
            "per_index": {k: pct(v[1], v[0]) for k, v in per_index.items()},
        }

    stream_stats = {
        "cot_mixed": stream_accuracy(cot_ans, mixed_cot_rows),
        "code_mixed": stream_accuracy(code_ans, mixed_code_rows),
    }
    if cot16_ans is not None:
        stream_stats["cot16"] = stream_accuracy(cot16_ans, None)
    if code16_ans is not None:
        stream_stats["code16"] = stream_accuracy(code16_ans, None)

    # ---- code exec coverage -------------------------------------------------
    def status_counts(status, ids_):
        c = {"ok": 0, "empty": 0, "error": 0, "timeout": 0, "nocode": 0}
        for i in ids_:
            for s in status[i]:
                c[s] = c.get(s, 0) + 1
        return c

    code_cov = {"mixed": status_counts(code_status, ids)}
    if code16_status is not None:
        code_cov["code16"] = status_counts(code16_status, ids)

    def compliant_counts(okmap, ids_):
        yes = tot = 0
        for i in ids_:
            for k in okmap[i]:
                tot += 1
                yes += 1 if k else 0
        return {"compliant": yes, "total": tot}

    code_compliant = {"mixed": compliant_counts(code_ok, ids)}
    if code16_ok is not None:
        code_compliant["code16"] = compliant_counts(code16_ok, ids)

    # ---- wrong-answer diagnostics -------------------------------------------
    # Abstentions are labelled separately; "both wrong" never counts a missing
    # plurality as wrong. The Jaccard is conditioned on both substrates having
    # produced at least one answer.
    jaccards = []
    outcomes = {"both_answered_both_wrong": 0, "both_answered_one_right": 0,
                "both_answered_plurality_abstained": 0,
                "cot_only_answered": 0, "code_only_answered": 0, "both_abstained": 0}
    recover = {"M16_pool": 0, "M16_gate": 0}
    n_bothanswered = 0
    n_bothanswered_bothwrong = 0
    for i in ids:
        g = canon(gold[i])
        cot_has = any(a is not None for a in cot_ans[i])
        code_has = any(a is not None for a in code_ans[i])
        if cot_has and code_has:
            n_bothanswered += 1
            cw = {canon(a) for a in cot_ans[i] if a is not None and canon(a) != g}
            pw = {canon(a) for a in code_ans[i] if a is not None and canon(a) != g}
            if cw or pw:
                jaccards.append(len(cw & pw) / len(cw | pw))
            c8r, p8r = arms["C8"][i]["answer"], arms["P8"][i]["answer"]
            if c8r is None or p8r is None:
                outcomes["both_answered_plurality_abstained"] += 1
                continue
            c8_ok = canon(c8r) == g
            p8_ok = canon(p8r) == g
            if not c8_ok and not p8_ok:
                outcomes["both_answered_both_wrong"] += 1
                n_bothanswered_bothwrong += 1
                for a in ("M16_pool", "M16_gate"):
                    if arms[a][i]["answer"] is not None and canon(arms[a][i]["answer"]) == g:
                        recover[a] += 1
            else:
                outcomes["both_answered_one_right"] += 1
        elif cot_has:
            outcomes["cot_only_answered"] += 1
        elif code_has:
            outcomes["code_only_answered"] += 1
        else:
            outcomes["both_abstained"] += 1
    mean_jaccard = statistics.fmean(jaccards) if jaccards else float("nan")

    # ---- batch-composition reproducibility ---------------------------------
    # Mixed and homogeneous runs share prompt+seed for overlapping stream
    # indices, but different batch composition changes GPU numerics. Measure
    # how often the sampled text is actually identical.
    overlap = {}
    if cot16_by_id is not None:
        same = tot = 0
        for i in ids:
            for s, row in mixed_by_id[i].items():
                other = cot16_by_id[i].get(s)
                if row["mode"] == "cot" and other is not None and other["mode"] == "cot":
                    tot += 1
                    same += row["text"] == other["text"]
        overlap["cot"] = (same, tot)
    if code16_by_id is not None:
        same = tot = 0
        for i in ids:
            for s, row in mixed_by_id[i].items():
                other = code16_by_id[i].get(s)
                if row["mode"] == "code" and other is not None and other["mode"] == "code":
                    tot += 1
                    same += row["text"] == other["text"]
        overlap["code"] = (same, tot)

    # ---- paired tests --------------------------------------------------------
    def acc(a):
        return arm_metrics[a]["point"]

    comparisons = []
    best8 = "C8" if acc("C8") >= acc("P8") else "P8"
    homog_candidates = [a for a in ("C16", "P16") if a in arm_metrics]
    if not homog_candidates:
        homog_candidates = ["C8", "P8"]
    homog_ref = max(homog_candidates, key=lambda a: arm_metrics[a]["point"])
    for arm in ("M16_pool", "M16_gate"):
        comparisons.append((arm, best8))
        comparisons.append((arm, homog_ref))
    if "C16" in arm_metrics:
        for arm in ("M16_pool", "M16_gate"):
            comparisons.append((arm, "C16"))
    if "P16" in arm_metrics:
        for arm in ("M16_pool", "M16_gate"):
            comparisons.append((arm, "P16"))
    if "C16" in arm_metrics and "P16" in arm_metrics:
        best16 = "C16" if acc("C16") >= acc("P16") else "P16"
        for arm in ("M16_pool", "M16_gate"):
            comparisons.append((arm, best16))
    comparisons.append(("M16_gate", "M16_pool"))

    seen = set()
    tests = []
    primary_set = {("M16_pool", "C16"), ("M16_gate", "C16"), ("M16_pool", best8)}
    for arm, ref in comparisons:
        if (arm, ref) in seen or arm == ref:
            continue
        seen.add((arm, ref))
        av = [1 if arms[arm][i]["answer"] is not None and canon(arms[arm][i]["answer"]) == canon(gold[i]) else 0
              for i in ids]
        bv = [1 if arms[ref][i]["answer"] is not None and canon(arms[ref][i]["answer"]) == canon(gold[i]) else 0
              for i in ids]
        b = sum(1 for x, y in zip(av, bv) if x and not y)
        c = sum(1 for x, y in zip(av, bv) if not x and y)
        p = mcnemar_exact(b, c)
        lo, hi = bootstrap_ci(av, bv)
        lo_tie, hi_tie = tie_aware_paired_bootstrap(arms[arm], arms[ref], ids, gold)
        tests.append({
            "arm": arm, "ref": ref, "b": b, "c": c, "p": p,
            "primary": (arm, ref) in primary_set,
            "delta": pct(sum(av), len(ids)) - pct(sum(bv), len(ids)),
            "lo": 100 * lo, "hi": 100 * hi,
            "lo_tie": lo_tie, "hi_tie": hi_tie,
        })

    # ---- latency / token diagnostics ---------------------------------------
    def latency(side, rows, tag):
        d = {"tag": tag}
        if side:
            d["prefill_mean"] = statistics.fmean(side.get("prefill_ms", [])) if side.get("prefill_ms") else float("nan")
            d["decode_mean"] = statistics.fmean(side.get("decode_ms", [])) if side.get("decode_ms") else float("nan")
            d["decode_median"] = statistics.median(side.get("decode_ms", [])) if side.get("decode_ms") else float("nan")
            d["total_mean"] = statistics.fmean(side.get("total_ms", [])) if side.get("total_ms") else float("nan")
            d["step_p50"] = percentile(side.get("step_ms", []), 0.50)
            d["step_p95"] = percentile(side.get("step_ms", []), 0.95)
            d["step_recorded"] = side.get("step_ms_recorded", len(side.get("step_ms", [])))
            d["step_total"] = side.get("step_ms_total", len(side.get("step_ms", [])))
            d["step_fraction"] = side.get("step_ms_fraction", 1.0)
            probs = side.get("problems", [])
            d["prompt_tokens_max"] = max((p.get("prompt_tokens_max", 0) for p in probs), default=0)
            d["prompt_tokens_mean"] = statistics.fmean(
                p.get("prompt_tokens_sum", 0) / max(1, side.get("n_streams", 1)) for p in probs
            ) if probs else float("nan")
        if rows:
            by_mode = {}
            for r in rows:
                m = by_mode.setdefault(r["mode"], {"n": 0, "cap": 0, "gen": []})
                m["n"] += 1
                m["cap"] += 1 if r["finish"] == "cap" else 0
                m["gen"].append(r["generated"])
            d["by_mode"] = {
                m: {"n": v["n"], "cap": v["cap"], "cap_pct": pct(v["cap"], v["n"]),
                    "gen_mean": statistics.fmean(v["gen"]), "gen_max": max(v["gen"])}
                for m, v in by_mode.items()
            }
        return d

    mixed_rows_all = mixed_rows
    lat = {"mixed": latency(mixed_sidecar, mixed_rows_all, "mixed")}
    if cot16_by_id is not None:
        lat["cot16"] = latency(cot16_sidecar, [r for i in ids for r in cot16_by_id[i].values()], "cot16")
    if code16_by_id is not None:
        lat["code16"] = latency(code16_sidecar, [r for i in ids for r in code16_by_id[i].values()], "code16")

    return {
        "label": label,
        "paths": {"mixed": mixed_path, "cot16": cot16_path, "code16": code16_path},
        "sidecar_path": mixed_sidecar_path,
        "sidecar": mixed_sidecar,
        "provenance": prov, "provenance_path": prov_path,
        "ids": ids, "gold": gold,
        "rows_by_id": mixed_by_id, "code_ans": code_ans, "code_status": code_status,
        "cot_paths": cot_paths, "code_compliant": code_compliant,
        "arms": arms, "arm_names": arm_names, "arm_metrics": arm_metrics,
        "oracle": oracle, "stream_stats": stream_stats,
        "code_cov": code_cov,
        "homog_ref": homog_ref, "overlap": overlap,
        "n_bothanswered": n_bothanswered, "mean_jaccard": mean_jaccard,
        "n_jaccard": len(jaccards), "outcomes": outcomes, "recover": recover,
        "n_bothanswered_bothwrong": n_bothanswered_bothwrong,
        "tests": tests, "latency": lat,
    }


# ------------------------------------------------------------------- output ----


def write_csv(r, path):
    arm_order = ["C8", "P8", "C16", "P16", "M16_pool", "M16_gate"]
    oracle_order = ["cot_mixed", "code_mixed", "union_mixed", "cot16", "code16", "union16"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        header = ["id", "gold"]
        for a in arm_order:
            header += [a, a + "_ok"]
        header += oracle_order
        header += ["cot_answers", "code_answers", "code_status",
                   "cot_finish", "code_finish", "cot_generated", "code_generated"]
        w.writerow(header)
        for i in r["ids"]:
            g = canon(r["gold"][i])
            row = [i, r["gold"][i]]
            for a in arm_order:
                if a in r["arms"]:
                    ans = r["arms"][a][i]["answer"]
                    row += [ans if ans is not None else "",
                            1 if ans is not None and canon(ans) == g else 0]
                else:
                    row += ["", ""]
            for o in oracle_order:
                row.append(int(r["oracle"][o][i]) if o in r["oracle"] else "")
            by_id = r["rows_by_id"]
            cot_rows = [by_id[i][s] for s in sorted(by_id[i]) if by_id[i][s]["mode"] == "cot"]
            code_rows = [by_id[i][s] for s in sorted(by_id[i]) if by_id[i][s]["mode"] == "code"]
            cot_vals = [extract_cot(x["text"])[0] for x in cot_rows]
            row.append("|".join(v if v is not None else "" for v in cot_vals))
            row.append("|".join(v if v is not None else "" for v in r["code_ans"][i]))
            row.append("|".join(r["code_status"][i]))
            row.append("|".join(x["finish"] for x in cot_rows))
            row.append("|".join(x["finish"] for x in code_rows))
            row.append("|".join(str(x["generated"]) for x in cot_rows))
            row.append("|".join(str(x["generated"]) for x in code_rows))
            w.writerow(row)


def format_group(r):
    L = []
    n = len(r["ids"])
    L.append(f"=== group {r['label']} (n={n}) ===")
    L.append(f"  mixed : {r['paths']['mixed']}")
    L.append(f"  cot16 : {r['paths']['cot16'] or '(none)'}")
    L.append(f"  code16: {r['paths']['code16'] or '(none)'}")
    if r.get("sidecar"):
        L.append(f"  sidecar: {r.get('sidecar_path')}")
        L.append(f"  n_ctx={r['sidecar'].get('n_ctx')} max_trace={r['sidecar'].get('max_trace')} "
                 f"seed={r['sidecar'].get('seed')} n_streams={r['sidecar'].get('n_streams')}")
        frac = r["sidecar"].get("step_ms_fraction")
        if frac is not None:
            L.append(f"  step_ms recorded {r['sidecar'].get('step_ms_recorded')} / "
                     f"{r['sidecar'].get('step_ms_total')} steps (fraction {frac:.4f})")

    L.append("")
    L.append("provenance:")
    pv = r.get("provenance") or {}
    if pv:
        for k in ("model_file", "model_sha256", "source_file", "source_sha256",
                  "exe_file", "exe_sha256", "data_file", "data_sha256",
                  "max_trace", "seed", "limit", "generated"):
            if k in pv:
                L.append(f"  {k:>14}: {pv[k]}")
        L.append(f"  {'file':>14}: {r.get('provenance_path')}")
    else:
        sc = r.get("sidecar") or {}
        L.append(f"  {'model_file':>14}: {sc.get('model', 'n/a')}")
        L.append(f"  {'max_trace':>14}: {sc.get('max_trace', 'n/a')}")
        L.append(f"  {'seed':>14}: {sc.get('seed', 'n/a')}")
        L.append(f"  {'provenance':>14}: (no provenance.txt; binary/source hashes unavailable)")
    L.append(f"  {'n':>14}: {n}")

    L.append("")
    L.append("per-stream accuracy by substrate (stream-level, mixed run unless noted):")
    for tag in ("cot_mixed", "code_mixed", "cot16", "code16"):
        if tag not in r["stream_stats"]:
            continue
        s = r["stream_stats"][tag]
        L.append(f"  {tag:>10}: answered {s['answered']}/{s['total']} ({pct(s['answered'], s['total']):.1f}%), "
                 f"correct {s['correct']}/{s['total']} ({pct(s['correct'], s['total']):.1f}%)")
        L.append(f"              per-index acc%: " +
                 ", ".join(f"{k}:{v:.0f}" for k, v in sorted(s["per_index"].items())))

    L.append("")
    L.append("code exec coverage:")
    for tag, c in r["code_cov"].items():
        tot = sum(c.values())
        L.append(f"  {tag} (n={tot}): ok {c['ok']} ({pct(c['ok'], tot):.1f}%), "
                 f"empty {c['empty']}, error {c['error']}, timeout {c['timeout']}, nocode {c['nocode']}")
    for tag, c in r.get("code_compliant", {}).items():
        L.append(f"  code_compliant {tag}: {c['compliant']}/{c['total']} "
                 f"({pct(c['compliant'], c['total']):.1f}%) of code streams print a single number")

    L.append("")
    L.append("CoT extraction path (stream-level, mixed):")
    tally = {}
    for i in r["ids"]:
        for p in r["cot_paths"][i]:
            tally[p] = tally.get(p, 0) + 1
    for k in sorted(tally):
        L.append(f"  {k:>14}: {tally[k]}")

    L.append("")
    L.append("oracle coverage (any stream correct):")
    for o in ("cot_mixed", "code_mixed", "union_mixed", "cot16", "code16", "union16"):
        if o in r["oracle"]:
            k = sum(r["oracle"][o].values())
            L.append(f"  {o:>11}: {k}/{n} ({pct(k, n):.1f}%)")

    L.append("")
    L.append("arms (tie-aware; point uses the deterministic tie-break):")
    L.append(f"  {'arm':>10} {'correct':>8} {'point%':>7} {'tied':>5} {'rand%':>7} {'upper%':>7}")
    for a in r["arm_names"]:
        m = r["arm_metrics"][a]
        L.append(f"  {a:>10} {m['correct']:>8} {m['point']:>7.1f} {m['tied']:>5} "
                 f"{m['rand']:>7.1f} {m['upper']:>7.1f}")
    if "M16_gate" in r["arms"]:
        fb = sum(1 for i in r["ids"] if r["arms"]["M16_gate_fallback"][i])
        L.append(f"  M16_gate fell back to M16_pool when no cross-substrate support: {fb}/{n}")
    h = r["homog_ref"]
    hm = r["arm_metrics"][h]
    L.append(f"  max homogeneous reference: {h} (point {hm['point']:.1f}%, "
             f"tie upper {hm['upper']:.1f}%, tied {hm['tied']})")

    L.append("")
    L.append("batch-composition reproducibility (same prompt+seed and stream index):")
    if r["overlap"]:
        for tag, (same, tot) in r["overlap"].items():
            L.append(f"  {tag}: identical sampled text {same}/{tot} "
                     f"({pct(same, tot):.1f}%); same-batch reruns are deterministic")
    else:
        L.append("  n/a (homogeneous runs not supplied)")

    L.append("")
    L.append("substrate outcome tally (abstention labelled separately, not counted wrong):")
    o = r["outcomes"]
    L.append(f"  both answered, both pluralities wrong : {o['both_answered_both_wrong']}/{n} "
             f"(abstentions excluded; {r['n_bothanswered_bothwrong']} counted)")
    L.append(f"  both answered, >=1 plurality right    : {o['both_answered_one_right']}/{n}")
    L.append(f"  both answered, plurality abstained    : {o['both_answered_plurality_abstained']}/{n}")
    L.append(f"  only CoT answered (code abstained)    : {o['cot_only_answered']}/{n}")
    L.append(f"  only code answered (CoT abstained)    : {o['code_only_answered']}/{n}")
    L.append(f"  both substrates abstained             : {o['both_abstained']}/{n}")
    L.append(f"  both-answered problems                : {r['n_bothanswered']}/{n}")
    if r["n_jaccard"]:
        L.append(f"  wrong-answer Jaccard (both answered)  : {r['mean_jaccard']:.3f} "
                 f"over {r['n_jaccard']} problems with a non-empty wrong union")
    else:
        L.append("  wrong-answer Jaccard (both answered)  : n/a (no non-empty wrong union)")
    L.append(f"  recovery among both-wrong problems    : M16_pool {r['recover']['M16_pool']}, "
             f"M16_gate {r['recover']['M16_gate']} (of {o['both_answered_both_wrong']})")

    L.append("")
    L.append("paired tests: PRIMARY = tie-aware paired bootstrap (2000 resamples, seed 1234, 95% CI);")
    L.append("  secondary = McNemar exact on the point outcome + point-outcome 10k bootstrap.")
    for t in r["tests"]:
        tag = " [PRIMARY]" if t.get("primary") else ""
        L.append(f"  {t['arm']} vs {t['ref']}{tag}: delta={t['delta']:+.1f} pts "
                 f"tieCI [{t['lo_tie']:+.1f}, {t['hi_tie']:+.1f}] "
                 f"pointCI [{t['lo']:+.1f}, {t['hi']:+.1f}] b={t['b']} c={t['c']} p={t['p']:.4f}")
    for arm in ("M16_pool", "M16_gate"):
        t = next((x for x in r["tests"] if x["arm"] == arm and x["ref"] == r["homog_ref"]), None)
        if t:
            L.append(f"  mixed-vs-homogeneous [{arm} vs {t['ref']}]: "
                     f"{'positive at 95%' if t['delta'] > 0 and t['lo_tie'] > 0 else 'not clearly positive'} "
                     f"(delta {t['delta']:+.1f} pts, tieCI [{t['lo_tie']:+.1f}, {t['hi_tie']:+.1f}], p={t['p']:.4f})")

    L.append("")
    L.append("latency / tokens (sidecar + TSV):")
    for tag, d in r["latency"].items():
        L.append(f"  {tag}: prefill_mean={d.get('prefill_mean', float('nan')):.1f} ms, "
                 f"decode_mean={d.get('decode_mean', float('nan')):.1f} ms, "
                 f"decode_median={d.get('decode_median', float('nan')):.1f} ms, "
                 f"total_mean={d.get('total_mean', float('nan')):.1f} ms, "
                 f"step_p50={d.get('step_p50', float('nan')):.2f} ms, "
                 f"step_p95={d.get('step_p95', float('nan')):.2f} ms "
                 f"(over recorded prefix {d.get('step_recorded')}/{d.get('step_total')}, "
                 f"frac {d.get('step_fraction', 1.0):.3f})")
        if "prompt_tokens_max" in d:
            L.append(f"         prompt tokens: max {d['prompt_tokens_max']}, mean/stream "
                     f"{d['prompt_tokens_mean']:.1f}")
        for m, v in d.get("by_mode", {}).items():
            L.append(f"         {m}: n={v['n']} cap={v['cap']} ({v['cap_pct']:.1f}%), "
                     f"gen mean {v['gen_mean']:.1f}, max {v['gen_max']}")
    L.append("")
    return "\n".join(L)


# --------------------------------------------------------------------- main ----


def parse_run_arg(s):
    parts = s.split(",")
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise SystemExit(f"--run expects NAME,mixed[,cot16[,code16]], got {s!r}")
    parts += [""] * (4 - len(parts))
    return parts[0], parts[1], parts[2], parts[3]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="append", default=[],
                    help="NAME,mixed_tsv[,cot16_tsv[,code16_tsv]] (repeatable)")
    ap.add_argument("--label", default="run")
    ap.add_argument("--mixed", default="")
    ap.add_argument("--cot16", default="")
    ap.add_argument("--code16", default="")
    ap.add_argument("--gold-file", default=DEFAULT_GOLD)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--provenance", default="", help="explicit provenance.txt path")
    ap.add_argument("--exec-timeout", type=float, default=EXEC_TIMEOUT)
    args = ap.parse_args()

    groups = [parse_run_arg(s) for s in args.run]
    if not groups:
        if not args.mixed:
            ap.error("give --run or --mixed")
        groups = [(args.label, args.mixed, args.cot16, args.code16)]

    gold_raw = load_gold(args.gold_file)
    out_lines = [
        "RecycleLM cross-substrate modality probe 2 (clean harness)",
        f"gold: {os.path.abspath(args.gold_file)}",
        "source: harness/engine/modality.cpp (escaped text, per-stream prompts,",
        "one batched decode over all live streams; code blocks preserved)",
        "",
    ]

    for label, mixed, cot16, code16 in groups:
        r = analyze_group(label, mixed, cot16, code16, gold_raw, args.exec_timeout,
                          args.provenance or None)
        csv_path = os.path.join(RESULTS, f"modality_probe2_{label}.csv")
        write_csv(r, csv_path)
        out_lines.append(format_group(r))
        out_lines.append(f"[written] {csv_path}")
        out_lines.append("")

    text = "\n".join(out_lines).rstrip() + "\n"
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(text)
    sys.stdout.write(text)
    sys.stdout.write(f"\n[written] {args.out}\n")


if __name__ == "__main__":
    main()
