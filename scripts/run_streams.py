"""DesignStreams driver: mixed CoT + code streams, sandboxed execution, voting.

Inputs: two TSVs from streams.exe (--mode cot and --mode code), each
"id <TAB> stream <TAB> text". Outputs accuracy for the arms and a code-share
sweep, with paired bootstrap intervals.

Generated code is untrusted: each program runs in an isolated subprocess
(`python -I -S`), in a temp dir, with a wall-clock timeout, stdout captured,
and no inherited environment. (Windows lacks cheap network/memory jail, so this
is best-effort isolation; the code is tiny and model-generated.)
"""
import csv
import os
import random
import re
import statistics
import subprocess
import sys
import tempfile

COT = sys.argv[1] if len(sys.argv) > 1 else "results/streams_1b_cot.tsv"
CODE = sys.argv[2] if len(sys.argv) > 2 else "results/streams_1b_code.tsv"
GOLD = sys.argv[3] if len(sys.argv) > 3 else "data/gsm8k_heldout.tsv"
TAG = sys.argv[4] if len(sys.argv) > 4 else "1b"
N = 8
TIMEOUT = 6.0


def last_number(s):
    nums = re.findall(r"-?\d[\d,]*(?:\.\d+)?", s.replace("$", ""))
    if not nums:
        return None
    return nums[-1].replace(",", "")


def extract_code(text):
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.S)
    if m:
        return m.group(1)
    if re.search(r"^\s*(print|import|for|def|n\s*=)", text):
        return text
    return None


def run_code(code):
    """Returns (answer_or_None, status). status in ok|error|timeout|empty."""
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "prog.py")
        with open(p, "w", encoding="utf-8") as f:
            f.write(code)
        try:
            r = subprocess.run(
                [sys.executable, "-I", "-S", p],
                capture_output=True, text=True, timeout=TIMEOUT, cwd=d,
                env={"PATH": os.environ.get("PATH", ""), "SYSTEMROOT": os.environ.get("SYSTEMROOT", "")},
            )
        except subprocess.TimeoutExpired:
            return None, "timeout"
        if r.returncode != 0:
            return None, "error"
        a = last_number(r.stdout)
        return (a, "ok") if a is not None else (None, "empty")


def load(path):
    d = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) < 3:
                continue
            d.setdefault(int(row[0]), {})[int(row[1])] = row[2]
    return d


cot = load(COT)
code = load(CODE)
gold = {}
with open(GOLD, encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 2:
            gold[i] = parts[1].replace(",", "")

ids = sorted(i for i in cot if i in gold and i in code)
print(f"problems={len(ids)}  model={TAG}")

# precompute answers
cot_ans, code_ans = {}, {}
crashes = 0
for i in ids:
    cot_ans[i] = [last_number(cot[i].get(j, "")) for j in range(N)]
    ca, statuses = [], []
    for j in range(N):
        c = extract_code(code[i].get(j, ""))
        if not c:
            ca.append(None); statuses.append("nocode"); continue
        a, st = run_code(c)
        ca.append(a); statuses.append(st)
    code_ans[i] = ca
    crashes += sum(1 for s in statuses if s in ("error", "timeout"))
print(f"code streams dropped (error/timeout): {crashes}/{len(ids)*N} = {100*crashes/(len(ids)*N):.1f}%")


def vote(ans):
    vals = [a for a in ans if a is not None]
    if not vals:
        return None
    return statistics.mode(vals)


def acc(getter):
    return 100 * sum(1 for i in ids if getter(i) is not None and getter(i) == gold[i]) / len(ids)


def paired(getter, ref):
    d = []
    for _ in range(10000):
        idx = [ids[random.randrange(len(ids))] for _ in ids]
        d.append(sum((getter(i) == gold[i]) - (ref(i) == gold[i]) for i in idx) / len(ids))
    d.sort()
    return 100 * d[250], 100 * d[9750]


single = lambda i: cot_ans[i][0]
text_only = lambda i: vote(cot_ans[i])
code_only = lambda i: vote(code_ans[i])
mixed = lambda i: vote(cot_ans[i][:4] + code_ans[i][:4])

print(f"{'arm':>12} {'acc%':>7}")
for name, fn in [("single", single), ("text-only", text_only), ("code-only", code_only), ("mixed 4+4", mixed)]:
    print(f"{name:>12} {acc(fn):>7.1f}")

print("\ncode-share sweep (cs code streams + 8-cs text streams)")
print(f"{'cs':>3} {'acc%':>7} {'vs text-only':>22}")
tb = acc(text_only)
for cs in (0, 2, 4, 8):
    fn = lambda i, cs=cs: vote(cot_ans[i][:8 - cs] + code_ans[i][:cs])
    lo, hi = paired(fn, text_only)
    print(f"{cs:>3} {acc(fn):>7.1f}  {acc(fn)-tb:>+6.1f} pts CI [{lo:+.1f},{hi:+.1f}]")
