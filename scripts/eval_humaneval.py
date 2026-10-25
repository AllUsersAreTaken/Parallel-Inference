#!/usr/bin/env python3
"""Evaluate HumanEval harness output (results/humaneval_bench_raw.csv).

Official HumanEval protocol: a completion is executed as

    prompt + "\n" + completion + "\n" + test + "\n\ncheck(<entry_point>)\n"

for `stitch == 0` (the harness emitted a standalone artifact that does not
contain the prompt). When the harness already stitched the prompt/import
preamble back into the artifact (`stitch == 1/2`) the artifact is executed
as-is and the prompt is not prepended again.

The old evaluator always ran `code` alone, which false-fails HumanEval/38 and
/50: those prompts define helper functions (`encode_cyclic`, `encode_shift`)
that the completion calls. `--selftest` asserts the canonical solutions pass
under the official protocol and that the old standalone form raises NameError.

For every row the artifact is executed in a fresh namespace in a child process
with a 5 s wall-clock limit. pass1 = the program raised no exception. Syntax
errors are detected by compiling the assembled module in the parent (no
execution) so they can be counted separately from test failures. The pre-retry
actor artifact is scored with its own stitch mode (actor_pass1).

Writes results/humaneval_bench.csv: the harness columns plus
  pass1, syntax_ok, eval_status, eval_error, eval_error_line, eval_ms,
  actor_pass1, actor_syntax_ok
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time

RUNNER = r'''
import json, sys, traceback

payload_path, result_path = sys.argv[1], sys.argv[2]
with open(payload_path, encoding="utf-8") as fh:
    spec = json.load(fh)

code = spec["code"]
prefix = (spec["prompt"] + "\n") if spec.get("stitch", 0) == 0 else ""
program = prefix + code + "\n" + spec["test"] + "\n\ncheck(" + spec["entry_point"] + ")\n"
lines = program.splitlines()
res = {"status": "fail", "error": "", "error_line": ""}
ns = {"__name__": "__humaneval_solution__"}

def fail(ex):
    tb = ex.__traceback__
    res["error"] = type(ex).__name__
    msg = str(ex).strip().replace("\n", " ").replace("\r", " ")
    if msg:
        res["error"] = res["error"] + ": " + msg
    lineno = None
    while tb is not None:
        if tb.tb_frame.f_code.co_filename == "<humaneval>":
            lineno = tb.tb_lineno
        tb = tb.tb_next
    if lineno is not None:
        res["error_line"] = (lines[lineno - 1].strip() if 0 < lineno <= len(lines) else f"line {lineno}")
    res["error"] = res["error"][:400]
    res["error_line"] = res["error_line"][:300]

try:
    exec(compile(program, "<humaneval>", "exec"), ns)
except BaseException as ex:
    res["status"] = "fail"
    fail(ex)
else:
    res["status"] = "pass"

with open(result_path, "w", encoding="utf-8") as fh:
    json.dump(res, fh)
'''


def module_text(prompt, code, stitch):
    return (prompt + "\n" + code) if stitch == 0 else code


def eval_code(code, test, entry_point, workdir, timeout, prompt="", stitch=0):
    payload = os.path.join(workdir, "payload.json")
    result = os.path.join(workdir, "result.json")
    with open(payload, "w", encoding="utf-8") as fh:
        json.dump({"code": code, "test": test, "entry_point": entry_point,
                   "prompt": prompt, "stitch": stitch}, fh)
    if os.path.exists(result):
        os.remove(result)
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, "-X", "utf8", "-c", RUNNER, payload, result],
            capture_output=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "error": f"timeout after {timeout}s",
                "error_line": ""}, (time.perf_counter() - t0) * 1000.0
    dt = (time.perf_counter() - t0) * 1000.0
    if not os.path.exists(result):
        err = (proc.stderr or b"").decode("utf-8", "replace").strip().splitlines()
        return {"status": "crash", "error": (err[-1] if err else "no result")[:400],
                "error_line": ""}, dt
    with open(result, encoding="utf-8") as fh:
        res = json.load(fh)
    return res, dt


def score(code, task, workdir, timeout, stitch=0):
    """Returns (pass1, syntax_ok, status, error, error_line, ms)."""
    src = module_text(task["prompt"], code, stitch)
    try:
        compile(src, "<solution>", "exec")
    except SyntaxError as ex:
        return (0, 0, "syntax", f"SyntaxError: {ex.msg} (line {ex.lineno})"[:400],
                (ex.text or "").strip()[:300], 0.0)
    res, ms = eval_code(code, task["test"], task["entry_point"], workdir, timeout,
                        prompt=task["prompt"], stitch=stitch)
    return (1 if res["status"] == "pass" else 0, 1, res["status"],
            res.get("error", ""), res.get("error_line", ""), ms)


def run_selftest(tasks, full_src, workdir, timeout):
    """Checks the official protocol and reproduces the old false-fail.

    HumanEval/38 and /50 are the two tasks whose prompt defines a helper used by
    the completion. Standalone execution of a completion that only defines the
    entry point raises NameError; prompt + completion passes.
    """
    full = {}
    with open(full_src, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                full[r["task_id"]] = r

    ok = True
    for tid in ("HumanEval/38", "HumanEval/50"):
        task = tasks[tid]
        entry = task["entry_point"]
        body = full[tid]["canonical_solution"]
        d = task["prompt"].find("def " + entry)
        if d < 0:
            print(f"FAIL {tid}: entry point not found in prompt")
            ok = False
            continue
        standalone = task["prompt"][d:] + body

        stitched = task["prompt"] + "\n" + body
        can, can_ms = eval_code(stitched, task["test"], entry, workdir, timeout,
                                prompt=task["prompt"], stitch=1)
        new, new_ms = eval_code(standalone, task["test"], entry, workdir, timeout,
                                prompt=task["prompt"], stitch=0)
        old, old_ms = eval_code(standalone, task["test"], entry, workdir, timeout,
                                prompt="", stitch=1)

        helper = "encode_cyclic" if tid.endswith("38") else "encode_shift"
        case_ok = (can["status"] == "pass" and new["status"] == "pass"
                   and old["status"] == "fail" and helper in old.get("error", ""))
        ok = ok and case_ok
        print(f"{'PASS' if case_ok else 'FAIL'} {tid} {entry} ({helper})")
        print(f"  canonical prompt+completion: {can['status']} ({can_ms:.1f} ms)")
        print(f"  official protocol (stitch=0): {new['status']} ({new_ms:.1f} ms)")
        print(f"  old standalone (no prompt):   {old['status']} "
              f"({old_ms:.1f} ms) {old.get('error', '')[:120]}")
    print(f"selftest: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/humaneval_bench_raw.csv")
    ap.add_argument("--data", default="data/humaneval_100.jsonl")
    ap.add_argument("--out", default="results/humaneval_bench.csv")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--selftest", action="store_true",
                    help="run the HumanEval/38 and /50 protocol self-test and exit")
    ap.add_argument("--full-src", default="data/HumanEval.jsonl",
                    help="official HumanEval.jsonl (canonical solutions for --selftest)")
    args = ap.parse_args()

    tasks = {}
    with open(args.data, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                tasks[r["task_id"]] = r

    if args.selftest:
        workdir = tempfile.mkdtemp(prefix="humaneval_selftest_")
        sys.exit(run_selftest(tasks, args.full_src, workdir, args.timeout))

    with open(args.csv, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if args.limit:
        keep = {r["task_id"] for r in rows[: args.limit]}
        rows = [r for r in rows if r["task_id"] in keep]

    out_fields = list(rows[0].keys())
    for extra in ("pass1", "syntax_ok", "eval_status", "eval_error",
                  "eval_error_line", "eval_ms", "actor_pass1", "actor_syntax_ok"):
        if extra not in out_fields:
            out_fields.append(extra)

    workdir = tempfile.mkdtemp(prefix="humaneval_eval_")
    counts = {}
    t_all = time.perf_counter()
    for i, row in enumerate(rows):
        tid = row["task_id"]
        task = tasks.get(tid)
        if task is None:
            for k in ("pass1", "syntax_ok", "actor_pass1", "actor_syntax_ok"):
                row[k] = 0
            row["eval_status"] = "missing_task"
            row["eval_error"] = f"task_id not in {args.data}"
            row["eval_error_line"] = ""
            row["eval_ms"] = 0.0
            counts["missing_task"] = counts.get("missing_task", 0) + 1
            continue

        stitch = int(row.get("stitch") or 0)
        p1, syn, status, err, errline, ms = score(
            row["code"], task, workdir, args.timeout, stitch)
        row["pass1"] = p1
        row["syntax_ok"] = syn
        row["eval_status"] = status
        row["eval_error"] = err
        row["eval_error_line"] = errline
        row["eval_ms"] = f"{ms:.1f}"
        counts[status] = counts.get(status, 0) + 1

        acode = row.get("actor_code", "")
        if acode and acode != row["code"]:
            ast = int(row.get("actor_stitch") or stitch)
            ap1, asyn, astatus, aerr, aerrline, ams = score(
                acode, task, workdir, args.timeout, ast)
            row["actor_pass1"] = ap1
            row["actor_syntax_ok"] = asyn
            row["actor_eval_status"] = astatus
            if astatus != "pass" and astatus != status:
                counts["actor_" + astatus] = counts.get("actor_" + astatus, 0) + 1
        else:
            row["actor_pass1"] = p1
            row["actor_syntax_ok"] = syn

        if (i + 1) % 25 == 0:
            print(f"  evaluated {i + 1}/{len(rows)}")
            sys.stdout.flush()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=out_fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in out_fields})

    wall = time.perf_counter() - t_all
    print(f"rows: {len(rows)} -> {args.out}")
    for k in sorted(counts):
        print(f"  {k}: {counts[k]}")
    print(f"exec wall: {wall:.1f}s (timeout {args.timeout}s)")


if __name__ == "__main__":
    main()
