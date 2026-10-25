"""Run the bug-repair eval: base Qwen2.5-3B-Instruct vs base+debug LoRA.

Prompt format follows the adapter card (v1: buggy code only)."""
import json
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLI = os.path.join(ROOT, "app", "prefix_cli.exe")
BIN = os.path.join(ROOT, "build", "cuda-graphs", "bin")
MODEL = os.path.join(ROOT, "models", "Qwen2.5-3B-Instruct-Q4_K_M.gguf")
EVAL = os.path.join(ROOT, "data", "_repair_eval.jsonl")
SCRATCH = os.path.join(ROOT, "data", "_repair_batch.tsv")
OUT = os.path.join(ROOT, "data", "_repair_batch.json")

SYSTEM = ("You are an expert Python debugging assistant. The user gives you a buggy "
          "Python program (and optionally its runtime error and an execution trace). "
          "Reply with the corrected program in a single ```python code block. Do not "
          "add explanations.")
CHUNK = 4
MAXTOK = 512


def esc(s):
    return s.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")


def extract_code(text):
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.S)
    code = m.group(1) if m else text
    return code.strip()


def passes(sol, row):
    fd, path = tempfile.mkstemp(suffix=".py")
    os.close(fd)
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(sol + "\n" + row["test"] + "\n\ncheck(%s)\n" % row["entry_point"])
        r = subprocess.run([sys.executable, path], capture_output=True, timeout=6)
        return r.returncode == 0
    except Exception:
        return False
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def run_arm(tag, loras):
    rows = [json.loads(l) for l in open(EVAL, encoding="utf-8") if l.strip()]
    env = dict(os.environ); env["PATH"] = BIN + os.pathsep + env.get("PATH", "")
    ok, walls, details = [], [], []
    for start in range(0, len(rows), CHUNK):
        batch = rows[start:start + CHUNK]
        with open(SCRATCH, "w", encoding="utf-8") as fh:
            for i, r in enumerate(batch):
                user = "Fix the following buggy Python program.\n\n```python\n%s\n```" % r["buggy"]
                fh.write("%d\t%s\n" % (i + 1, esc(user)))
        cmd = [CLI, "--model", MODEL, "--data", SCRATCH, "--out", OUT, "--mode", "batch",
               "--max-tokens", str(MAXTOK), "--temperature", "0.0", "--top-p", "1.0",
               "--seed", "1234", "--system", SYSTEM]
        for p in loras:
            cmd += ["--lora", p]
        r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
        if r.returncode != 0:
            raise SystemExit("cli failed: %s" % r.stderr[-500:])
        d = json.load(open(OUT, encoding="utf-8"))
        walls.append(d["wall_ms"])
        for st, row in zip(d["streams"], batch):
            code = extract_code(st["text"])
            sol = code if ("def %s(" % row["entry_point"]) in code else row["prompt"] + code
            good = passes(sol, row)
            ok.append(good)
            details.append({"task_id": row["task_id"], "ok": good})
    return {"tag": tag, "ok": ok, "acc": sum(ok) / len(ok), "walls": walls, "details": details}


def main():
    lora = os.path.join(ROOT, "data", "_hf_lora", "debug_v1.gguf")
    base = run_arm("base", [])
    lr = run_arm("debug_lora", [lora])
    print("bug repair (HumanEval mutants), n=%d, greedy" % len(base["ok"]))
    print("  base      pass@1 %.1f%% (%d/%d)  median wall %.0f ms"
          % (100 * base["acc"], sum(base["ok"]), len(base["ok"]),
             sorted(base["walls"])[len(base["walls"]) // 2]))
    print("  +debug    pass@1 %.1f%% (%d/%d)  median wall %.0f ms"
          % (100 * lr["acc"], sum(lr["ok"]), len(lr["ok"]),
             sorted(lr["walls"])[len(lr["walls"]) // 2]))
    b2l = sum(1 for a, b in zip(base["ok"], lr["ok"]) if not a and b)
    l2b = sum(1 for a, b in zip(base["ok"], lr["ok"]) if a and not b)
    print("  flips: wrong->right %d, right->wrong %d | delta %+0.1f pp"
          % (b2l, l2b, 100 * (lr["acc"] - base["acc"])))
    json.dump({"base": base, "lora": lr},
              open(os.path.join(ROOT, "results", "repair_lora_eval.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
