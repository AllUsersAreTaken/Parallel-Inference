"""Capability check: base Qwen2.5-3B-Instruct vs base+LoRA on our held-out GSM8K split.

Same 100 length-quantile problems as the n=100 disagreement probe, N=1 stream,
greedy decode, chunked 8 problems per invocation (capacity-safe).
"""
import io
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "harness"))
import analyze_modality_vote as M
import run_disagree_probe as D

CLI = os.path.join(ROOT, "app", "prefix_cli.exe")
BIN = os.path.join(ROOT, "build", "cuda-graphs", "bin")
MODEL = os.path.join(ROOT, "models", "Qwen2.5-3B-Instruct-Q4_K_M.gguf")
SCRATCH = os.path.join(ROOT, "data", "_gsm8k_lora.tsv")
OUT = os.path.join(ROOT, "data", "_gsm8k_lora.json")
MAXTOK = 512
CHUNK = 8


def run_arm(tag, loras):
    problems, _ = D.load_problems(100)
    env = dict(os.environ)
    env["PATH"] = BIN + os.pathsep + env.get("PATH", "")
    preds, walls = [], []
    for start in range(0, len(problems), CHUNK):
        batch = problems[start:start + CHUNK]
        with open(SCRATCH, "w", encoding="utf-8") as fh:
            for i, (q, g) in enumerate(batch):
                fh.write("%d\t%s\n" % (i + 1, D.esc(D.INSTR + q)))
        cmd = [CLI, "--model", MODEL, "--data", SCRATCH, "--out", OUT,
               "--mode", "batch", "--max-tokens", str(MAXTOK),
               "--temperature", "0.0", "--top-p", "1.0", "--seed", "1234"]
        for p in loras:
            cmd += ["--lora", p]
        r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
        if r.returncode != 0:
            raise SystemExit("cli failed %s: %s" % (tag, r.stderr[-500:]))
        d = json.load(open(OUT, encoding="utf-8"))
        walls.append(d["wall_ms"])
        for st in d["streams"]:
            a, _ = M.extract_cot(st["text"])
            preds.append(M.canon(a))
    gold = [M.canon(M.normalize_number(g)) for _, g in problems]
    ok = [p is not None and p == g for p, g in zip(preds, gold)]
    return {"tag": tag, "preds": preds, "gold": gold, "ok": ok,
            "acc": sum(ok) / len(ok), "wall_ms": walls}


def main():
    lora = os.path.join(ROOT, "data", "_hf_lora", "gsm8k_reasoning.gguf")
    base = run_arm("base", [])
    lr = run_arm("gsm8k_lora", [lora])
    print("held-out GSM8K, n=%d, greedy, 1 stream" % len(base["ok"]))
    print("  base        acc %.1f%% (%d/%d)  median wall %.0f ms"
          % (100 * base["acc"], sum(base["ok"]), len(base["ok"]), sorted(base["wall_ms"])[len(base["wall_ms"]) // 2]))
    print("  +LoRA       acc %.1f%% (%d/%d)  median wall %.0f ms"
          % (100 * lr["acc"], sum(lr["ok"]), len(lr["ok"]), sorted(lr["wall_ms"])[len(lr["wall_ms"]) // 2]))
    b2l = sum(1 for a, b in zip(base["ok"], lr["ok"]) if not a and b)
    l2b = sum(1 for a, b in zip(base["ok"], lr["ok"]) if a and not b)
    print("  flips: base-wrong->lora-right %d, base-right->lora-wrong %d" % (b2l, l2b))
    print("  delta: %+0.1f pp" % (100 * (lr["acc"] - base["acc"])))
    json.dump({"base": base, "lora": lr}, open(os.path.join(ROOT, "results", "gsm8k_lora_eval.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
