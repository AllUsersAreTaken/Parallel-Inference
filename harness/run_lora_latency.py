"""Latency matrix for per-stream LoRA on the shared-prefix batch.

Arms (N=8 streams, 5.9k shared prefix, fixed 128 max tokens):
  A  base                      no adapter
  B  full r16 throughout        adapter applied to prefill + decode
  C  8 adapters additive        upper bound on grouped per-stream decode cost
  D  8 sequential N=1 runs      cost if every adapter needs its own prefill
  E  base prefill + r16 decode  one prefill, adapted decode (exact for KV-safe adapters)
  E8 base prefill + 8 additive  upper bound for KV-safe + grouped decode
  F  lastblock r16 throughout   KV-safe class, applied to prefill + decode
  G  full r64 throughout        rank scaling of the same overhead
"""
import json
import os
import subprocess
import threading
import time

ROOT = r"S:\Agent\QSExplore\RecycleLM"
CLI = os.path.join(ROOT, "app", "prefix_cli.exe")
MODEL = os.path.join(ROOT, "models", "Qwen2.5-3B-Instruct-Q4_K_M.gguf")
PREFIX = os.path.join(ROOT, "data", "_rc_prefix.txt")
SUFFIX8 = os.path.join(ROOT, "data", "_rc_suffixes.tsv")
SUFFIX1 = os.path.join(ROOT, "data", "_rc_suffixes1.tsv")
SYN = os.path.join(ROOT, "data", "_synth")
OUT = os.path.join(ROOT, "results", "_lora_latency")
REPS = 3
MAXTOK = 128

AD = {
    "full16": os.path.join(SYN, "lora_full_r16_zero.gguf"),
    "full64": os.path.join(SYN, "lora_full_r64_zero.gguf"),
    "lastblk": os.path.join(SYN, "lora_lastblock_r16_zero.gguf"),
}

os.makedirs(OUT, exist_ok=True)
env = dict(os.environ)
env["PATH"] = os.path.join(ROOT, "build", "cuda-graphs", "bin") + os.pathsep + env["PATH"]


def peak_vram(stop, box):
    while not stop.is_set():
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                                "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=5)
            v = int(r.stdout.strip().splitlines()[0])
            box[0] = max(box[0], v)
        except Exception:
            pass
        time.sleep(0.25)


def run(tag, data, loras, after_prefill=False, rep=0):
    out = os.path.join(OUT, "%s_%d.json" % (tag, rep))
    cmd = [CLI, "--model", MODEL, "--prefix", PREFIX, "--data", data,
           "--out", out, "--mode", "batch", "--max-tokens", str(MAXTOK),
           "--temperature", "0.6", "--top-p", "0.95", "--seed", "1234"]
    for p in loras:
        cmd += ["--lora", p]
    if after_prefill:
        cmd += ["--lora-after-prefill"]
    box = [0]
    stop = threading.Event()
    th = threading.Thread(target=peak_vram, args=(stop, box), daemon=True)
    th.start()
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
    wall = time.time() - t0
    stop.set()
    th.join(timeout=3)
    if r.returncode != 0:
        raise SystemExit("%s failed rc=%d\n%s" % (tag, r.returncode, r.stderr[-600:]))
    d = json.load(open(out, encoding="utf-8"))
    return {
        "tag": tag, "prefill_ms": d["prefill_ms"], "wall_ms": d["wall_ms"],
        "batch_wall": wall * 1000.0, "steps": d.get("steps"),
        "tokens": sum(s["generated"] for s in d["streams"]),
        "vram_peak": box[0], "n": len(d["streams"]),
    }


def med(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2]


def main():
    arms = []
    for rep in range(REPS):
        arms.append(run("A_base", SUFFIX8, [], rep=rep))
        arms.append(run("B_full16", SUFFIX8, [AD["full16"]], rep=rep))
        arms.append(run("C_8add", SUFFIX8, [AD["full16"]] * 8, rep=rep))
        arms.append(run("E_afterpref", SUFFIX8, [AD["full16"]], after_prefill=True, rep=rep))
        arms.append(run("E8_afterpref", SUFFIX8, [AD["full16"]] * 8, after_prefill=True, rep=rep))
        arms.append(run("F_lastblk", SUFFIX8, [AD["lastblk"]], rep=rep))
        arms.append(run("G_full64", SUFFIX8, [AD["full64"]], rep=rep))
        # D: eight sequential single-stream runs, each with one adapter
        pf, wl, tk = [], [], []
        for k in range(8):
            r1 = run("D_seq%d" % k, SUFFIX1, [AD["full16"]], rep=rep)
            pf.append(r1["prefill_ms"]); wl.append(r1["wall_ms"]); tk.append(r1["tokens"])
        arms.append({"tag": "D_seq8", "prefill_ms": sum(pf), "wall_ms": sum(wl),
                     "batch_wall": None, "steps": None, "tokens": sum(tk),
                     "vram_peak": None, "n": 1})

    by = {}
    for a in arms:
        by.setdefault(a["tag"], []).append(a)
    rows = []
    for tag, rs in by.items():
        rows.append({
            "tag": tag,
            "prefill_ms": med([x["prefill_ms"] for x in rs]),
            "wall_ms": med([x["wall_ms"] for x in rs]),
            "tokens": med([x["tokens"] for x in rs]),
            "vram": max(x["vram_peak"] or 0 for x in rs),
        })
    base = [r for r in rows if r["tag"] == "A_base"][0]
    print("%-14s %10s %10s %8s %8s %8s" % ("arm", "prefill", "wall", "pref_x", "wall_x", "tokens"))
    for r in sorted(rows, key=lambda z: z["wall_ms"]):
        print("%-14s %10.0f %10.0f %8.2f %8.2f %8.0f"
              % (r["tag"], r["prefill_ms"], r["wall_ms"],
                 r["prefill_ms"] / base["prefill_ms"], r["wall_ms"] / base["wall_ms"],
                 r["tokens"]))
    print("\nVRAM peak (MiB):", {r["tag"]: r["vram"] for r in rows})
    json.dump({"reps": REPS, "max_tokens": MAXTOK, "rows": rows, "raw": arms},
              open(os.path.join(ROOT, "results", "lora_latency.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
