"""MBR vs greedy vs single-sample vs majority on 20 programmatically-checked probes.

Arms: A greedy, B single stochastic, C chrF-MBR@4, D majority@4, E oracle@4,
F constraint-filtered MBR@4. 3 trials per stochastic arm (4 seeds per trial).
"""
import io
import json
import os
import re
import subprocess
import time

ROOT = r"S:\Agent\QSExplore\RecycleLM"
CLI = os.path.join(ROOT, "app", "prefix_cli.exe")
BIN = os.path.join(ROOT, "build", "cuda-graphs", "bin")
MODEL = os.path.join(ROOT, "models", "Qwen2.5-3B-Instruct-Q4_K_M.gguf")
PROBES = os.path.join(ROOT, "data", "_mbr_probes.json")
SCRATCH = os.path.join(ROOT, "data", "_mbr_batch.tsv")
OUT = os.path.join(ROOT, "data", "_mbr_batch.json")
MAXTOK = 200
TRIALS = 3
SEED0 = 1234

from sacrebleu.metrics import CHRF
CHRF_M = CHRF()

import sys
sys.path.insert(0, os.path.join(ROOT, "harness"))
import analyze_modality_vote as M


def esc(s):
    return s.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")


def run_samples(prompt, seed, temp, n):
    """One shared chat prefix, n branches (seq_cp), n samples of the same prompt."""
    pp = os.path.join(ROOT, "data", "_mbr_prompt.txt")
    with io.open(pp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(prompt)
    env = dict(os.environ); env["PATH"] = BIN + os.pathsep + env.get("PATH", "")
    cmd = [CLI, "--model", MODEL, "--prompt", pp, "--samples", str(n), "--mode", "samples",
           "--out", OUT, "--max-tokens", str(MAXTOK), "--temperature", str(temp),
           "--top-p", "0.9", "--seed", str(seed)]
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise SystemExit("cli failed: %s" % r.stderr[-400:])
    d = json.load(open(OUT, encoding="utf-8"))
    return [s["text"] for s in d["streams"]], d["wall_ms"], d["prefill_ms"]


def numbers(text):
    return re.findall(r"-?\d+(?:\.\d+)?", text)


def check(text, c):
    t = c["type"]
    low = text.lower()
    if t == "num":
        ans, _ = M.extract_cot(text)
        vals = []
        try:
            vals.append(float(ans))
        except Exception:
            vals = [float(x) for x in numbers(text)]
        for g in c["gold"]:
            for v in vals:
                if abs(v - g) <= max(1e-6, abs(g) * 1e-4):
                    return True
        return False
    if t == "sentences":
        n = len(re.findall(r"[.!?](?:\s|$)", text.strip()))
        ok = n == c["n"]
        if "all" in c:
            ok = ok and all(k in low for k in c["all"])
        return ok
    if t == "no_letter":
        return c["letter"].lower() not in low
    if t == "json_keys":
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return False
        try:
            o = json.loads(m.group(0))
        except Exception:
            return False
        if not isinstance(o, dict):
            return False
        if c.get("exact"):
            return set(o.keys()) == set(c["keys"])
        return all(k in o for k in c["keys"])
    if t == "json_schema":
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return False
        try:
            o = json.loads(m.group(0))
        except Exception:
            return False
        if not isinstance(o, dict) or "properties" not in o:
            return False
        props = o["properties"]
        for k, ty in c["keys"].items():
            if k not in props:
                return False
            jt = props[k].get("type")
            if ty == "int" and jt != "integer":
                return False
            if ty == "str" and jt != "string":
                return False
        return True
    if t == "lines_dash":
        lines = [l for l in text.splitlines() if l.strip()]
        return len([l for l in lines if l.strip().startswith("- ")]) == c["n"]
    if t == "upper":
        s = text.strip()
        return s == s.upper() and c["must"] in s
    if t == "count_word":
        return len(re.findall(r"\b%s\b" % re.escape(c["word"]), low)) == c["n"]
    if t == "keywords":
        return all(k in low for k in c["all"])
    if t == "cmd":
        return all(s.lower() in low for s in c["substrings"])
    if t == "code":
        return all(s in text for s in c["substrings"])
    return False


def canon(text, probe):
    if probe["family"] == "logic":
        a, _ = M.extract_cot(text)
        return a if a is not None else text.strip().lower()
    return re.sub(r"\s+", " ", text.strip().lower())


def mbr(cands):
    n = len(cands)
    if n == 1:
        return 0
    scores = []
    for i in range(n):
        s = 0.0
        for j in range(n):
            if i != j:
                s += CHRF_M.sentence_score(cands[i], [cands[j]]).score
        scores.append(s / (n - 1))
    return scores.index(max(scores))


def majority(cands, probe):
    keys = [canon(c, probe) for c in cands]
    counts = {}
    for k in keys:
        counts[k] = counts.get(k, 0) + 1
    best = max(counts.items(), key=lambda kv: kv[1])[0]
    return keys.index(best)


def main():
    probes = json.load(open(PROBES, encoding="utf-8"))
    arms = {k: [] for k in ("A", "B", "C", "D", "E", "F")}
    walls = {k: [] for k in ("A", "B", "C")}
    host_ms = []
    rows = []
    for p in probes:
        text_a, wa, _ = run_samples(p["prompt"], SEED0, 0.0, 1)
        arms["A"].append(check(text_a[0], p["check"]))
        walls["A"].append(wa)
        b_ok = c_ok = d_ok = e_ok = f_ok = 0
        for t in range(TRIALS):
            seed = SEED0 + 4 * t
            text_b, wb, _ = run_samples(p["prompt"], seed, 0.7, 1)
            cands, wc, _ = run_samples(p["prompt"], seed, 0.7, 4)
            walls["B"].append(wb); walls["C"].append(wc)
            t0 = time.perf_counter()
            i_mbr = mbr(cands)
            i_maj = majority(cands, p)
            filt = [i for i, c in enumerate(cands) if check(c, p["check"])]
            i_f = mbr([cands[i] for i in filt]) if filt else i_mbr
            if filt:
                i_f = filt[i_f]
            host_ms.append((time.perf_counter() - t0) * 1000)
            b_ok += check(text_b[0], p["check"])
            c_ok += check(cands[i_mbr], p["check"])
            d_ok += check(cands[i_maj], p["check"])
            e_ok += any(check(c, p["check"]) for c in cands)
            f_ok += check(cands[i_f], p["check"])
        arms["B"].append(b_ok / TRIALS); arms["C"].append(c_ok / TRIALS)
        arms["D"].append(d_ok / TRIALS); arms["E"].append(e_ok / TRIALS)
        arms["F"].append(f_ok / TRIALS)
        rows.append({"id": p["id"], "family": p["family"],
                     "A": arms["A"][-1], "B": arms["B"][-1], "C": arms["C"][-1],
                     "D": arms["D"][-1], "E": arms["E"][-1], "F": arms["F"][-1]})
        print("%-3s %-10s A=%d B=%.2f C=%.2f D=%.2f E=%.2f F=%.2f"
              % (p["id"], p["family"], arms["A"][-1], arms["B"][-1], arms["C"][-1],
                 arms["D"][-1], arms["E"][-1], arms["F"][-1]), flush=True)

    def acc(arm, fam=None):
        vals = [r[arm] for r in rows if fam is None or r["family"] == fam]
        return sum(vals) / len(vals) * 100

    print("\n%-4s %8s %8s %8s" % ("arm", "all", "logic", "constr", "tech")[:4])
    for a in ("A", "B", "C", "D", "E", "F"):
        print("%-4s %7.1f%% %7.1f%% %7.1f%% %7.1f%%"
              % (a, acc(a), acc(a, "logic"), acc(a, "constraint"), acc(a, "technical")))
    import statistics as st
    ratio = st.mean(walls["C"]) / st.mean(walls["A"])
    print("\nwall A %.0f ms, C %.0f ms, C/A %.3f" % (st.mean(walls["A"]), st.mean(walls["C"]), ratio))
    print("host MBR+majority mean %.2f ms, max %.2f ms" % (st.mean(host_ms), max(host_ms)))
    print("\nG1 C>B: %s (%.1f vs %.1f)" % (acc("C") > acc("B"), acc("C"), acc("B")))
    print("G2 C>=A: %s (%.1f vs %.1f)" % (acc("C") >= acc("A"), acc("C"), acc("A")))
    print("G3 wall C/A<=1.25: %s (%.3f)" % (ratio <= 1.25, ratio))
    print("G4 host<=5ms: %s (max %.2f)" % (max(host_ms) <= 5, max(host_ms)))
    print("G5 C>D: %s (%.1f vs %.1f)" % (acc("C") > acc("D"), acc("C"), acc("D")))
    print("G6 F>=C on constraints: %s (%.1f vs %.1f)" % (acc("F", "constraint") >= acc("C", "constraint"),
                                                          acc("F", "constraint"), acc("C", "constraint")))
    json.dump({"rows": rows, "arms": {k: acc(k) for k in arms},
               "walls": {k: sum(v) / len(v) for k, v in walls.items()},
               "host_ms": {"mean": st.mean(host_ms), "max": max(host_ms)},
               "gates": {"G1": acc("C") > acc("B"), "G2": acc("C") >= acc("A"),
                         "G3": ratio <= 1.25, "G4": max(host_ms) <= 5,
                         "G5": acc("C") > acc("D"),
                         "G6": acc("F", "constraint") >= acc("C", "constraint")}},
              open(os.path.join(ROOT, "results", "mbr_probe.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
