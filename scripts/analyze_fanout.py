"""Analyze RecycleLM fan-out raw CSVs (full3 / Amendment 2).

Inputs are the harness outputs:
  raw     <out>.csv          source_id,mode,lang,wall_ms,prefill_ms,ttft_ms,
                             done_ms,generated,finish,text
  runs    <out>.runs.csv     source_id,mode,wall_all_ms,prefill_ms
  steps   <out>.steps.csv    source_id,mode,steps,step_p50_ms,step_p90_ms

Arms: batched, seq (cold sequential), warm (shared prefix, per-language
scratch, single-stream decode). Speedup cold = seq_wall / batched_wall.
Speedup warm = warm_wall / batched_wall (isolates decode batching; both arms
pay the shared-prefix prefill once).

Outputs: pairs CSV, summary text, spot-check markdown, flagged CSV. Optional
round-trip back-translation uses the same fanout.exe with a temporary
one-column TSV.
"""

import argparse
import csv
import math
import os
import pathlib
import random
import re
import statistics
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]

LANG_NAMES = {
    "fra": "French", "spa": "Spanish", "deu": "German", "ita": "Italian",
    "por": "Portuguese", "rus": "Russian", "zho": "Chinese", "jpn": "Japanese",
    "eng": "English",
}
LATIN_STOP = {
    "fra": {"le", "la", "les", "des", "une", "un", "du", "et", "est", "que"},
    "spa": {"el", "la", "los", "las", "de", "del", "una", "un", "y", "es"},
    "deu": {"der", "die", "das", "und", "ist", "ein", "eine", "den", "dem", "zu"},
    "ita": {"il", "lo", "gli", "della", "la", "le", "un", "una", "e", "di"},
    "por": {"o", "a", "os", "as", "das", "de", "do", "da", "um", "uma", "e"},
}
ENGLISH_WORDS = {
    "the", "and", "of", "to", "is", "in", "that", "it", "you", "for", "was",
    "with", "as", "on", "are", "this", "be", "or", "by", "from", "at", "an",
    "not", "have", "but", "they", "his", "her", "she", "he", "we", "i", "will",
    "would", "can", "should", "has", "had", "were", "their", "there", "what",
}
CJK_LANGS = {"zho", "jpn"}

CYR = re.compile(r"[\u0400-\u04ff\u0500-\u052f]")
HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
KANA = re.compile(r"[\u3040-\u30ff\u31f0-\u31ff\uff66-\uff9d]")
WORD = re.compile(r"[^\W\d_]+", re.UNICODE)

REFUSAL = re.compile(
    r"(i (?:do not|don't) see|i (?:do not|don't) have|no text|no source text|"
    r"not provided|please provide|could you provide|"
    r"je ne vois pas|aucun texte|pas de texte|veuillez fournir|"
    r"no veo|no hay (?:ning[nú]n )?texto|no has proporcionado|"
    r"ich sehe (?:kein|keinen) text|kein text|"
    r"non vedo|nessun testo|"
    r"n[ãa]o vejo|nenhum texto|"
    r"не вижу|нет текста|"
    r"没有(?:任何)?文本|未提供|看不到|请提供|"
    r"テキスト(?:が|は)?(?:ありません|見つかりません|を提供))",
    re.IGNORECASE,
)


def is_refusal(text):
    return bool(REFUSAL.search(text))


def source_type(sid, compact_rows, gsm_rows):
    if sid in compact_rows:
        return "compact"
    if sid in gsm_rows:
        return "gsm8k"
    return "other"


def parse_row_range(s):
    if not s:
        return set()
    if "-" in s:
        a, b = s.split("-", 1)
        return set(range(int(a), int(b) + 1))
    return {int(s)}


def unescape_text(s):
    out = []
    i = 0
    while i < len(s):
        if s[i] == "\\" and i + 1 < len(s):
            c = s[i + 1]
            if c == "\\":
                out.append("\\")
                i += 2
                continue
            if c == "n":
                out.append("\n")
                i += 2
                continue
            if c == "t":
                out.append("\t")
                i += 2
                continue
            if c == "r":
                out.append("\r")
                i += 2
                continue
        out.append(s[i])
        i += 1
    return "".join(out)


def read_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _char_ngrams(text, n=3):
    text = re.sub(r"\s+", " ", text.strip())
    return Counter(text[i:i + n] for i in range(len(text) - n + 1))


def chrf(hyp, ref, n=3):
    h = _char_ngrams(hyp, n)
    r = _char_ngrams(ref, n)
    if not h and not r:
        return 1.0
    if not h or not r:
        return 0.0
    overlap = sum((h & r).values())
    p = overlap / sum(h.values())
    rec = overlap / sum(r.values())
    if p + rec == 0.0:
        return 0.0
    return 2.0 * p * rec / (p + rec)


def _letters(text):
    return [c for c in text if c.isalpha()]


def script_check(lang, text):
    """Raw frozen check + documented corrected Latin check (reported separately)."""
    letters = _letters(text)
    if not letters:
        return {"raw_ok": False, "corrected_ok": False, "reason": "no letters"}
    if lang == "rus":
        ratio = len(CYR.findall(text)) / len(letters)
        ok = ratio > 0.5
        return {"raw_ok": ok, "corrected_ok": ok, "reason": f"cyr={ratio:.3f}"}
    if lang == "zho":
        ratio = len(HAN.findall(text)) / len(letters)
        ok = ratio > 0.5
        return {"raw_ok": ok, "corrected_ok": ok, "reason": f"han={ratio:.3f}"}
    if lang == "jpn":
        ratio = (len(HAN.findall(text)) + len(KANA.findall(text))) / len(letters)
        ok = ratio > 0.5
        return {"raw_ok": ok, "corrected_ok": ok, "reason": f"han+kana={ratio:.3f}"}
    if lang in LATIN_STOP:
        words = [w.lower() for w in WORD.findall(text)]
        if not words:
            return {"raw_ok": False, "corrected_ok": False, "reason": "no words"}
        hits = sum(1 for w in words if w in LATIN_STOP[lang])
        en = sum(1 for w in words if w in ENGLISH_WORDS)
        en_rate = en / len(words)
        hit_rate = hits / len(words)
        raw_ok = hits >= 1 and en_rate < 0.15
        corrected_ok = (hits >= 1) or (en <= 2 and len(words) >= 6)
        return {
            "raw_ok": raw_ok,
            "corrected_ok": corrected_ok,
            "reason": (f"stop={hit_rate:.3f},en={en_rate:.3f},"
                       f"hits={hits},en_count={en},words={len(words)}"),
        }
    return {"raw_ok": True, "corrected_ok": True, "reason": "not-checked"}


def _units(text, lang):
    text = re.sub(r"\s+", " ", text.strip())
    if lang in CJK_LANGS:
        return [c for c in text if not c.isspace()]
    return WORD.findall(text.lower())


def concentrated_loop(text, lang, n=4, window=40, rep=3):
    """Supplementary: same n-gram repeated >=3 times inside a 40-unit window.

    This separates degenerate back-to-back loops from n-grams that recur
    naturally across a long, source-repetitive text.
    """
    units = _units(text, lang)
    if len(units) < 2 * n:
        return False
    for i in range(len(units) - n + 1):
        g = tuple(units[i:i + n])
        count = 1
        for j in range(i + 1, min(len(units) - n + 1, i + window)):
            if tuple(units[j:j + n]) == g:
                count += 1
                if count >= rep:
                    return True
    return False


def repetition_stats(text, lang, n=4):
    """Loop detector: max count of any repeated n-gram in generated text.

    Word n-grams for spaced languages; character n-grams for Chinese/Japanese.
    loop = max repeated n-gram count >= 3 (literal Amendment 2 definition).
    conc_loop is the supplementary concentrated-window flag.
    """
    units = _units(text, lang)
    if len(units) < n:
        return {"max_ngram": 0, "loop": False, "conc_loop": False,
                "units": len(units)}
    counts = Counter(tuple(units[i:i + n]) for i in range(len(units) - n + 1))
    mx = max(counts.values())
    return {
        "max_ngram": mx,
        "loop": mx >= 3,
        "conc_loop": concentrated_loop(text, lang, n),
        "units": len(units),
    }


def bootstrap_median(values, reps=10000, seed=1234):
    rng = random.Random(seed)
    n = len(values)
    samples = []
    for _ in range(reps):
        draw = [values[rng.randrange(n)] for _ in range(n)]
        samples.append(statistics.median(draw))
    samples.sort()
    lo = samples[int(0.025 * reps)]
    hi = samples[min(reps - 1, int(0.975 * reps))]
    return lo, hi


def fmt(x, nd=4):
    return f"{x:.{nd}f}"


def arm_quality(rows, compact_rows, gsm_rows):
    """Truncation / loop / script rates for one arm's rows."""
    n = len(rows)
    if not n:
        return None

    def trunc(sub):
        return sum(1 for r in sub if r["finish"] == "cap") / len(sub) if sub else float("nan")

    def loop(sub):
        return sum(1 for r in sub
                   if repetition_stats(unescape_text(r["text"]), r["lang"])["loop"]) / len(sub) if sub else float("nan")

    def conc(sub):
        return sum(1 for r in sub
                   if repetition_stats(unescape_text(r["text"]), r["lang"])["conc_loop"]) / len(sub) if sub else float("nan")

    def script(sub):
        vals = [script_check(r["lang"], unescape_text(r["text"])) for r in sub]
        return (sum(1 for v in vals if v["raw_ok"]) / len(vals) if vals else float("nan"),
                sum(1 for v in vals if v["corrected_ok"]) / len(vals) if vals else float("nan"))

    comp = [r for r in rows if int(r["source_id"]) in compact_rows]
    gsm = [r for r in rows if int(r["source_id"]) in gsm_rows]
    raw, corr = script(rows)
    return {
        "n": n,
        "trunc": trunc(rows),
        "trunc_compact": trunc(comp),
        "trunc_gsm8k": trunc(gsm),
        "loop": loop(rows),
        "loop_compact": loop(comp),
        "loop_gsm8k": loop(gsm),
        "conc": conc(rows),
        "conc_compact": conc(comp),
        "conc_gsm8k": conc(gsm),
        "script_raw": raw,
        "script_corrected": corr,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "results" / "fanout3_raw.csv"))
    ap.add_argument("--runs", default="")
    ap.add_argument("--steps", default="")
    ap.add_argument("--homog", default="")
    ap.add_argument("--n1", default="")
    ap.add_argument("--pairs-out", default=str(ROOT / "results" / "fanout3_pairs.csv"))
    ap.add_argument("--summary", default=str(ROOT / "results" / "fanout3_summary.txt"))
    ap.add_argument("--spotcheck", default=str(ROOT / "results" / "fanout3_spotcheck.md"))
    ap.add_argument("--flagged-out", default=str(ROOT / "results" / "fanout3_flagged.csv"))
    ap.add_argument("--compact-rows", default="1-10")
    ap.add_argument("--gsm8k-rows", default="11-20")
    ap.add_argument("--label", default="full3")
    ap.add_argument("--roundtrip", type=int, default=0)
    ap.add_argument("--roundtrip-select", default="balanced",
                    choices=["balanced", "compact", "gsm8k", "all"])
    ap.add_argument("--roundtrip-cap", type=int, default=512)
    ap.add_argument("--no-repeat-ngram", type=int, default=4)
    ap.add_argument("--repeat-penalty", default="1.1")
    ap.add_argument("--exe", default=str(ROOT / "harness" / "engine" / "fanout.exe"))
    ap.add_argument("--model", default=str(ROOT / "models" / "Qwen2.5-3B-Instruct-Q4_K_M.gguf"))
    ap.add_argument("--data", default=str(ROOT / "data" / "fanout_sources_20.tsv"))
    ap.add_argument("--bin", default=str(ROOT / "build" / "cuda-graphs" / "bin"))
    ap.add_argument("--compare-raw", default="",
                    help="optional earlier raw CSV for before/after truncation and loop rates")
    ap.add_argument("--compare-label", default="full2")
    args = ap.parse_args()

    raw_path = pathlib.Path(args.raw)
    runs_path = pathlib.Path(args.runs) if args.runs else pathlib.Path(str(raw_path) + ".runs.csv")
    steps_path = pathlib.Path(args.steps) if args.steps else pathlib.Path(str(raw_path) + ".steps.csv")
    compact_rows = parse_row_range(args.compact_rows)
    gsm_rows = parse_row_range(args.gsm8k_rows)
    flagged_path = pathlib.Path(args.flagged_out)

    rows = read_rows(raw_path)
    if not rows:
        sys.exit(f"no rows in {raw_path}")

    by_source = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_source[int(r["source_id"])][r["mode"]].append(r)

    order = []
    seen = set()
    for r in rows:
        if r["mode"] == "batched" and r["lang"] not in seen:
            seen.add(r["lang"])
            order.append(r["lang"])
    if not order:
        order = sorted({r["lang"] for r in rows})

    runs = {}
    if runs_path.exists():
        for r in read_rows(runs_path):
            runs[(int(r["source_id"]), r["mode"])] = (
                float(r["wall_all_ms"]), float(r["prefill_ms"]))

    step_stats = {}
    if steps_path.exists():
        for r in read_rows(steps_path):
            step_stats[(int(r["source_id"]), r["mode"])] = (
                int(r["steps"]), float(r["step_p50_ms"]), float(r["step_p90_ms"]))

    pairs = []
    source_speedups_cold = []
    source_speedups_warm = []
    per_lang_chrf = defaultdict(list)
    all_chrf = []
    batched_rows = []
    seq_rows = []
    warm_rows = []
    source_speedup_by_id = {}

    for sid in sorted(by_source):
        b_rows = by_source[sid].get("batched", [])
        s_rows = by_source[sid].get("seq", [])
        w_rows = by_source[sid].get("warm", [])
        batched_rows.extend(b_rows)
        seq_rows.extend(s_rows)
        warm_rows.extend(w_rows)
        if not b_rows or not s_rows:
            continue
        b_by = {r["lang"]: r for r in b_rows}
        s_by = {r["lang"]: r for r in s_rows}
        w_by = {r["lang"]: r for r in w_rows}

        if (sid, "batched") in runs:
            b_wall = runs[(sid, "batched")][0]
        else:
            b_wall = max(float(r["wall_ms"]) for r in b_rows)
        if (sid, "seq") in runs:
            s_cold = runs[(sid, "seq")][0]
        else:
            s_cold = sum(float(r["wall_ms"]) for r in s_rows)
        w_wall = None
        if (sid, "warm") in runs:
            w_wall = runs[(sid, "warm")][0]
        elif w_rows:
            w_wall = sum(float(r["wall_ms"]) for r in w_rows)
        if b_wall <= 0:
            continue
        speedup_cold = s_cold / b_wall
        speedup_warm = (w_wall / b_wall) if w_wall and w_wall > 0 else None
        source_speedups_cold.append(speedup_cold)
        if speedup_warm is not None:
            source_speedups_warm.append(speedup_warm)
        source_speedup_by_id[sid] = (speedup_cold, speedup_warm)

        for lang in order:
            if lang not in b_by or lang not in s_by:
                continue
            b = b_by[lang]
            s = s_by[lang]
            w = w_by.get(lang)
            b_text = unescape_text(b["text"])
            s_text = unescape_text(s["text"])
            w_text = unescape_text(w["text"]) if w else ""
            score = chrf(b_text, s_text)
            chk = script_check(lang, b_text)
            s_chk = script_check(lang, s_text)
            w_chk = script_check(lang, w_text) if w else {"raw_ok": False,
                                                          "corrected_ok": False,
                                                          "reason": "no warm row"}
            b_loop = repetition_stats(b_text, lang)
            s_loop = repetition_stats(s_text, lang)
            w_loop = repetition_stats(w_text, lang) if w else {"max_ngram": 0, "loop": False}
            all_chrf.append(score)
            per_lang_chrf[lang].append(score)
            pairs.append({
                "source_id": sid,
                "source_type": source_type(sid, compact_rows, gsm_rows),
                "lang": lang,
                "seq_cold_wall_ms": f"{s_cold:.3f}",
                "warm_wall_ms": f"{w_wall:.3f}" if w_wall else "",
                "batched_wall_ms": f"{b_wall:.3f}",
                "speedup_cold": f"{speedup_cold:.4f}",
                "speedup_warm": f"{speedup_warm:.4f}" if speedup_warm is not None else "",
                "lang_speedup_cold": f"{float(s['wall_ms']) / b_wall:.4f}",
                "lang_speedup_warm": (f"{float(s['wall_ms']) / float(w['wall_ms']):.4f}"
                                      if w and float(w["wall_ms"]) > 0 else ""),
                "chrf_surface": f"{score:.6f}",
                "batched_generated": b["generated"],
                "seq_generated": s["generated"],
                "warm_generated": w["generated"] if w else "",
                "batched_finish": b["finish"],
                "seq_finish": s["finish"],
                "warm_finish": w["finish"] if w else "",
                "batched_ttft_ms": b["ttft_ms"],
                "batched_done_ms": b["done_ms"],
                "batched_loop_max4": b_loop["max_ngram"],
                "batched_loop": "1" if b_loop["loop"] else "0",
                "batched_loop_conc": "1" if b_loop["conc_loop"] else "0",
                "seq_loop_max4": s_loop["max_ngram"],
                "seq_loop": "1" if s_loop["loop"] else "0",
                "seq_loop_conc": "1" if s_loop["conc_loop"] else "0",
                "warm_loop_max4": w_loop["max_ngram"] if w else "",
                "warm_loop": ("1" if w_loop["loop"] else "0") if w else "",
                "warm_loop_conc": ("1" if w_loop["conc_loop"] else "0") if w else "",
                "script_raw_ok": "1" if chk["raw_ok"] else "0",
                "script_corrected_ok": "1" if chk["corrected_ok"] else "0",
                "script_reason": chk["reason"],
                "seq_script_raw_ok": "1" if s_chk["raw_ok"] else "0",
                "seq_script_corrected_ok": "1" if s_chk["corrected_ok"] else "0",
                "seq_script_reason": s_chk["reason"],
                "warm_script_raw_ok": ("1" if w_chk["raw_ok"] else "0") if w else "",
                "warm_script_corrected_ok": ("1" if w_chk["corrected_ok"] else "0") if w else "",
                "refusal_heuristic": "1" if is_refusal(b_text) else "0",
                "rt_batched_chrf": "",
                "rt_seq_chrf": "",
                "rt_delta_pts": "",
            })

    if not pairs:
        sys.exit("no paired batched/seq rows found")

    pairs_path = pathlib.Path(args.pairs_out)
    pairs_path.parent.mkdir(parents=True, exist_ok=True)

    cold_median = statistics.median(source_speedups_cold)
    cold_mean = statistics.mean(source_speedups_cold)
    cold_ci_lo, cold_ci_hi = bootstrap_median(source_speedups_cold)
    if source_speedups_warm:
        warm_median = statistics.median(source_speedups_warm)
        warm_mean = statistics.mean(source_speedups_warm)
        warm_ci_lo, warm_ci_hi = bootstrap_median(source_speedups_warm)
    else:
        warm_median = warm_mean = warm_ci_lo = warm_ci_hi = float("nan")

    lang_macro = {lang: statistics.mean(vals) for lang, vals in per_lang_chrf.items()}
    macro = statistics.mean(lang_macro.values())
    overall_chrf = statistics.mean(all_chrf)
    min_lang = min(lang_macro.values())
    min_lang_name = min(lang_macro, key=lang_macro.get)

    arm_q = {
        "batched": arm_quality(batched_rows, compact_rows, gsm_rows),
        "seq": arm_quality(seq_rows, compact_rows, gsm_rows),
        "warm": arm_quality(warm_rows, compact_rows, gsm_rows),
    }
    trunc_b = arm_q["batched"]["trunc"]
    trunc_s = arm_q["seq"]["trunc"]
    loop_b = arm_q["batched"]["loop"]
    loop_s = arm_q["seq"]["loop"]

    script_raw_pass = sum(1 for p in pairs if p["script_raw_ok"] == "1") / len(pairs)
    script_corr_pass = sum(1 for p in pairs if p["script_corrected_ok"] == "1") / len(pairs)
    seq_script_raw_pass = sum(1 for p in pairs if p["seq_script_raw_ok"] == "1") / len(pairs)
    seq_script_corr_pass = sum(1 for p in pairs if p["seq_script_corrected_ok"] == "1") / len(pairs)
    warm_pairs = [p for p in pairs if p["warm_script_raw_ok"] != ""]
    warm_script_raw_pass = (sum(1 for p in warm_pairs if p["warm_script_raw_ok"] == "1") /
                            len(warm_pairs)) if warm_pairs else float("nan")
    warm_script_corr_pass = (sum(1 for p in warm_pairs if p["warm_script_corrected_ok"] == "1") /
                             len(warm_pairs)) if warm_pairs else float("nan")
    per_lang_script_raw = {
        lang: statistics.mean([1.0 if p["script_raw_ok"] == "1" else 0.0
                               for p in pairs if p["lang"] == lang])
        for lang in order
    }
    per_lang_script_corr = {
        lang: statistics.mean([1.0 if p["script_corrected_ok"] == "1" else 0.0
                               for p in pairs if p["lang"] == lang])
        for lang in order
    }
    refusal_count = sum(1 for p in pairs if p["refusal_heuristic"] == "1")
    rescued = [p for p in pairs
               if p["script_raw_ok"] == "0" and p["script_corrected_ok"] == "1"]
    rescued_by_lang = Counter(p["lang"] for p in rescued)

    homog = None
    if args.homog and pathlib.Path(args.homog).exists():
        hrows = [r for r in read_rows(args.homog) if r["finish"] != ""]
        htexts = [unescape_text(r["text"]) for r in hrows]
        if len(htexts) >= 2:
            scores = []
            for i in range(len(htexts)):
                for j in range(i + 1, len(htexts)):
                    scores.append(chrf(htexts[i], htexts[j]))
            fin = Counter(r["finish"] for r in hrows)
            homog = {
                "n": len(htexts),
                "pairs": len(scores),
                "min": min(scores),
                "median": statistics.median(scores),
                "max": max(scores),
                "cap": fin.get("cap", 0),
                "eos": fin.get("eos", 0),
                "other": sum(v for k, v in fin.items() if k not in ("cap", "eos")),
                "loops": sum(1 for r in hrows
                             if repetition_stats(unescape_text(r["text"]), r["lang"])["loop"]),
                "finish": dict(fin),
            }

    stragglers = []
    for sid in sorted(by_source):
        br = by_source[sid].get("batched", [])
        if not br:
            continue
        dones = [float(r["done_ms"]) for r in br]
        med = statistics.median(dones)
        if med > 0:
            stragglers.append(max(dones) / med)
    straggler_median = statistics.median(stragglers) if stragglers else float("nan")
    straggler_max = max(stragglers) if stragglers else float("nan")

    first_lang = order[0] if order else "fra"
    ttft = {}
    for mode in ("batched", "seq", "warm"):
        vals = []
        for sid in sorted(by_source):
            for r in by_source[sid].get(mode, []):
                if r["lang"] == first_lang:
                    vals.append(float(r["ttft_ms"]))
        ttft[mode] = statistics.median(vals) if vals else float("nan")

    p50 = {}
    for mode in ("batched", "seq", "warm"):
        vals = [v[1] for (sid, m), v in step_stats.items() if m == mode]
        p50[mode] = statistics.median(vals) if vals else float("nan")

    source_text = {}
    if pathlib.Path(args.data).exists():
        with open(args.data, "r", encoding="utf-8", newline="") as f:
            for i, line in enumerate(f, start=1):
                source_text[i] = line.rstrip("\r\n")

    roundtrip = None
    rt_note = "not run"
    rt_pairs = {}
    if args.roundtrip > 0:
        rt_note = ""
        if not (pathlib.Path(args.exe).exists() and pathlib.Path(args.model).exists()):
            rt_note = "skipped: exe or model not found"
        elif not source_text:
            rt_note = "skipped: source data not found"
        else:
            rt_note = "ok"
            candidates = [sid for sid in sorted(by_source)
                          if by_source[sid].get("batched") and by_source[sid].get("seq")]
            if args.roundtrip_select == "compact":
                preferred = [sid for sid in candidates if sid in compact_rows]
            elif args.roundtrip_select == "gsm8k":
                preferred = [sid for sid in candidates if sid in gsm_rows]
            elif args.roundtrip_select == "all":
                preferred = candidates
            else:
                compact_c = [sid for sid in candidates if sid in compact_rows]
                gsm_c = [sid for sid in candidates if sid in gsm_rows]
                preferred = []
                for i in range(max(len(compact_c), len(gsm_c))):
                    if i < len(compact_c):
                        preferred.append(compact_c[i])
                    if i < len(gsm_c):
                        preferred.append(gsm_c[i])
            chosen = preferred[:args.roundtrip]
            if len(chosen) < args.roundtrip:
                chosen += [sid for sid in candidates if sid not in chosen][:args.roundtrip - len(chosen)]
            rt_rows = []
            env = os.environ.copy()
            bin_dir = pathlib.Path(args.bin)
            if bin_dir.exists():
                env["PATH"] = str(bin_dir.resolve()) + os.pathsep + env.get("PATH", "")

            with tempfile.TemporaryDirectory(prefix="fanout_rt_") as td:
                td = pathlib.Path(td)

                def backtranslate(texts, tag):
                    data_path = td / f"{tag}.tsv"
                    out_path = td / f"{tag}.csv"
                    with open(data_path, "w", encoding="utf-8", newline="\n") as f:
                        for t in texts:
                            one = re.sub(r"\s+", " ", t).strip() or "(empty)"
                            f.write(one + "\n")
                    cmd = [
                        str(pathlib.Path(args.exe).resolve()),
                        "--model", str(pathlib.Path(args.model).resolve()),
                        "--data", str(data_path),
                        "--rows", f"1-{len(texts)}",
                        "--langs", "eng",
                        "--mode", "seq",
                        "--cap", str(args.roundtrip_cap),
                        "--out", str(out_path),
                        "--warmup", "0",
                        "--no-repeat-ngram", str(args.no_repeat_ngram),
                        "--repeat-penalty", str(args.repeat_penalty),
                    ]
                    subprocess.run(cmd, check=True, env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    out_rows = read_rows(out_path)
                    out_rows.sort(key=lambda r: int(r["source_id"]))
                    return [unescape_text(r["text"]) for r in out_rows]

                for idx, sid in enumerate(chosen):
                    b_by = {r["lang"]: r for r in by_source[sid].get("batched", [])}
                    s_by = {r["lang"]: r for r in by_source[sid].get("seq", [])}
                    langs_here = [l for l in order if l in b_by and l in s_by]
                    if not langs_here:
                        continue
                    print(f"[roundtrip] source={sid} ({idx + 1}/{len(chosen)})", file=sys.stderr)
                    b_texts = [unescape_text(b_by[l]["text"]) for l in langs_here]
                    s_texts = [unescape_text(s_by[l]["text"]) for l in langs_here]
                    b_bt = backtranslate(b_texts, f"s{sid}_b")
                    s_bt = backtranslate(s_texts, f"s{sid}_s")
                    src = source_text.get(sid, "")
                    b_scores = [chrf(src, x) for x in b_bt]
                    s_scores = [chrf(src, x) for x in s_bt]
                    b_mean = statistics.mean(b_scores)
                    s_mean = statistics.mean(s_scores)
                    for lang, b_score, s_score in zip(langs_here, b_scores, s_scores):
                        rt_pairs[(sid, lang)] = (b_score, s_score)
                    rt_rows.append({
                        "source_id": sid,
                        "batched_chrf": b_mean,
                        "seq_chrf": s_mean,
                        "delta_pts": 100.0 * (b_mean - s_mean),
                        "n_langs": len(langs_here),
                    })
            if rt_rows:
                roundtrip = {
                    "sources": rt_rows,
                    "source_ids": [r["source_id"] for r in rt_rows],
                    "batched_mean": statistics.mean(r["batched_chrf"] for r in rt_rows),
                    "seq_mean": statistics.mean(r["seq_chrf"] for r in rt_rows),
                    "delta_pts": statistics.mean(r["delta_pts"] for r in rt_rows),
                }
            else:
                rt_note = "no sources completed"

    if roundtrip is not None:
        for p in pairs:
            key = (int(p["source_id"]), p["lang"])
            if key in rt_pairs:
                b_score, s_score = rt_pairs[key]
                p["rt_batched_chrf"] = f"{b_score:.6f}"
                p["rt_seq_chrf"] = f"{s_score:.6f}"
                p["rt_delta_pts"] = f"{100.0 * (b_score - s_score):+.4f}"
        rt_by_lang = defaultdict(list)
        for (sid, lang), (b_score, s_score) in rt_pairs.items():
            rt_by_lang[lang].append((b_score, s_score))
        rt_lang = {}
        for lang, vals in rt_by_lang.items():
            b_mean = statistics.mean(v[0] for v in vals)
            s_mean = statistics.mean(v[1] for v in vals)
            rt_lang[lang] = {
                "batched": b_mean,
                "seq": s_mean,
                "delta_pts": 100.0 * (b_mean - s_mean),
                "n": len(vals),
            }
        roundtrip["lang"] = rt_lang
        roundtrip["macro_delta_pts"] = statistics.mean(
            v["delta_pts"] for v in rt_lang.values())
        roundtrip["pair_mean_delta_pts"] = statistics.mean(
            100.0 * (b - s) for b, s in rt_pairs.values())
        roundtrip["worst_lang"] = min(rt_lang, key=lambda l: rt_lang[l]["delta_pts"])
        roundtrip["worst_lang_delta"] = rt_lang[roundtrip["worst_lang"]]["delta_pts"]

    with open(pairs_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(pairs[0].keys()))
        writer.writeheader()
        writer.writerows(pairs)

    flagged = []
    for p in pairs:
        if (p["script_corrected_ok"] != "1" or p["script_raw_ok"] != "1"
                or p["batched_loop"] == "1" or p["seq_loop"] == "1"
                or p["warm_loop"] == "1" or p["refusal_heuristic"] == "1"
                or p["batched_loop_conc"] == "1" or p["seq_loop_conc"] == "1"
                or p["warm_loop_conc"] == "1"):
            flagged.append(p)
    with open(flagged_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source_id", "source_type", "lang", "script_raw_ok",
                    "script_corrected_ok", "script_reason", "refusal_heuristic",
                    "batched_finish", "batched_loop_max4", "batched_loop",
                    "batched_loop_conc",
                    "seq_finish", "seq_loop_max4", "seq_loop", "seq_loop_conc",
                    "warm_finish", "warm_loop_max4", "warm_loop", "warm_loop_conc",
                    "batched_text", "seq_text", "warm_text"])
        for p in flagged:
            sid = int(p["source_id"])
            arm_text = {}
            for mode in ("batched", "seq", "warm"):
                rows_here = {r["lang"]: r for r in by_source[sid].get(mode, [])}
                arm_text[mode] = (unescape_text(rows_here[p["lang"]]["text"])
                                  if p["lang"] in rows_here else "")
            w.writerow([sid, p["source_type"], p["lang"], p["script_raw_ok"],
                        p["script_corrected_ok"], p["script_reason"],
                        p["refusal_heuristic"],
                        p["batched_finish"], p["batched_loop_max4"], p["batched_loop"],
                        p["batched_loop_conc"],
                        p["seq_finish"], p["seq_loop_max4"], p["seq_loop"],
                        p["seq_loop_conc"],
                        p["warm_finish"], p["warm_loop_max4"], p["warm_loop"],
                        p["warm_loop_conc"],
                        arm_text["batched"], arm_text["seq"], arm_text["warm"]])

    def group_stats(ids):
        gpairs = [p for p in pairs if int(p["source_id"]) in ids]
        gspeed_cold = [source_speedup_by_id[s][0] for s in sorted(ids)
                       if s in source_speedup_by_id]
        gspeed_warm = [source_speedup_by_id[s][1] for s in sorted(ids)
                       if s in source_speedup_by_id and source_speedup_by_id[s][1] is not None]
        if not gpairs:
            return None
        langs_in = sorted({p["lang"] for p in gpairs})
        surf_macro = statistics.mean(
            statistics.mean(float(p["chrf_surface"]) for p in gpairs if p["lang"] == lang)
            for lang in langs_in)
        rt_vals = [100.0 * (b - s) for (sid, lang), (b, s) in rt_pairs.items()
                   if sid in ids]
        gb = [r for r in batched_rows if int(r["source_id"]) in ids]
        gs = [r for r in seq_rows if int(r["source_id"]) in ids]
        gw = [r for r in warm_rows if int(r["source_id"]) in ids]
        return {
            "sources": len(gspeed_cold),
            "pairs": len(gpairs),
            "speedup_cold_median": statistics.median(gspeed_cold) if gspeed_cold else float("nan"),
            "speedup_warm_median": statistics.median(gspeed_warm) if gspeed_warm else float("nan"),
            "surface_macro": surf_macro,
            "script_raw": statistics.mean(
                1.0 if p["script_raw_ok"] == "1" else 0.0 for p in gpairs),
            "script_corr": statistics.mean(
                1.0 if p["script_corrected_ok"] == "1" else 0.0 for p in gpairs),
            "trunc": statistics.mean(
                1.0 if p["batched_finish"] == "cap" else 0.0 for p in gpairs),
            "loop": statistics.mean(
                1.0 if p["batched_loop"] == "1" else 0.0 for p in gpairs),
            "trunc_arms": {
                "batched": (sum(1 for r in gb if r["finish"] == "cap") / len(gb)) if gb else float("nan"),
                "seq": (sum(1 for r in gs if r["finish"] == "cap") / len(gs)) if gs else float("nan"),
                "warm": (sum(1 for r in gw if r["finish"] == "cap") / len(gw)) if gw else float("nan"),
            },
            "loop_arms": {
                "batched": (sum(1 for r in gb if repetition_stats(unescape_text(r["text"]), r["lang"])["loop"]) / len(gb)) if gb else float("nan"),
                "seq": (sum(1 for r in gs if repetition_stats(unescape_text(r["text"]), r["lang"])["loop"]) / len(gs)) if gs else float("nan"),
                "warm": (sum(1 for r in gw if repetition_stats(unescape_text(r["text"]), r["lang"])["loop"]) / len(gw)) if gw else float("nan"),
            },
            "conc_arms": {
                "batched": (sum(1 for r in gb if repetition_stats(unescape_text(r["text"]), r["lang"])["conc_loop"]) / len(gb)) if gb else float("nan"),
                "seq": (sum(1 for r in gs if repetition_stats(unescape_text(r["text"]), r["lang"])["conc_loop"]) / len(gs)) if gs else float("nan"),
                "warm": (sum(1 for r in gw if repetition_stats(unescape_text(r["text"]), r["lang"])["conc_loop"]) / len(gw)) if gw else float("nan"),
            },
            "refusals": sum(1 for p in gpairs if p["refusal_heuristic"] == "1"),
            "rt_macro": statistics.mean(rt_vals) if rt_vals else None,
            "rt_n": len(rt_vals),
        }

    groups = {
        "compact": group_stats(compact_rows),
        "gsm8k": group_stats(gsm_rows),
    }

    n1_list = []
    for n1_arg in [p for p in args.n1.split(",") if p]:
        if not pathlib.Path(n1_arg).exists():
            continue
        nrows = read_rows(n1_arg)
        nb = {r["lang"]: r for r in nrows if r["mode"] == "batched"}
        ns = {r["lang"]: r for r in nrows if r["mode"] == "seq"}
        nw = {r["lang"]: r for r in nrows if r["mode"] == "warm"}
        shared = sorted(set(nb) & set(ns))
        if not shared:
            continue
        scores = [chrf(unescape_text(nb[l]["text"]), unescape_text(ns[l]["text"]))
                  for l in shared]
        warm_scores = [chrf(unescape_text(nb[l]["text"]), unescape_text(nw[l]["text"]))
                       for l in shared if l in nw]
        identity = [unescape_text(nb[l]["text"]) == unescape_text(nw[l]["text"])
                    for l in shared if l in nw]
        n1_runs = {}
        n1_runs_path = pathlib.Path(str(n1_arg) + ".runs.csv")
        if n1_runs_path.exists():
            for r in read_rows(n1_runs_path):
                n1_runs[(int(r["source_id"]), r["mode"])] = float(r["wall_all_ms"])
        sid = int(next(iter(nb.values()))["source_id"])
        bwall = n1_runs.get((sid, "batched"))
        swall = n1_runs.get((sid, "seq"))
        wwall = n1_runs.get((sid, "warm"))
        n1_list.append({
            "path": str(n1_arg),
            "source_id": sid,
            "langs": len(shared),
            "chrf": statistics.mean(scores),
            "min_chrf": min(scores),
            "speedup_cold": (swall / bwall) if bwall and swall else None,
            "speedup_warm": (wwall / bwall) if bwall and wwall else None,
            "warm_chrf": statistics.mean(warm_scores) if warm_scores else None,
            "warm_identity": (sum(identity) / len(identity)) if identity else None,
            "loops_batched": sum(1 for l in shared
                                 if repetition_stats(unescape_text(nb[l]["text"]), l)["loop"]),
            "loops_seq": sum(1 for l in shared
                             if repetition_stats(unescape_text(ns[l]["text"]), l)["loop"]),
            "loops_warm": sum(1 for l in shared if l in nw
                              and repetition_stats(unescape_text(nw[l]["text"]), l)["loop"]),
        })
    n1 = n1_list[0] if n1_list else None

    compare = None
    if args.compare_raw and pathlib.Path(args.compare_raw).exists():
        crows = read_rows(args.compare_raw)
        cby = defaultdict(list)
        for r in crows:
            cby[r["mode"]].append(r)
        compare = {
            "label": args.compare_label,
            "path": str(args.compare_raw),
            "arms": {mode: arm_quality(sub, compact_rows, gsm_rows)
                     for mode, sub in cby.items()},
        }

    gates = []
    gates.append(("speedup cold median >= 2.5x", cold_median >= 2.5))
    gates.append(("speedup cold CI lower >= 2.0x", cold_ci_lo >= 2.0))
    gates.append(("speedup warm median >= 1.8x (pure decode batching)",
                  None if not source_speedups_warm else warm_median >= 1.8))
    gates.append(("speedup warm CI lower >= 1.5x",
                  None if not source_speedups_warm else warm_ci_lo >= 1.5))
    gates.append(("truncation batched <= 5% overall", trunc_b <= 0.05))
    gates.append(("truncation batched <= 10% compact", arm_q["batched"]["trunc_compact"] <= 0.10))
    gates.append(("truncation batched <= 10% gsm8k", arm_q["batched"]["trunc_gsm8k"] <= 0.10))
    gates.append(("repetition-loop rate batched <= 2% (literal 4-gram>=3)", loop_b <= 0.02))
    gates.append(("SUPPLEMENTARY concentrated-loop rate batched <= 2% (not pre-registered)",
                  arm_q["batched"]["conc"] <= 0.02))
    if roundtrip is not None:
        gates.append(("round-trip macro delta >= -2 pts", roundtrip["macro_delta_pts"] >= -2.0))
        gates.append((f"round-trip no language <= -5 pts (worst "
                      f"{roundtrip['worst_lang']} {roundtrip['worst_lang_delta']:+.2f})",
                      roundtrip["worst_lang_delta"] > -5.0))
    else:
        gates.append(("round-trip macro delta >= -2 pts", None))
        gates.append(("round-trip no language <= -5 pts", None))
    gates.append(("script check raw (frozen) pass >= 95%", script_raw_pass >= 0.95))

    lines = []
    lines.append(f"FANOUT SUMMARY — {args.label}")
    lines.append("=" * 78)
    lines.append(f"raw:       {raw_path}")
    lines.append(f"runs:      {runs_path}")
    lines.append(f"steps:     {steps_path}")
    lines.append(f"homog:     {args.homog or '(none)'}")
    lines.append(f"pairs:     {pairs_path}")
    lines.append(f"sources:   {len(source_speedups_cold)}")
    lines.append(f"languages: {', '.join(f'{l} ({LANG_NAMES.get(l, l)})' for l in order)}")
    lines.append(f"pairs:     {len(pairs)}")
    lines.append("")
    lines.append("SPEEDUP COLD (seq cold wall / batched wall, per source)")
    lines.append(f"  median {cold_median:.3f}x  mean {cold_mean:.3f}x  "
                 f"95% bootstrap CI [{cold_ci_lo:.3f}, {cold_ci_hi:.3f}]  "
                 f"(10k, seed 1234)")
    lines.append("  per source: " + " ".join(
        f"{sid}:{source_speedup_by_id[sid][0]:.2f}x" for sid in sorted(source_speedup_by_id)))
    lines.append("")
    lines.append("SPEEDUP WARM (warm wall / batched wall, per source)")
    lines.append("  warm = shared-prefix prefill once + per-language scratch suffix "
                 "+ single-stream decode.")
    lines.append("  Both warm and batched pay the shared prefill once, so this ratio "
                 "isolates decode batching.")
    if source_speedups_warm:
        lines.append(f"  median {warm_median:.3f}x  mean {warm_mean:.3f}x  "
                     f"95% bootstrap CI [{warm_ci_lo:.3f}, {warm_ci_hi:.3f}]  "
                     f"(10k, seed 1234)")
        lines.append("  per source: " + " ".join(
            f"{sid}:{source_speedup_by_id[sid][1]:.2f}x"
            for sid in sorted(source_speedup_by_id)
            if source_speedup_by_id[sid][1] is not None))
    else:
        lines.append("  no warm rows")
    lines.append("")
    lines.append("PER-ARM QUALITY")
    lines.append("  arm      rows  trunc%  compact  gsm8k   loop%  loopC%  script_raw%  script_corr%")
    for mode in ("batched", "seq", "warm"):
        q = arm_q[mode]
        if q is None:
            lines.append(f"  {mode:<8} 0")
            continue
        lines.append(f"  {mode:<8} {q['n']:4d}  {100.0 * q['trunc']:5.1f}  "
                     f"{100.0 * q['trunc_compact']:6.1f}  {100.0 * q['trunc_gsm8k']:5.1f}  "
                     f"{100.0 * q['loop']:5.1f}  {100.0 * q['conc']:5.1f}  "
                     f"{100.0 * q['script_raw']:10.1f}  "
                     f"{100.0 * q['script_corrected']:11.1f}")
    lines.append("  loop  = any word 4-gram (char 4-gram for zho/jpn) repeated >= 3 times "
                 "(literal Amendment 2 definition).")
    lines.append("  loopC = supplementary: same 4-gram repeated >= 3 times inside a 40-unit "
                 "window (degenerate back-to-back loop).")
    lines.append("")
    lines.append("PRIMARY PARITY — ROUND-TRIP DELTA (back-translate each arm to English, chrF vs source)")
    if roundtrip is not None:
        lines.append("  sources: " + ", ".join(str(s) for s in roundtrip["source_ids"]))
        lines.append("  back-translation: seq greedy, same model, cap "
                     f"{args.roundtrip_cap}, no-repeat-ngram {args.no_repeat_ngram}, "
                     f"repeat-penalty {args.repeat_penalty}.")
        for lang in order:
            if lang in roundtrip["lang"]:
                v = roundtrip["lang"][lang]
                lines.append(f"  {lang} ({LANG_NAMES.get(lang, lang)}): "
                             f"batched {v['batched']:.4f}  seq {v['seq']:.4f}  "
                             f"delta {v['delta_pts']:+.2f} pts  (n={v['n']})")
        lines.append(f"  macro delta (per-language, equal weight) "
                     f"{roundtrip['macro_delta_pts']:+.2f} pts  <-- primary")
        lines.append(f"  mean delta over (source, language) pairs "
                     f"{roundtrip['pair_mean_delta_pts']:+.2f} pts")
        lines.append(f"  mean delta over sources {roundtrip['delta_pts']:+.2f} pts")
        for r in roundtrip["sources"]:
            lines.append(f"  source {r['source_id']}: batched {r['batched_chrf']:.4f}  "
                         f"seq {r['seq_chrf']:.4f}  delta {r['delta_pts']:+.2f} pts")
    else:
        lines.append(f"  {rt_note}")
    lines.append("")
    lines.append("SURFACE PARITY (character 3-gram F1, batched vs seq)")
    lines.append("  Label: surface parity, NOT a quality metric for translation.")
    lines.append("  Two valid translations of the same source can diverge and score low here.")
    lines.append(f"  macro {macro:.4f}  overall {overall_chrf:.4f}  "
                 f"min language {min_lang_name}={min_lang:.4f}")
    for lang in order:
        if lang in lang_macro:
            lines.append(f"  {lang} ({LANG_NAMES.get(lang, lang)}): {lang_macro[lang]:.4f}")
    lines.append("")
    lines.append("SCRIPT / LANGUAGE CHECK (batched outputs)")
    lines.append(f"  raw frozen check:       {100.0 * script_raw_pass:.1f}%  "
                 f"(seq control {100.0 * seq_script_raw_pass:.1f}%, "
                 f"warm control {100.0 * warm_script_raw_pass:.1f}%)")
    lines.append(f"  corrected check:        {100.0 * script_corr_pass:.1f}%  "
                 f"(seq control {100.0 * seq_script_corr_pass:.1f}%, "
                 f"warm control {100.0 * warm_script_corr_pass:.1f}%)")
    lines.append("  Correction (Latin only): pass if >=1 target stopword OR "
                 "(<=2 English function words AND >=6 words).")
    lines.append("  Reported separately; the frozen gate still uses the raw check.")
    if rescued:
        lines.append(f"  correction rescued {len(rescued)} raw failures ("
                     + ", ".join(f"{k}={v}" for k, v in sorted(rescued_by_lang.items()))
                     + "); inspect flagged outputs.")
    for lang in order:
        if lang in per_lang_script_raw:
            lines.append(f"  {lang} ({LANG_NAMES.get(lang, lang)}): "
                         f"raw {100.0 * per_lang_script_raw[lang]:.1f}%  "
                         f"corrected {100.0 * per_lang_script_corr[lang]:.1f}%")
    lines.append(f"  refusals (heuristic):   {refusal_count}/{len(pairs)} = "
                 f"{100.0 * refusal_count / len(pairs):.1f}%")
    lines.append(f"  flagged outputs (manual review): {flagged_path}")
    lines.append("")
    if homog is not None:
        lines.append("HOMOGENEOUS CONTROL (8 identical French branches, source 1)")
        lines.append(f"  pairwise chrF n={homog['pairs']} min {homog['min']:.4f} "
                     f"median {homog['median']:.4f} max {homog['max']:.4f}")
        lines.append(f"  finish: eos={homog['eos']} cap={homog['cap']} other={homog['other']}  "
                     f"loops={homog['loops']}  n={homog['n']}")
        lines.append("  Reported as the batch-decode noise floor; NOT gated on byte identity "
                     "(Amendment 2).")
    else:
        lines.append("HOMOGENEOUS CONTROL: not run")
    lines.append("")
    if n1_list:
        lines.append("N=1 DIAGNOSTICS (one branch; batched vs seq vs warm)")
        for item in n1_list:
            extra = ""
            if item["speedup_cold"] is not None:
                extra += f" cold speedup {item['speedup_cold']:.3f}x"
            if item["speedup_warm"] is not None:
                extra += f" warm speedup {item['speedup_warm']:.3f}x"
            if item["warm_chrf"] is not None:
                extra += (f" warm-vs-batched chrF {item['warm_chrf']:.4f}"
                          f" identity {100.0 * item['warm_identity']:.0f}%")
            lines.append(f"  source {item['source_id']} langs={item['langs']} "
                         f"{item['path']}: surface chrF mean {item['chrf']:.4f} "
                         f"min {item['min_chrf']:.4f}{extra}; "
                         f"loops b/s/w {item['loops_batched']}/"
                         f"{item['loops_seq']}/{item['loops_warm']}")
    else:
        lines.append("N=1 DIAGNOSTIC: not available")
    lines.append("")
    lines.append("PER-SOURCE (speedups, batched trunc/loop, script raw)")
    lines.append("  id  type     cold    warm    trunc%  loop%  loopC%  script_raw%  surface")
    for sid in sorted(source_speedup_by_id):
        cold, warm = source_speedup_by_id[sid]
        bsub = [r for r in batched_rows if int(r["source_id"]) == sid]
        psub = [p for p in pairs if int(p["source_id"]) == sid]
        if not bsub or not psub:
            continue
        tr = 100.0 * sum(1 for r in bsub if r["finish"] == "cap") / len(bsub)
        st = [repetition_stats(unescape_text(r["text"]), r["lang"]) for r in bsub]
        lp = 100.0 * sum(1 for s in st if s["loop"]) / len(bsub)
        lc = 100.0 * sum(1 for s in st if s["conc_loop"]) / len(bsub)
        sr = 100.0 * sum(1 for p in psub if p["script_raw_ok"] == "1") / len(psub)
        sf = statistics.mean(float(p["chrf_surface"]) for p in psub)
        wcol = f"{warm:5.2f}x" if warm is not None else "   - "
        lines.append(f"  {sid:3d} {source_type(sid, compact_rows, gsm_rows):<8} "
                     f"{cold:5.2f}x  {wcol}  {tr:5.1f}  {lp:5.1f}  {lc:5.1f}  "
                     f"{sr:10.1f}  {sf:.4f}")
    lines.append("")
    lines.append("SOURCE-TYPE SPLIT")
    for name in ("compact", "gsm8k"):
        g = groups.get(name)
        if g is None:
            lines.append(f"  {name}: no paired rows")
            continue
        rt = (f"{g['rt_macro']:+.2f} pts over {g['rt_n']} (source,language) pairs"
              if g["rt_macro"] is not None else "not run")
        ta = g["trunc_arms"]
        la = g["loop_arms"]
        ca = g["conc_arms"]
        lines.append(f"  {name}: sources {g['sources']} pairs {g['pairs']}  "
                     f"speedup cold median {g['speedup_cold_median']:.3f}x  "
                     f"warm median {g['speedup_warm_median']:.3f}x")
        lines.append(f"           trunc b/s/w {100.0 * ta['batched']:.1f}/"
                     f"{100.0 * ta['seq']:.1f}/{100.0 * ta['warm']:.1f}%  "
                     f"loop b/s/w {100.0 * la['batched']:.1f}/"
                     f"{100.0 * la['seq']:.1f}/{100.0 * la['warm']:.1f}%  "
                     f"loopC b/s/w {100.0 * ca['batched']:.1f}/"
                     f"{100.0 * ca['seq']:.1f}/{100.0 * ca['warm']:.1f}%")
        lines.append(f"           surface macro {g['surface_macro']:.4f}  "
                     f"script raw {100.0 * g['script_raw']:.1f}% / "
                     f"corrected {100.0 * g['script_corr']:.1f}%  "
                     f"refusals {g['refusals']}  round-trip {rt}")
    lines.append("")
    lines.append("OTHER")
    lines.append(f"  straggler ratio (max/median done_ms, batched): "
                 f"median {straggler_median:.2f}x max {straggler_max:.2f}x")
    for mode in ("batched", "seq", "warm"):
        note = (" (per-language; excludes the shared-prefix prefill)"
                if mode == "warm" else "")
        lines.append(f"  TTFT {first_lang} ({LANG_NAMES.get(first_lang, first_lang)}) "
                     f"{mode}: median {ttft[mode]:.1f} ms{note}")
    for mode in ("batched", "seq", "warm"):
        if not math.isnan(p50[mode]):
            lines.append(f"  step p50 ({mode}, median across sources): {p50[mode]:.2f} ms")
    lines.append("")
    if compare is not None:
        lines.append(f"BEFORE/AFTER (same metric definitions; before = {compare['label']})")
        lines.append(f"  before raw: {compare['path']}")
        lines.append("  arm      before_trunc%  after_trunc%  before_loop%  after_loop%  "
                     "before_loopC%  after_loopC%")
        for mode in ("batched", "seq", "warm"):
            before = compare["arms"].get(mode)
            after = arm_q.get(mode)
            if before is None and after is None:
                continue
            bt = 100.0 * before["trunc"] if before else float("nan")
            at = 100.0 * after["trunc"] if after else float("nan")
            bl = 100.0 * before["loop"] if before else float("nan")
            al = 100.0 * after["loop"] if after else float("nan")
            bc = 100.0 * before["conc"] if before else float("nan")
            ac = 100.0 * after["conc"] if after else float("nan")
            lines.append(f"  {mode:<8} {bt:12.1f}  {at:12.1f}  {bl:12.1f}  {al:11.1f}  "
                         f"{bc:12.1f}  {ac:11.1f}")
        lines.append("  by source type (batched trunc% / loop% / loopC%):")
        for name in ("compact", "gsm8k"):
            before = compare["arms"].get("batched")
            after = arm_q.get("batched")
            btf = before[f"trunc_{name}"] if before else float("nan")
            atf = after[f"trunc_{name}"] if after else float("nan")
            blf = before[f"loop_{name}"] if before else float("nan")
            alf = after[f"loop_{name}"] if after else float("nan")
            bcf = before[f"conc_{name}"] if before else float("nan")
            acf = after[f"conc_{name}"] if after else float("nan")
            lines.append(f"    {name}: before {100.0 * btf:.1f}/{100.0 * blf:.1f}/"
                         f"{100.0 * bcf:.1f}  after {100.0 * atf:.1f}/{100.0 * alf:.1f}/"
                         f"{100.0 * acf:.1f}")
        lines.append("")
    lines.append("GATES (Amendment 2)")
    for name, ok in gates:
        status = "PASS" if ok is True else ("FAIL" if ok is False else "NO DATA")
        lines.append(f"  [{status}] {name}")
    lines.append("")
    lines.append("CARRIED-OVER FROZEN GATES (original thresholds unchanged)")
    lines.append(f"  [{'PASS' if macro >= 0.95 else 'FAIL'}] surface parity macro >= 0.95 "
                 f"(observed {macro:.4f})")
    lines.append(f"  [{'PASS' if min_lang >= 0.90 else 'FAIL'}] surface: no language < 0.90 "
                 f"(min {min_lang_name}={min_lang:.4f})")
    lines.append(f"  homogeneous control gate >= 0.95 replaced by reported noise floor "
                 f"in Amendment 2"
                 + (f" (observed min {homog['min']:.4f})" if homog else " (not run)"))
    lines.append("")
    lines.append("Observed on one RTX 4060 8 GB only. No extrapolation.")

    summary_path = pathlib.Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")

    spot_path = pathlib.Path(args.spotcheck)
    spot_path.parent.mkdir(parents=True, exist_ok=True)

    def source_surface_mean(sid):
        vals = [float(p["chrf_surface"]) for p in pairs if int(p["source_id"]) == sid]
        return statistics.mean(vals) if vals else 0.0

    chosen_ids = []
    if source_speedup_by_id:
        compact_ids = [s for s in sorted(source_speedup_by_id) if s in compact_rows]
        gsm_ids = [s for s in sorted(source_speedup_by_id) if s in gsm_rows]
        if compact_ids:
            chosen_ids.append(min(compact_ids, key=source_surface_mean))
        if gsm_ids:
            chosen_ids.append(min(gsm_ids, key=source_surface_mean))
        if not chosen_ids:
            chosen_ids.append(min(source_speedup_by_id, key=source_surface_mean))
    with open(spot_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"# Fan-out spot-check — {args.label}\n\n")
        f.write("Surface chrF is parity between two valid translations, not quality. "
                "Loop = a word 4-gram (char 4-gram for zho/jpn) repeated >= 3 times.\n\n")
        for sid in chosen_ids:
            f.write(f"## Source {sid} ({source_type(sid, compact_rows, gsm_rows)})\n\n")
            snippet = source_text.get(sid, "")
            if snippet:
                f.write(f"Source text (first 240 chars): `{snippet[:240]}`\n\n")
            f.write("| lang | surface | script b raw/corr | loop max4 b/s/w | gen b/s/w | finish b/s/w | batched | seq | warm |\n")
            f.write("|---|---:|---|---|---|---|---|---|---|\n")
            b_by = {r["lang"]: r for r in by_source[sid].get("batched", [])}
            s_by = {r["lang"]: r for r in by_source[sid].get("seq", [])}
            w_by = {r["lang"]: r for r in by_source[sid].get("warm", [])}
            for lang in order:
                if lang not in b_by or lang not in s_by:
                    continue
                b = b_by[lang]
                s = s_by[lang]
                w = w_by.get(lang)
                bt = unescape_text(b["text"]).replace("|", "\\|").replace("\n", "<br>")
                st = unescape_text(s["text"]).replace("|", "\\|").replace("\n", "<br>")
                wt = (unescape_text(w["text"]).replace("|", "\\|").replace("\n", "<br>")
                      if w else "")
                score = chrf(unescape_text(b["text"]), unescape_text(s["text"]))
                chk = script_check(lang, unescape_text(b["text"]))
                lb = repetition_stats(unescape_text(b["text"]), lang)["max_ngram"]
                ls = repetition_stats(unescape_text(s["text"]), lang)["max_ngram"]
                lw = repetition_stats(unescape_text(w["text"]), lang)["max_ngram"] if w else ""
                f.write(f"| {lang} | {score:.4f} | "
                        f"{'ok' if chk['raw_ok'] else 'FAIL'}/"
                        f"{'ok' if chk['corrected_ok'] else 'FAIL'} ({chk['reason']}) | "
                        f"{lb}/{ls}/{lw} | "
                        f"{b['generated']}/{s['generated']}/{w['generated'] if w else ''} | "
                        f"{b['finish']}/{s['finish']}/{w['finish'] if w else ''} | "
                        f"{bt} | {st} | {wt} |\n")
            f.write("\n")

    print("\n".join(lines))
    print(f"\nwrote {pairs_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {spot_path}")
    print(f"wrote {flagged_path}")


if __name__ == "__main__":
    main()
