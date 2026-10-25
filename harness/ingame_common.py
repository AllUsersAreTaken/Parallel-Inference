"""Shared helpers for the in-game memory harnesses (angles 27+).

Extracted from run_ingame_memory.py for the angle-32+ harnesses so the old
harnesses (and their angle-27..31 raws) stay untouched. Differences vs the
original run_cli (reviewer debt):

- subprocess stdout is decoded with encoding="utf-8", errors="replace"
  (run_ingame_memory.py:77 used locale encoding -> mojibake in raw JSON
  text/prompt fields; identity comparisons remain valid because both arms
  were mangled by the same deterministic code path).
- cli_provenance() returns the parallel_cli.exe SHA256 + a dirty-tree note
  for the metadata block of every results JSON (reviewer M2).
- s0_identity() re-adds the Slot 0 text-identity + generated-token columns
  that angle 28+ dropped (reviewer M1).
"""
import hashlib
import json
import os
import statistics
import subprocess
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "app" / "parallel_cli.exe"
DLL_DIR = ROOT / "build" / "cuda-graphs" / "bin"
MODEL = "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
SYSTEM = "You are a helpful assistant."
SEEDS = [1234, 1235, 1236, 1237, 1238]
MAX_TOKENS = 160
TEMP = 0.2
TOP_P = 0.9
S0_MARKER = "Reply to the user"
S1_PROFILE_MARKER = "[User Profile]"
PROFILE_TOKEN_MAX = 75

# Angle-31 compaction rule (anti-bleed sentence included). Byte-identical to
# the instruction tail of data/ingame_bleed_batch.tsv's s1 prompt.
COMPACT_RULE = (
    "You maintain the active [User Profile]. Incorporate new facts from the "
    "latest conversation turn into the profile while strictly removing "
    "superseded or obsolete lines. Do not carry over adjectives, years, or "
    "specs from replaced items unless explicitly stated. Keep the profile "
    "under 60 words as a clean bulleted list. Output the updated block directly."
)
S0_INSTRUCTION = "Reply to the user in 2\u20133 warm, concise sentences."
SEED_PROFILE = (
    "- Location: Seattle\n"
    "- Pet: Buster (golden retriever, knee issues)\n"
    "- Vehicle: 2019 Honda Civic\n"
    "- Commute: 25 minutes highway"
)


def run_cli(mode, tsv, seed, out_path, max_tokens=None):
    """Run parallel_cli.exe; decode stdout as UTF-8 (fixes mojibake).

    max_tokens overrides the module default per call (the k3_fix harness
    uses 256 for Slot 1's folded budget without changing other harnesses).
    """
    env = dict(os.environ)
    env["PATH"] = str(DLL_DIR) + os.pathsep + env.get("PATH", "")
    cmd = [
        str(CLI), "--model", MODEL, "--data", tsv, "--out", str(out_path),
        "--mode", mode, "--max-tokens", str(MAX_TOKENS if max_tokens is None
                                             else max_tokens),
        "--temperature", str(TEMP), "--top-p", str(TOP_P),
        "--seed", str(seed), "--system", SYSTEM,
    ]
    r = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True,
                       text=True, encoding="utf-8", errors="replace",
                       timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"CLI failed ({mode} seed={seed}): {r.stderr.strip()}")
    return json.loads(r.stdout)


def get_stream(doc, marker):
    arr = doc.get("streams") if doc.get("mode") == "batch" else doc.get("serial")
    for s in arr:
        if marker in s["prompt"]:
            return s
    raise KeyError(f"stream with marker {marker!r} not found")


def summarize_branch(b):
    return {k: b[k] for k in ("text", "generated", "ttft_ms", "completion_ms",
                              "tok_s", "finish")}


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def cli_provenance():
    """SHA256 + build mtime + dirty-tree note for results JSON metadata."""
    h = hashlib.sha256(CLI.read_bytes()).hexdigest()
    import datetime
    built = datetime.datetime.fromtimestamp(
        CLI.stat().st_mtime).isoformat(timespec="seconds")
    return {
        "cli": "app/parallel_cli.exe",
        "cli_sha256": h,
        "cli_built_mtime": built,
        "git_tree": "dirty",
        "git_tree_note": (
            "exe built from dirty tree: parallel/src/runtime.cpp, "
            "parallel/include/parallel/types.hpp and scripts/build_tool.bat "
            "have uncommitted modifications; app/ is untracked. Same binary "
            "for all arms within each run, so comparisons are internally "
            "valid but not reproducible from HEAD."
        ),
    }


def first_divergence_word(a, b):
    """Index of first differing whitespace token; None if identical."""
    aw, bw = a.split(), b.split()
    for i, (x, y) in enumerate(zip(aw, bw)):
        if x != y:
            return i
    if len(aw) != len(bw):
        return min(len(aw), len(bw))
    return None


def s0_identity(solo_text, batch_text, solo_gen, batch_gen):
    """Slot 0 solo-vs-batch identity record (reviewer M1 columns)."""
    identical = solo_text == batch_text
    return {
        "s0_identical": identical,
        "s0_solo_generated": solo_gen,
        "s0_batch_generated": batch_gen,
        "s0_gen_delta": batch_gen - solo_gen,
        "s0_similarity": SequenceMatcher(None, solo_text, batch_text).ratio(),
        "s0_first_divergence_word": (None if identical
                                     else first_divergence_word(solo_text, batch_text)),
        "s0_mojibake_in_either": ("\ufffd" in solo_text or "\ufffd" in batch_text),
    }


def tsv_escape(prompt):
    """Escape a prompt for the CLI's TSV reader (literal \\n / \\t)."""
    return prompt.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t")


def write_tsv(path, rows):
    """Write an id\\tprompt TSV the way the historical data files look."""
    lines = [f"{rid}\t{tsv_escape(prompt)}" for rid, prompt in rows]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def load_raw(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
